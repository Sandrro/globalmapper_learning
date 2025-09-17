#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — обучение граф-генератора на иерархической канонизации

Что добавлено:
- Поддержка YAML-конфига через --config (CLI-параметры перекрывают YAML)
- Логгер с этапами обучения + прогресс-бары tqdm
- TensorBoard (./runs/<exp>_<timestamp>) — логи лоссов/метрик
- Выгрузка артефактов на Hugging Face Hub (repo_id, token)
- Совместимость с PyTorch Geometric (если установлен), иначе fallback на простой MLP по узлам
- Обработка новых признаков из transform.py: zone label, e_i, pos/size/phi/s_i/a_i,
  floors_num/has_floors/is_living/living_area, services_present (one-hot), services_capacity (суммарные по словарю)

Ожидаемая структура датасета (из transform.py):
  data_dir/
    blocks.parquet [block_id, zone, scale_l, mask_path]
    branches.parquet [block_id, branch_local_id, length]
    nodes_fixed.parquet [block_id, slot_id, e_i, branch_local_id, posx, posy,
      size_x, size_y, phi_resid, s_i, a_i, floors_num, living_area, is_living,
      has_floors, services_present, services_capacity, aspect_ratio]
    edges.parquet [block_id, src_slot, dst_slot]

Пример:
  python train.py --config ./train_gnn.yaml
"""
from __future__ import annotations
import os, sys, json, math, time, argparse, logging, random
from datetime import datetime
from typing import Dict, Any, List, Tuple

from pyproj import Transformer

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# --- логгер ---
log = logging.getLogger("train")

def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# --- зависимости ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from shapely.geometry import shape as shp_shape, mapping as shp_mapping, Polygon as ShpPolygon, Point as ShpPoint
from shapely.affinity import rotate as shp_rotate, scale as shp_scale, translate as shp_translate
from shapely.ops import unary_union
import math as _math
from PIL import Image, ImageDraw

try:
    from sklearn.cluster import KMeans  # опционально
    _SK_OK = True
except Exception:
    _SK_OK = False

from torch.utils.data import Sampler

def _quantile_bins(x: np.ndarray, nbins: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    qs = np.nanquantile(x, np.linspace(0, 1, nbins + 1))
    # сделаем края уникальными
    qs = np.unique(qs)
    if len(qs) <= 2:
        return np.zeros_like(x, dtype=int)
    bins = np.digitize(x, qs[1:-1], right=True)
    return np.clip(bins, 0, len(qs) - 2).astype(int)

try:
    import yaml  # pyyaml
except Exception as e:
    yaml = None

# PyG опционально
_PYG_OK = False
try:
    from torch_geometric.data import Data as GeomData
    from torch_geometric.loader import DataLoader as GeomLoader
    from torch_geometric.nn import GraphSAGE
    _PYG_OK = True
except Exception:
    _PYG_OK = False

# HuggingFace Hub опционально
_HF_OK = False
try:
    from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder
    _HF_OK = True
except Exception:
    _HF_OK = False

# ----------------------
# Конфиг через YAML + CLI
# ----------------------
_DEF_CFG: Dict[str, Any] = {
    "experiment": "graphgen_hcanon_v1",
    "paths": {
        "data_dir": "./dataset",
        "model_ckpt": "./dataset/model_graphgen.pt",
        "zones_json": "./dataset/zones.json",
        "services_json": "./dataset/services.json",
        "mask_dir": "./out/masks",              # <— НОВОЕ
    },
    "model": {
        "emb_dim": 128,
        "hidden": 256,
        "use_mask_cnn": True,                   # <— НОВОЕ
        "mask_dim": 64,                         # <— НОВОЕ
        "mask_size": 128,                       # <— НОВОЕ (квадрат)
    },
    "training": {
        "batch_size": 8,
        "epochs": 50,
        "lr": 2.0e-4,
        "device": "cuda",
        "loss_weights": {
            "e": 1.0, "pos": 2.0, "sz": 2.0, "phi": 0.5,
            "s": 0.5, "a": 0.2, "fl": 0.5, "hf": 0.2,
            "il": 0.2, "la": 1.0, "sv1": 0.5, "svc": 1.0, "coll": 0.5
        }
    },
    "sampler": {
        "mode": "two_stream",
        "batch_size": 8,
        "majority_frac": 0.5,
        "tau_majority": 0.8,
        "use_residential_stratify": True,
        "use_residential_clusters": True,
        "epoch_len_batches": None
    },
    "hf": {
        "push": False,
        "repo_id": None,
        "token": None,
        "private": True,
    }
}

# предварительный парсер только для --config
_p0 = argparse.ArgumentParser(add_help=False)
_p0.add_argument("--config", type=str, default=None)
_cfg_args, _ = _p0.parse_known_args()

_CFG = _DEF_CFG.copy()
if _cfg_args.config and yaml is not None:
    with open(_cfg_args.config, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    # рекурсивный мердж
    def _merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                _merge(a[k], v)
            else:
                a[k] = v
    _merge(_CFG, user_cfg)

# --- основной парсер
parser = argparse.ArgumentParser(parents=[_p0])

# paths & training basics
parser.add_argument("--data-dir", default=_CFG["paths"]["data_dir"])
parser.add_argument("--mask-root", default=_CFG["paths"].get("mask_dir", "./out/masks"),
                    help="Каталог с масками кварталов (mask_root/block_id.png)")
parser.add_argument("--zones-json", default=_CFG["paths"]["zones_json"])
parser.add_argument("--services-json", default=_CFG["paths"]["services_json"])
parser.add_argument("--model-ckpt", default=_CFG["paths"]["model_ckpt"])
parser.add_argument("--batch-size", type=int, default=_CFG["training"]["batch_size"])
parser.add_argument("--epochs", type=int, default=_CFG["training"]["epochs"])
parser.add_argument("--lr", type=float, default=_CFG["training"]["lr"])
parser.add_argument("--device", default=_CFG["training"]["device"])
parser.add_argument("--mode", choices=["train", "infer"], default="train")
parser.add_argument("--infer-geojson-in")
parser.add_argument("--infer-out")

# sampler
parser.add_argument("--sampler-mode", choices=["two_stream","vanilla"], default=_CFG["sampler"]["mode"])
parser.add_argument("--majority-frac", type=float, default=_CFG["sampler"]["majority_frac"])
parser.add_argument("--tau-majority", type=float, default=_CFG["sampler"]["tau_majority"])
parser.add_argument("--no-res-stratify", action="store_true", help="Отключить стратификацию внутри residential")
parser.add_argument("--no-res-clusters", action="store_true", help="Отключить k-means кластеры внутри residential")

# --- ВАЖНО: инференс-параметры ДОЛЖНЫ быть ДО parse_args() ---
parser.add_argument("--zone", type=str, help="Функциональная зона квартала (label из zones.json)")
parser.add_argument("--la-target", type=float, default=None,
                    help="Целевая суммарная жилая площадь в м² (в мировых координатах)")
parser.add_argument("--services-target", type=str, default=None,
                    help="JSON-строка или путь к JSON: dict {name: capacity} или list[{name,value}]")
parser.add_argument("--infer-slots", type=int, default=256, help="Число слотов для сэмплинга в квартале")
parser.add_argument("--infer-knn", type=int, default=8, help="k для kNN-графа")
parser.add_argument("--infer-e-thr", type=float, default=0.5, help="Порог для e")
parser.add_argument("--infer-il-thr", type=float, default=0.5, help="Порог для is_living")
parser.add_argument("--infer-sv1-thr", type=float, default=0.5, help="Порог для наличия сервиса на узле")

# hf
parser.add_argument("--hf-push", action="store_true", default=_CFG.get("hf", {}).get("push", False))
parser.add_argument("--hf-repo-id", default=_CFG.get("hf", {}).get("repo_id"))
parser.add_argument("--hf-token", default=_CFG.get("hf", {}).get("token"))
parser.add_argument("--hf-private", action="store_true", default=_CFG.get("hf", {}).get("private", True))

# misc
parser.add_argument("--log-level", default="INFO")

# >>> парсим здесь <<<
args = parser.parse_args()

setup_logger(args.log_level)
log.info(f"Sampler: mode={args.sampler_mode}, batch_size={args.batch_size}, "
         f"majority_frac={args.majority_frac}, tau={args.tau_majority}, "
         f"res_stratify={not args.no_res_stratify}, res_clusters={not args.no_res_clusters}")
log.info("Запуск train.py (YAML+CLI) …")

# ----------------------
# Утилиты
# ----------------------
def log_service_presence_stats(ds: "HCanonGraphDataset"):
    """O(1): оцениваем только долю блоков, где вообще встречаются сервисы (по готовому кешу)."""
    n_blocks = max(1, len(ds.block_ids))
    n_with = len(getattr(ds, "blocks_with_services", []))
    p_blocks_with_cap = n_with / n_blocks
    log.info(f"[services] blocks_with_any_service: {p_blocks_with_cap:.4f}  (S={ds.num_services})")

# --- Target Normalizer (reversible) ---
class TargetNormalizer:
    """
    Обратимая нормализация таргетов:
      - la: log1p(living_area / (scale_l**2 + eps))
      - svc: log1p(capacity)
    Сохраняется в JSON для последующего инференса.
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)

    # ---------- living area ----------
    def encode_la(self, living_area: torch.Tensor, scale_l: torch.Tensor) -> torch.Tensor:
        # living_area, scale_l: (N,1) или broadcastable
        return torch.log1p(living_area / (scale_l**2 + self.eps))

    def decode_la(self, la_norm: torch.Tensor, scale_l: torch.Tensor) -> torch.Tensor:
        return torch.expm1(la_norm) * (scale_l**2 + self.eps)

    # ---------- services capacity ----------
    def encode_svc(self, capacity: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp_min(capacity, 0.0))

    def decode_svc(self, svc_norm: torch.Tensor) -> torch.Tensor:
        return torch.expm1(svc_norm)

    # ---------- persistence ----------
    def to_dict(self) -> dict:
        return {"version": 1, "eps": self.eps,
                "la": {"type": "log1p_div_scale2"}, "svc": {"type": "log1p"}}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetNormalizer":
        return cls(eps=float(d.get("eps", 1e-6)))


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)


def _maybe_json_to_list(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return []

def _load_mask(path: str, size: tuple[int, int] = (128, 128)) -> torch.Tensor:
    """
    Возвращает тензор (1, H, W) в диапазоне [0,1].
    Если файла нет/ошибка чтения — вернёт нули.
    """
    H, W = int(size[0]), int(size[1])
    arr = np.zeros((H, W), dtype=np.float32)
    if path and isinstance(path, str) and os.path.exists(path):
        try:
            from PIL import Image
            img = Image.open(path).convert("L")
            img = img.resize((W, H), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        except Exception:
            pass
    return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)

# ----------------------
# Utils for inference
# ---------------------
def _polygon_from_geojson(obj: dict) -> ShpPolygon:
    geom = shp_shape(obj["geometry"] if "geometry" in obj else obj)
    if geom.geom_type == "MultiPolygon":
        # берём самый большой контур как квартал
        geom = max(list(geom.geoms), key=lambda g: g.area)
    if not isinstance(geom, ShpPolygon):
        raise ValueError("Input geometry must be (Multi)Polygon")
    return geom

def _principal_angle(poly: ShpPolygon) -> float:
    """Грубая PCA-ориентация: угол (в градусах), чтобы главный размер лег по X."""
    xs, ys = poly.exterior.xy
    X = np.vstack([np.asarray(xs) - np.mean(xs), np.asarray(ys) - np.mean(ys)])
    C = X @ X.T / max(1, X.shape[1] - 1)
    eigvals, eigvecs = np.linalg.eig(C)
    v = eigvecs[:, np.argmax(eigvals)]
    angle = _math.degrees(_math.atan2(v[1], v[0]))  # к X
    return angle

def canonicalize_polygon(poly: ShpPolygon):
    """
    Канонизация полигона квартала к [0,1]^2 с PCA-ориентацией.
    Возвращает poly_can, fwd(world->can), inv(can->world), scale_l=sqrt(area_world).

    Порядок однородных преобразований строго соответствует train.py:
      world -> canonical:  T(-minx2,-miny2) @ S(s) @ R(-angle) @ T(-cx,-cy)
      canonical -> world:  T(cx,cy) @ R(angle) @ S(1/s) @ T(minx2,miny2)
    """
    scale_l = _math.sqrt(max(1e-9, float(poly.area)))
    angle = _principal_angle(poly)
    c = poly.centroid

    poly0 = shp_translate(poly, xoff=-c.x, yoff=-c.y)
    poly1 = shp_rotate(poly0, -angle, origin=(0, 0), use_radians=False)

    minx, miny, maxx, maxy = poly1.bounds
    w, h = maxx - minx, maxy - miny
    if max(w, h) < 1e-9:
        raise ValueError("Degenerate polygon: near-zero extent after PCA alignment")

    s = 1.0 / max(w, h)
    poly2 = shp_scale(poly1, xfact=s, yfact=s, origin=(0, 0))
    minx2, miny2, _, _ = poly2.bounds
    poly_can = shp_translate(poly2, xoff=-minx2, yoff=-miny2)

    def _mat_translate(dx: float, dy: float) -> np.ndarray:
        return np.array([[1.0, 0.0, float(dx)],
                         [0.0, 1.0, float(dy)],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def _mat_rotate(deg: float) -> np.ndarray:
        th = _math.radians(float(deg))
        c_, s_ = _math.cos(th), _math.sin(th)
        return np.array([[ c_, -s_, 0.0],
                         [ s_,  c_, 0.0],
                         [ 0.0, 0.0, 1.0]], dtype=np.float64)

    def _mat_scale(sx: float, sy: float) -> np.ndarray:
        return np.array([[float(sx), 0.0, 0.0],
                         [0.0, float(sy), 0.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    M    = _mat_translate(-minx2, -miny2) @ _mat_scale(s, s) @ _mat_rotate(-angle) @ _mat_translate(-c.x, -c.y)
    Minv = _mat_translate( c.x,   c.y  ) @ _mat_rotate( angle) @ _mat_scale(1.0/s, 1.0/s) @ _mat_translate(minx2, miny2)

    def _apply(M_: np.ndarray, xy) -> np.ndarray:
        v = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=np.float64)
        r = M_ @ v
        return np.array([r[0], r[1]], dtype=np.float64)

    return poly_can, (lambda p: _apply(M, p)), (lambda p: _apply(Minv, p)), float(scale_l)


def polygon_to_mask(poly_can: ShpPolygon, size=(128,128)) -> torch.Tensor:
    """(1,H,W) mask в [0,1]"""
    W, H = int(size[0]), int(size[1])
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    xs, ys = poly_can.exterior.xy
    pts = [(x*(W-1), (H-1) - y*(H-1)) for x,y in zip(xs, ys)]  # y-вверх -> y-вниз
    draw.polygon(pts, fill=255)
    arr = np.asarray(img, dtype=np.float32)/255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1,H,W)

def sample_slots_grid(poly_can: ShpPolygon, n_slots: int, jitter: float = 0.35, seed: int = 42) -> np.ndarray:
    """Простой grid+джиттер: равномерно по [0,1], фильтрация точек внутри полигона."""
    rng = np.random.default_rng(seed)
    side = max(2, int(_math.sqrt(max(1, n_slots))))
    xs = (np.arange(side)+0.5)/side
    ys = (np.arange(side)+0.5)/side
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)
    P += rng.uniform(-jitter/side, jitter/side, size=P.shape)
    P = np.clip(P, 0.0, 1.0)
    inside = np.array([poly_can.contains(ShpPoint(p[0], p[1])) for p in P])
    return P[inside]

def build_knn_edges(pts: np.ndarray, k: int = 8) -> torch.Tensor:
    """Быстрый kNN на numpy; для N<=400-800 ок."""
    if len(pts) == 0:
        return torch.zeros((2,0), dtype=torch.long)
    X = pts.astype(np.float32)
    # попарные расстояния
    d2 = np.sum(X**2, axis=1, keepdims=True) + np.sum(X**2, axis=1) - 2*X@X.T
    np.fill_diagonal(d2, np.inf)
    idx = np.argsort(d2, axis=1)[:, :max(1,k)]
    src = np.repeat(np.arange(len(X)), idx.shape[1])
    dst = idx.ravel()
    ei = np.stack([src, dst], axis=0)
    return torch.from_numpy(ei.astype(np.int64))
    
def rect_polygon(center_xy, size_xy, angle_rad):
    """Возвращает shapely.Polygon прямоугольника в *каноническом* пространстве."""
    cx, cy = float(center_xy[0]), float(center_xy[1])
    sx, sy = max(1e-6, float(size_xy[0])), max(1e-6, float(size_xy[1]))
    dx, dy = sx/2.0, sy/2.0
    corners = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]], dtype=np.float64)
    c,s = _math.cos(float(angle_rad)), _math.sin(float(angle_rad))
    R = np.array([[c,-s],[s,c]], dtype=np.float64)
    rot = (R @ corners.T).T + np.array([cx,cy], dtype=np.float64)
    return ShpPolygon(rot)

# === NEW: post-fitting to targets ===
def post_fit_to_targets(
    pred: Dict[str, torch.Tensor],
    zone_onehot: torch.Tensor,
    la_target_norm: float | None,
    svc_target_norm_vec: torch.Tensor | None,
    normalizer: "TargetNormalizer",
    il_prob_thr: float = 0.5,
    sv1_thr: float = 0.5
) -> Dict[str, torch.Tensor]:
    device = next(iter(pred.values())).device
    out = pred.copy()

    # Денорм (если ещё не сделан)
    if "la_real" not in out and "la" in pred and "scale_l" in pred:
        la_norm = torch.clamp_min(pred["la"], 0.0)
        out["la_real"] = torch.clamp_min(
            normalizer.decode_la(la_norm, pred["scale_l"].to(device)),
            0.0
        )
    if "svc_real" not in out and "svc" in pred:
        out["svc_real"] = normalizer.decode_svc(pred["svc"])

    # Маска «жилых»
    il_prob = torch.sigmoid(pred["il"]).view(-1)
    live_mask = (il_prob >= il_prob_thr).float().view(-1,1)

    # Подгонка блочной жилплощади
    if la_target_norm is not None and "scale_l" in pred and "la_real" in out:
        if pred["scale_l"].ndim == 2:
            scale_blk = pred["scale_l"][0:1, :]
        else:
            scale_blk = pred["scale_l"].view(1,1)

        la_target_real = normalizer.decode_la(
            torch.tensor([[float(la_target_norm)]], device=device),
            scale_blk.to(device)
        ).item()

        la_pos = torch.clamp_min(out["la_real"], 0.0)
        sum_cur = (la_pos * live_mask).sum().item()
        if sum_cur > 1e-6:
            factor = la_target_real / sum_cur
            la_adj = la_pos * live_mask * factor + la_pos * (1.0 - live_mask)  # не трогаем не-жилые
            out["la_real"] = torch.clamp_min(la_adj, 0.0)
        else:
            out["la_real"] = torch.clamp_min(la_pos, 0.0)

    # Подгонка сервисов (как было, с ReLU-логикой по сути через decode_svc)
    if (svc_target_norm_vec is not None) and out.get("svc_real", None) is not None:
        S = out["svc_real"].shape[1]
        sv1_prob = torch.sigmoid(pred["sv1"]) if "sv1" in pred else torch.sigmoid(pred["svc"])
        sv1_bin = (sv1_prob >= sv1_thr).float()
        tgt_real = normalizer.decode_svc(svc_target_norm_vec.to(device).view(1, -1)).view(-1)
        cur_sum = (out["svc_real"] * sv1_bin).sum(dim=0)
        for s in range(S):
            t = float(tgt_real[s].item()); cs = float(cur_sum[s].item())
            if t <= 0:
                out["svc_real"][:, s] = 0.0; sv1_bin[:, s] = 0.0
            elif cs <= 1e-6:
                k = max(1, min(5, out["svc_real"].shape[0]))
                topk = torch.topk(sv1_prob[:, s], k=k).indices
                out["svc_real"][:, s] = 0.0
                out["svc_real"][topk, s] = t / float(k)
                sv1_bin[topk, s] = 1.0
            else:
                out["svc_real"][:, s] = out["svc_real"][:, s] * sv1_bin[:, s] * (t / cs)
        out["sv1_bin"] = sv1_bin

    # На выходе гарантируем неотрицательность
    if "la_real" in out:
        out["la_real"] = torch.clamp_min(out["la_real"], 0.0)

    return out

# -------------------------------
# INFERENCE
# --------------------------------
# === NEW: inference runner ===
def run_infer(
    args,
    model: "GraphModel",
    normalizer: "TargetNormalizer",
    zone2id: Dict[str,int],
    service2id: Dict[str,int]
):
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(device)

    with open(args.infer_geojson_in, "r", encoding="utf-8") as f:
        gj = json.load(f)
    if gj.get("type") == "FeatureCollection":
        feats = gj.get("features", [])
        if not feats:
            raise ValueError("Empty FeatureCollection")
        feat = feats[0]
    elif gj.get("type") == "Feature":
        feat = gj
    else:
        feat = {"type":"Feature","geometry":gj}
    poly_world = _polygon_from_geojson(feat)

    poly_can, fwd, inv, scale_l = canonicalize_polygon(poly_world)
    mask_size = tuple([int(_CFG["model"].get("mask_size", 128))]*2)
    mask_img = polygon_to_mask(poly_can, size=mask_size).to(device)  # (1,H,W)

    n_slots = int(getattr(args, "infer_slots", 256))
    pts = sample_slots_grid(poly_can, n_slots=n_slots, jitter=0.35, seed=42)
    if len(pts) == 0:
        raise ValueError("No slots sampled inside polygon")
    edge_index = build_knn_edges(pts, k=int(getattr(args, "infer_knn", 8))).to(device)

    zone_label = args.zone
    if zone_label not in zone2id:
        raise ValueError(f"Unknown zone '{zone_label}'. Известные: {list(zone2id.keys())[:8]}…")
    zone_id = zone2id[zone_label]
    zone_onehot = torch.as_tensor(np.eye(len(zone2id), dtype=np.float32)[zone_id])  # (Z,)

    la_target = float(args.la_target) if args.la_target is not None else 0.0
    la_target_norm = normalizer.encode_la(
        torch.tensor([[la_target]], dtype=torch.float32),
        torch.tensor([[scale_l]], dtype=torch.float32)
    ).item()

    S = len(service2id)
    sv1_block = torch.zeros((S,), dtype=torch.float32)
    svc_block_norm = torch.zeros((S,), dtype=torch.float32)
    if args.services_target is not None:
        st_raw = args.services_target
        if isinstance(st_raw, str) and os.path.exists(st_raw):
            with open(st_raw, "r", encoding="utf-8") as f:
                st = json.load(f)
        else:
            st = json.loads(st_raw)
        if isinstance(st, dict):
            iters = st.items()
        elif isinstance(st, list):
            iters = [(d.get("name"), d.get("value", 0.0)) for d in st]
        else:
            raise ValueError("services_target must be dict or list of {name,value}")
        for name, v in iters:
            if name in service2id:
                sid = service2id[name]
                val = float(v or 0.0)
                if val > 0:
                    sv1_block[sid] = 1.0
                    svc_block_norm[sid] = normalizer.encode_svc(torch.tensor([val], dtype=torch.float32)).item()

    N = pts.shape[0]
    e_stub = torch.ones((N,1), dtype=torch.float32)
    scale_feat = torch.full((N,1), float(scale_l), dtype=torch.float32)
    la_blk_feat = torch.full((N,1), float(la_target_norm), dtype=torch.float32)
    sv1_blk_feat = sv1_block.view(1,-1).repeat(N,1)
    svc_blk_feat = svc_block_norm.view(1,-1).repeat(N,1)
    zone_feat = torch.from_numpy(np.tile(zone_onehot.numpy(), (N,1))).float()

    x_in = torch.cat([e_stub, zone_feat, scale_feat, la_blk_feat, sv1_blk_feat, svc_blk_feat], dim=1).to(device)
    batch_index = torch.zeros((N,), dtype=torch.long, device=device)

    with torch.no_grad():
        pred = model(
            x=x_in, edge_index=edge_index,
            batch_index=batch_index, mask_img=mask_img
        )
        pred["scale_l"] = torch.full((N,1), float(scale_l), device=device)

        den = denorm_predictions({"y":{"scale_l": pred["scale_l"]}}, pred, normalizer)
        pred.update(den)

        svc_vec = svc_block_norm.clone() if S > 0 else None
        pred_pf = post_fit_to_targets(
            {**pred, **{"scale_l": pred["scale_l"]}},
            zone_feat[0], la_target_norm,
            (svc_vec if svc_vec is None else svc_vec),
            normalizer=normalizer,
            il_prob_thr=float(getattr(args, "infer_il_thr", 0.5)),
            sv1_thr=float(getattr(args, "infer_sv1_thr", 0.5))
        )
        pred.update(pred_pf)

    e_mask = (torch.sigmoid(pred["e"].view(-1)) > float(getattr(args,"infer_e_thr",0.5))).cpu().numpy()

    # размер в канонике — аккуратный клип
    sz = pred["sz"].detach().cpu().numpy()
    sz[:, 0] = np.clip(sz[:, 0], 0.01, 0.35)
    sz[:, 1] = np.clip(sz[:, 1], 0.01, 0.35)

    out_features = []
    for i, keep in enumerate(e_mask):
        if not keep:
            continue
        center = pts[i]
        size = sz[i]
        phi = float(pred["phi"][i].detach().cpu().item())
        poly_can = rect_polygon(center, size, phi)

        xs, ys = poly_can.exterior.xy
        world_np = [inv((float(x), float(y))) for x, y in zip(xs, ys)]
        world = [(float(p[0]), float(p[1])) for p in world_np]
        if world and world[0] != world[-1]:
            world.append(world[0])

        il_prob_i = float(torch.sigmoid(pred["il"][i]).item())
        la_i = float(torch.clamp_min(pred.get("la_real", torch.zeros_like(pred["e"]))[i], 0.0).item())
        # Пишем жилую площадь только для «жилых», иначе 0
        living_area_out = la_i if il_prob_i >= float(getattr(args,"infer_il_thr",0.5)) else 0.0

        props = {
            "zone": zone_label,
            "e": float(torch.sigmoid(pred["e"][i]).item()),
            "is_living": il_prob_i,
            "living_area": living_area_out,
        }
        if "svc_real" in pred and pred["svc_real"].numel() > 0:
            caps = pred["svc_real"][i].detach().cpu().numpy().tolist()
            if any(c > 0 for c in caps):
                props["services_capacity"] = [
                    {"name": name, "value": float(caps[sid])}
                    for name, sid in service2id.items()
                    if 0 <= sid < len(caps) and caps[sid] > 0
                ]

        out_features.append({
            "type":"Feature",
            "geometry": {"type":"Polygon","coordinates":[world]},
            "properties": props
        })

    out_fc = {"type":"FeatureCollection","features":out_features}
    os.makedirs(os.path.dirname(args.infer_out) or ".", exist_ok=True)
    with open(args.infer_out, "w", encoding="utf-8") as f:
        json.dump(out_fc, f, ensure_ascii=False, indent=2)
    log.info(f"[infer] Saved → {args.infer_out}")


# ----------------------
# Датасет
# ----------------------
class HCanonGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        zones_json: str,
        services_json: str,
        split: str = "train",
        split_ratio: float = 0.9,
        mask_size: tuple[int,int] = (128,128),
        mask_root: str | None = None,
    ):
        super().__init__()
        # загрузка таблиц
        self.blocks = pd.read_parquet(os.path.join(data_dir, "blocks.parquet"))
        self.nodes  = pd.read_parquet(os.path.join(data_dir, "nodes_fixed.parquet"))
        self.edges  = pd.read_parquet(os.path.join(data_dir, "edges.parquet"))

        # словарь зон
        if os.path.exists(zones_json):
            with open(zones_json, "r", encoding="utf-8") as f:
                self.zone2id = json.load(f)
        else:
            uniq = sorted([str(z) for z in self.blocks["zone"].fillna("nan").unique().tolist()])
            self.zone2id = {z:i for i,z in enumerate(uniq)}
            _ensure_dir(zones_json)
            with open(zones_json, "w", encoding="utf-8") as f:
                json.dump(self.zone2id, f, ensure_ascii=False, indent=2)
        self.id2zone = {v:k for k,v in self.zone2id.items()}

        # словарь сервисов
        if os.path.exists(services_json):
            with open(services_json, "r", encoding="utf-8") as f:
                self.service2id = json.load(f)
        else:
            vocab = set()
            for x in self.nodes["services_present"].fillna("").values.tolist():
                lst = _maybe_json_to_list(x)
                for s in lst:
                    vocab.add(str(s))
            self.service2id = {s:i for i,s in enumerate(sorted(vocab))}
            _ensure_dir(services_json)
            with open(services_json, "w", encoding="utf-8") as f:
                json.dump(self.service2id, f, ensure_ascii=False, indent=2)
        self.id2service = {v:k for k,v in self.service2id.items()}
        self.num_services = len(self.service2id)

        # индексы блоков и сплит
        self.block_ids = sorted(self.blocks["block_id"].astype(str).unique().tolist())
        n = len(self.block_ids)
        n_train = int(round(n * split_ratio))
        self.split = split
        if split == "train":
            self.block_ids = self.block_ids[:n_train]
        else:
            self.block_ids = self.block_ids[n_train:]

        # кэши
        self.block_zone = {str(r.block_id): r.zone for r in self.blocks.itertuples(index=False)}
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_root = mask_root

        # mask paths
        self.block_mask_path = {}
        has_col = "mask_path" in self.blocks.columns
        for r in self.blocks.itertuples(index=False):
            bid = str(r.block_id)
            if has_col:
                self.block_mask_path[bid] = getattr(r, "mask_path", None)
            else:
                self.block_mask_path[bid] = (os.path.join(self.mask_root, f"{bid}.png")
                                             if self.mask_root else None)

        # индексации
        self.idx2block = {i: b for i, b in enumerate(self.block_ids)}
        self.block2idx = {b: i for i, b in enumerate(self.block_ids)}

        # группировки
        self._nodes_by_block = self.nodes.groupby("block_id")

        # метрики/страты/частоты
        self._compute_block_stats_and_strata(nbins=4, k_clusters=8)
        self._compute_label_stats()

        # масштаб квартала и нормализатор
        self._block_scale = {
            str(r.block_id): float(r.scale_l) if pd.notna(r.scale_l) else 0.0
            for r in self.blocks.itertuples(index=False)
        }
        self._normalizer = TargetNormalizer(eps=1e-6)

        # список блоков, где вообще есть сервисы (для service-stream)
        self.blocks_with_services: set[str] = set()
        if self.num_services > 0:
            for blk in self.block_ids:
                g = self._nodes_by_block.get_group(blk) if blk in self._nodes_by_block.groups else None
                if g is None or len(g) == 0:
                    continue
                has = False
                for r in g.itertuples(index=False):
                    m, c = self._services_vecs(getattr(r, "services_present", None),
                                               getattr(r, "services_capacity", None))
                    if (m.sum() > 0) or (float(np.asarray(c).sum()) > 0):
                        has = True
                        break
                if has:
                    self.blocks_with_services.add(blk)

        log.info(
            f"Dataset[{split}] blocks={len(self.block_ids)} "
            f"nodes={len(self.nodes)} edges={len(self.edges)} services={self.num_services}"
        )

    def __len__(self) -> int:
        return len(self.block_ids)

    def _node_targets(self, df_nodes_block: pd.DataFrame) -> Dict[str, torch.Tensor]:
        N = len(df_nodes_block)

        e = torch.as_tensor(pd.to_numeric(df_nodes_block["e_i"], errors="coerce")
                            .fillna(0.0).values, dtype=torch.float32).view(-1, 1)
        pos = torch.as_tensor(df_nodes_block[["posx","posy"]]
                              .apply(pd.to_numeric, errors="coerce").fillna(0.0).values, dtype=torch.float32)
        sz  = torch.as_tensor(df_nodes_block[["size_x","size_y"]]
                              .apply(pd.to_numeric, errors="coerce").fillna(0.0).values, dtype=torch.float32)
        phi = torch.as_tensor(pd.to_numeric(df_nodes_block["phi_resid"], errors="coerce")
                              .fillna(0.0).values, dtype=torch.float32).view(-1,1)
        s   = torch.as_tensor(pd.to_numeric(df_nodes_block["s_i"], errors="coerce")
                              .fillna(0).astype(int).values, dtype=torch.long).view(-1)
        a   = torch.as_tensor(pd.to_numeric(df_nodes_block["a_i"], errors="coerce")
                              .fillna(0.0).values, dtype=torch.float32).view(-1,1)
        hf  = torch.as_tensor(df_nodes_block["has_floors"].astype(bool).values, dtype=torch.float32).view(-1,1)
        fl  = torch.as_tensor(pd.to_numeric(df_nodes_block["floors_num"], errors="coerce")
                              .fillna(-1).astype(int).values, dtype=torch.long).view(-1)
        il  = torch.as_tensor(df_nodes_block["is_living"].astype(bool).values, dtype=torch.float32).view(-1,1)

        la_raw = torch.as_tensor(pd.to_numeric(df_nodes_block["living_area"], errors="coerce")
                                 .fillna(0.0).values, dtype=torch.float32).view(-1,1)

        blk_id_val = str(df_nodes_block["block_id"].iloc[0])
        sc = float(self._block_scale.get(blk_id_val, 0.0))
        scale_col = torch.full((N,1), sc, dtype=torch.float32)

        la = self._normalizer.encode_la(la_raw, scale_col)

        mhot_list, cap_list = [], []
        for r in df_nodes_block.itertuples(index=False):
            m, c = self._services_vecs(getattr(r, "services_present", None),
                                       getattr(r, "services_capacity", None))
            mhot_list.append(m); cap_list.append(c)
        if self.num_services > 0:
            sv1 = torch.as_tensor(np.vstack(mhot_list), dtype=torch.float32)
            svc_raw = torch.as_tensor(np.vstack(cap_list), dtype=torch.float32)
            svc = self._normalizer.encode_svc(svc_raw)
            svc_mask = (sv1 > 0).float()
        else:
            sv1 = torch.zeros((N,0), dtype=torch.float32)
            svc = torch.zeros((N,0), dtype=torch.float32)
            svc_mask = torch.zeros((N,0), dtype=torch.float32)

        return {
            "e": e, "pos": pos, "sz": sz, "phi": phi, "s": s, "a": a,
            "hf": hf, "fl": fl, "il": il,
            "la": la, "sv1": sv1, "svc": svc, "svc_mask": svc_mask,
            "scale_l": scale_col,
        }

    def __getitem__(self, idx: int):
        blk_id = self.block_ids[idx]
        nodes_b = self.nodes[self.nodes["block_id"] == blk_id].sort_values("slot_id").reset_index(drop=True)
        edges_b = self.edges[self.edges["block_id"] == blk_id]
        src = torch.as_tensor(edges_b["src_slot"].values, dtype=torch.long)
        dst = torch.as_tensor(edges_b["dst_slot"].values, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0) if len(src) else torch.zeros((2,0), dtype=torch.long)

        zone_label = str(self.block_zone.get(blk_id, "nan"))
        zone_id = self.zone2id.get(zone_label, 0)
        zone_onehot = torch.as_tensor(self._one_hot(zone_id, len(self.zone2id)))

        # --- узловые таргеты ---
        targets = self._node_targets(nodes_b)

        # --- блочные цели для кондиционирования и блочных лоссов ---
        sc = float(self._block_scale.get(str(blk_id), 0.0))
        scale_scalar = torch.tensor([sc], dtype=torch.float32)  # (1,)

        la_block_raw = float(pd.to_numeric(nodes_b["living_area"], errors="coerce").fillna(0.0).sum())
        la_block_norm = self._normalizer.encode_la(
            torch.tensor([[la_block_raw]], dtype=torch.float32),
            torch.tensor([[sc]], dtype=torch.float32)
        ).view(1)  # (1,)

        if self.num_services > 0:
            # суммируем по блоку сырые capacity и presence
            acc_caps = np.zeros((self.num_services,), dtype=np.float32)
            acc_pres = np.zeros((self.num_services,), dtype=np.float32)
            for r in nodes_b.itertuples(index=False):
                m, c = self._services_vecs(getattr(r, "services_present", None),
                                           getattr(r, "services_capacity", None))
                acc_caps += c
                acc_pres = np.maximum(acc_pres, m)  # multi-hot по блоку
            svc_block_norm = self._normalizer.encode_svc(torch.tensor(acc_caps, dtype=torch.float32)).view(-1)    # (S,)
            sv1_block = torch.tensor(acc_pres, dtype=torch.float32).view(-1)                                      # (S,)
        else:
            svc_block_norm = torch.zeros((0,), dtype=torch.float32)
            sv1_block = torch.zeros((0,), dtype=torch.float32)

        # --- глобальное кондиционирование c добавляем в признаки каждого узла ---
        N = len(nodes_b)
        scale_feat = scale_scalar.view(1,1).repeat(N, 1)          # (N,1)
        la_blk_feat = la_block_norm.view(1,1).repeat(N, 1)        # (N,1)
        sv1_blk_feat = sv1_block.view(1, -1).repeat(N, 1)         # (N,S)
        svc_blk_feat = svc_block_norm.view(1, -1).repeat(N, 1)    # (N,S)

        x_in = torch.cat([
            torch.as_tensor(nodes_b[["e_i"]].values, dtype=torch.float32),    # (N,1)
            torch.tile(zone_onehot.view(1,-1), (N, 1)),                       # (N,|zones|)
            scale_feat,                                                       # (N,1) — ДОБАВЛЕНО
            la_blk_feat,                                                      # (N,1) — ДОБАВЛЕНО
            sv1_blk_feat,                                                     # (N,S) — ДОБАВЛЕНО
            svc_blk_feat,                                                     # (N,S) — ДОБАВЛЕНО
        ], dim=1)

        # путь к маске
        mask_path = self.block_mask_path.get(str(blk_id))
        mask_img = _load_mask(mask_path, size=self.mask_size)  # (1,H,W) или нули

        if _PYG_OK:
            data = GeomData(x=x_in, edge_index=edge_index)
            data.num_nodes = x_in.shape[0]
            data.block_id = blk_id
            data.zone_id = zone_id

            data.y_e   = targets["e"].view(-1,1)
            data.y_pos = targets["pos"].view(-1,2)
            data.y_sz  = targets["sz"].view(-1,2)
            data.y_phi = targets["phi"].view(-1,1)
            data.y_s   = targets["s"].view(-1)
            data.y_a   = targets["a"].view(-1,1)
            data.y_hf  = targets["hf"].view(-1,1)
            data.y_fl  = targets["fl"].view(-1)
            data.y_il  = targets["il"].view(-1,1)
            data.y_la  = targets["la"].view(-1,1)
            data.y_sv1 = targets["sv1"].view(-1, self.num_services) if self.num_services>0 else torch.zeros((len(nodes_b),0))
            data.y_svc = targets["svc"].view(-1, self.num_services) if self.num_services>0 else torch.zeros((len(nodes_b),0))
            data.y_svc_mask = targets["svc_mask"].view(-1, self.num_services) if self.num_services>0 else torch.zeros((len(nodes_b),0))
            data.scale_l = targets["scale_l"].view(-1,1)

            # блочные таргеты (по одному на граф)
            data.y_la_block  = la_block_norm.view(1,1)                 # (1,1)
            data.y_svc_block = svc_block_norm.view(1, -1)              # (1,S)

            data.mask_img = mask_img  # (1,H,W)
            return data
        else:
            # fallback: один граф на батч
            return {
                "x": x_in,
                "edge_index": edge_index,
                "y": {
                    **targets,
                    "la_block": la_block_norm.view(1,1),
                    "svc_block": svc_block_norm.view(1, -1),
                },
                "num_nodes": x_in.shape[0],
                "block_id": blk_id,
                "zone_id": zone_id,
                "mask_img": mask_img,
            }

    def _one_hot(self, idx: int, K: int) -> np.ndarray: 
        v = np.zeros((K,), dtype=np.float32) 
        if idx is not None and 0 <= idx < K: 
            v[idx] = 1.0 
        return v
    
    def _services_vecs(self, present_raw, cap_raw) -> Tuple[np.ndarray, np.ndarray]: 
        present = _maybe_json_to_list(present_raw) 
        cap = _maybe_json_to_list(cap_raw) 
        mhot = np.zeros((self.num_services,), dtype=np.float32) 
        for s in present: 
            sid = self.service2id.get(str(s)) 
            if sid is not None: 
                mhot[sid] = 1.0 
        cap_vec = np.zeros((self.num_services,), dtype=np.float32) 
        if isinstance(cap, list):
            if cap and isinstance(cap[0], dict): 
                for it in cap:
                    sid = self.service2id.get(str(it.get("name"))) 
                    if sid is not None:
                        try:
                            cap_vec[sid] = float(it.get("value", 0.0))
                        except Exception:
                            pass
            else:
                for i,v in enumerate(cap[:self.num_services]): 
                    try: 
                        cap_vec[i] = float(v) 
                    except Exception: 
                        pass 
        return mhot, cap_vec

    def get_pos_weights(self) -> Dict[str, float]:
        ls = getattr(self, "label_stats", None) or {}
        def _pw(p):
            p = max(min(float(p), 0.999999), 1e-6)
            return (1.0 - p) / p
        return {"e": _pw(ls.get("p_e", 0.01)),
                "il": _pw(ls.get("p_il", 0.1)),
                "hf": _pw(ls.get("p_hf", 0.5))}

    def get_s_class_weights(self) -> torch.Tensor:
        ls = getattr(self, "label_stats", None) or {}
        counts = ls.get("s_counts", {0:1,1:1,2:1,3:1})
        arr = np.array([counts.get(k, 1) for k in [0,1,2,3]], dtype=np.float64)
        arr = np.where(arr > 0, arr, 1.0)
        inv = 1.0 / arr
        inv = inv / inv.mean()
        return torch.tensor(inv, dtype=torch.float32)
        
    def _compute_block_stats_and_strata(self, nbins: int = 4, k_clusters: int = 8):
        rows = []
        cols_exist = set(self.nodes.columns)
        has_il  = "is_living" in cols_exist
        has_fl  = "floors_num" in cols_exist
        has_hf  = "has_floors" in cols_exist
        needed  = {"posx","posy","size_x","size_y","phi_resid"}
        has_geom = needed.issubset(cols_exist)

        for blk_id in self.block_ids:
            g = self._nodes_by_block.get_group(blk_id) if blk_id in self._nodes_by_block.groups else None
            if g is None or len(g) == 0:
                rows.append({"block_id": blk_id, "occupancy": 0.0, "living_share_active": 0.0,
                             "floors_mean": np.nan, "posx_m":0,"posx_s":0,"posy_m":0,"posy_s":0,
                             "sx_m":0,"sy_m":0,"phi_m":0,"phi_s":0})
                continue
            e = (g["e_i"].values > 0.5).astype(np.float32)
            n_tot = float(len(g))
            n_act = float(e.sum())
            occ = (n_act / n_tot) if n_tot > 0 else 0.0

            if has_il and n_act > 0:
                il = g["is_living"].astype(bool).values.astype(np.float32)
                living_share = float((il * e).sum() / (n_act + 1e-6))
            else:
                living_share = 0.0

            if has_fl and has_hf and n_act > 0:
                hf = g["has_floors"].astype(bool).values.astype(np.float32)
                fln = pd.to_numeric(g["floors_num"], errors="coerce").fillna(np.nan).values
                msk = (e > 0) & (hf > 0) & np.isfinite(fln)
                fl_mean = float(np.mean(fln[msk])) if np.any(msk) else np.nan
            else:
                fl_mean = np.nan

            if has_geom and n_act > 0:
                px, py = g["posx"].values, g["posy"].values
                sx, sy = g["size_x"].values, g["size_y"].values
                ph     = g["phi_resid"].values
                rows.append({
                    "block_id": blk_id,
                    "occupancy": occ,
                    "living_share_active": living_share,
                    "floors_mean": fl_mean,
                    "posx_m": float(np.mean(px[e>0])), "posx_s": float(np.std(px[e>0])),
                    "posy_m": float(np.mean(py[e>0])), "posy_s": float(np.std(py[e>0])),
                    "sx_m": float(np.mean(sx[e>0])), "sy_m": float(np.mean(sy[e>0])),
                    "phi_m": float(np.mean(ph[e>0])), "phi_s": float(np.std(ph[e>0])),
                })
            else:
                rows.append({
                    "block_id": blk_id, "occupancy": occ, "living_share_active": living_share,
                    "floors_mean": fl_mean, "posx_m":0,"posx_s":0,"posy_m":0,"posy_s":0,
                    "sx_m":0,"sy_m":0,"phi_m":0,"phi_s":0
                })

        meta = pd.DataFrame(rows).set_index("block_id")
        meta["scale_l"] = self.blocks.set_index("block_id").reindex(self.block_ids)["scale_l"].values
        meta["zone"]    = self.blocks.set_index("block_id").reindex(self.block_ids)["zone"].astype(str).values

        meta["bin_occ"]   = _quantile_bins(meta["occupancy"].values, nbins)
        meta["bin_lshare"]= _quantile_bins(meta["living_share_active"].values, nbins)
        meta["bin_floor"] = _quantile_bins(pd.to_numeric(meta["floors_mean"], errors="coerce").fillna(-1).values, nbins)
        meta["bin_scale"] = _quantile_bins(pd.to_numeric(meta["scale_l"],  errors="coerce").fillna(-1).values, nbins)

        if _SK_OK:
            res_mask = (meta["zone"].values == "residential")
            X = meta.loc[res_mask, ["posx_m","posx_s","posy_m","posy_s","sx_m","sy_m","phi_m","phi_s"]].values
            if X.shape[0] >= k_clusters and np.isfinite(X).all():
                km = KMeans(n_clusters=k_clusters, n_init=10, random_state=42)
                cl = km.fit_predict(X)
                meta.loc[res_mask, "res_cluster"] = cl.astype(int)
            else:
                meta["res_cluster"] = -1
        else:
            meta["res_cluster"] = -1

        self.block_meta = meta
        self.zone2indices = {}
        for z, grp in meta.groupby("zone"):
            self.zone2indices[str(z)] = [self.block2idx[b] for b in grp.index.tolist() if b in self.block2idx]

    def get_zone_indices(self, zone_label: str) -> List[int]:
        return self.zone2indices.get(str(zone_label), [])

    def get_block_meta(self, idx: int) -> Dict[str, Any]:
        blk = self.idx2block[idx]
        r = self.block_meta.loc[blk]
        return r.to_dict()
    
    def _compute_label_stats(self, thr: float = 0.5) -> None:
        """
        Считает частоты меток для балансировок лоссов:
            - p_e  : доля активных узлов (e_i > thr)
            - p_il : доля жилых среди активных
            - p_hf : доля "есть этажность" среди активных
            - s_counts: частоты классов формы s_i среди активных (ключи 0..3)
        Записывает в self.label_stats (dict).
        """
        df = self.nodes.copy()
        cols = set(df.columns)

        # e_i может быть не 0/1 — бинализуем порогом
        if "e_i" in cols:
            e = pd.to_numeric(df["e_i"], errors="coerce").fillna(0.0).values
            m_act = (e > float(thr))
            p_e = float(m_act.mean()) if len(m_act) else 0.0
        else:
            m_act = np.zeros((len(df),), dtype=bool)
            p_e = 0.0

        # is_living среди активных
        if "is_living" in cols and m_act.any():
            il = df["is_living"].astype(bool).values
            p_il = float(il[m_act].mean()) if m_act.sum() > 0 else 0.0
        else:
            p_il = 0.0

        # has_floors среди активных
        if "has_floors" in cols and m_act.any():
            hf = df["has_floors"].astype(bool).values
            p_hf = float(hf[m_act].mean()) if m_act.sum() > 0 else 0.0
        else:
            p_hf = 0.0

        # частоты классов формы среди активных
        s_counts = {0: 1, 1: 1, 2: 1, 3: 1}  # псевдосчётчики по умолчанию (чтобы не было деления на ноль)
        if "s_i" in cols and m_act.any():
            s_raw = pd.to_numeric(df["s_i"], errors="coerce").fillna(-1).astype(int).values
            s_active = s_raw[m_act]
            # считаем только допустимые классы 0..3
            uniq, cnt = np.unique(s_active[(s_active >= 0) & (s_active <= 3)], return_counts=True)
            # перезапишем поверх дефолта, оставив 1 для отсутствующих
            for k in range(4):
                val = int(cnt[uniq.tolist().index(k)]) if (k in uniq.tolist()) else 0
                s_counts[k] = max(1, val)

        self.label_stats = {
            "p_e": p_e,
            "p_il": p_il,
            "p_hf": p_hf,
            "s_counts": s_counts
        }

def make_zone_temperature_weights(ds: "HCanonGraphDataset", tau: float = 1.0) -> np.ndarray:
    """
    Возвращает веса длины len(ds) пропорционально f_z**tau,
    где f_z — частоты блоков в зонах. При tau=0 получаем рав.-по-зонам.
    """
    counts = {z: len(ixs) for z, ixs in ds.zone2indices.items()}
    # вероятности зон
    total = sum(counts.values()) + 1e-9
    pz = {z: (c / total) for z, c in counts.items()}

    # p(z|tau)
    for z in pz:
        pz[z] = max(pz[z], 1e-12) ** float(tau)

    # нормировка
    s = sum(pz.values())
    pz = {z: pz[z] / s for z in pz}

    w = np.zeros((len(ds),), dtype=np.float64)
    for z, ixs in ds.zone2indices.items():
        for i in ixs:
            w[i] = pz[z]
    # безопасная нормировка
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-12)
    w = w / w.sum()
    return w

class MajoritySampler:
    """Семплирование по весам f_z**tau; внутри residential — равномерно по стратам/кластерам."""
    def __init__(self, ds: HCanonGraphDataset, tau: float = 0.8,
                 residential_stratify: bool = True,
                 use_clusters: bool = True):
        self.ds = ds
        self.base_w = make_zone_temperature_weights(ds, tau=tau)
        self.residential_stratify = residential_stratify
        self.use_clusters = use_clusters

        self.res_indices = ds.get_zone_indices("residential")
        self._res_strata = None
        if self.residential_stratify and len(self.res_indices) > 0:
            strata = {}
            for i in self.res_indices:
                blk_id = ds.idx2block[i]
                m = ds.block_meta.loc[blk_id]
                b1 = int(m["bin_occ"]); b2 = int(m["bin_lshare"])
                b3 = int(m["bin_floor"]); b4 = int(m["bin_scale"])
                cl = int(m.get("res_cluster", -1))
                key = (b1, b2, b3, b4, cl if (self.use_clusters and cl >= 0) else -1)
                strata.setdefault(key, []).append(i)
            self._res_strata = {k: v for k, v in strata.items() if len(v) > 0}

    def sample(self, k: int) -> List[int]:
        out: List[int] = []
        # половина из «общей» выборки по весам, половина — из стратифицированного residential (если есть)
        k_res = min(k // 2, len(self.res_indices)) if self._res_strata else 0
        k_gen = k - k_res

        if k_gen > 0:
            choices = np.random.choice(np.arange(len(self.ds)), size=k_gen, replace=True, p=self.base_w)
            out.extend(choices.tolist())

        if k_res > 0:
            keys = list(self._res_strata.keys())
            for _ in range(k_res):
                key = random.choice(keys)
                out.append(random.choice(self._res_strata[key]))
        return out
    
class MinoritySampler:
    """Равномерно по зонам (τ=0). В каждом выборе: случайная зона, случайный блок внутри неё."""
    def __init__(self, ds: HCanonGraphDataset):
        self.ds = ds
        self.zones = [z for z, lst in ds.zone2indices.items() if len(lst) > 0]

    def sample(self, k: int) -> List[int]:
        out: List[int] = []
        for _ in range(k):
            z = random.choice(self.zones)
            i = random.choice(self.ds.zone2indices[z])
            out.append(i)
        return out
    
class TwoStreamBatchSampler(Sampler[List[int]]):
    """
    Собирает батчи длины batch_size = k_maj + k_min:
      - k_maj = round(batch_size * majority_frac)
      - k_min = batch_size - k_maj
    + Гарантирует, что в батче будет хотя бы один блок с сервисами (если такие есть).
    """
    def __init__(self, ds: HCanonGraphDataset,
                 batch_size: int = 8,
                 majority_frac: float = 0.5,
                 maj_sampler: MajoritySampler | None = None,
                 min_sampler: MinoritySampler | None = None,
                 epoch_len_batches: int | None = None):
        self.ds = ds
        self.batch_size = int(batch_size)
        self.k_maj = int(round(self.batch_size * float(majority_frac)))
        self.k_min = self.batch_size - self.k_maj
        self.maj = maj_sampler or MajoritySampler(ds)
        self.min = min_sampler or MinoritySampler(ds)
        self._n_batches = int(math.ceil(len(ds) / max(1, self.batch_size))) if epoch_len_batches is None else int(epoch_len_batches)

        # индексы (по split'у) блоков, где есть сервисы
        self.service_indices: List[int] = []
        if getattr(ds, "blocks_with_services", None):
            for b in ds.blocks_with_services:
                if b in ds.block2idx:
                    self.service_indices.append(ds.block2idx[b])

    def __iter__(self):
        for _ in range(self._n_batches):
            idxs = self.maj.sample(self.k_maj) + self.min.sample(self.k_min)
            random.shuffle(idxs)
            # гарантируем «service-stream»
            if self.service_indices and not any((i in set(self.service_indices)) for i in idxs):
                idxs[-1] = random.choice(self.service_indices)
            yield idxs

    def __len__(self) -> int:
        return self._n_batches

# ----------------------
# Модель
# ----------------------
class NodeHead(nn.Module):
    """Голова предсказаний по узлам — реконструирует все целевые атрибуты."""
    def __init__(self, in_dim: int, hidden: int, num_services: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # выходы
        self.e  = nn.Linear(hidden, 1)
        self.pos = nn.Linear(hidden, 2)
        self.sz  = nn.Linear(hidden, 2)
        self.phi = nn.Linear(hidden, 1)
        self.s   = nn.Linear(hidden, 4)  # классы форм (Rect/L/U/X)
        self.a   = nn.Linear(hidden, 1)
        self.hf  = nn.Linear(hidden, 1)
        self.fl  = nn.Linear(hidden, 1)  # регресс _условной_ этажности (софт)
        self.il  = nn.Linear(hidden, 1)
        self.la  = nn.Linear(hidden, 1)
        self.sv1 = nn.Linear(hidden, num_services)
        self.svc = nn.Linear(hidden, num_services)

    def forward(self, x):
        h = self.mlp(x)
        return {
            "e": self.e(h), "pos": self.pos(h), "sz": self.sz(h), "phi": self.phi(h),
            "s": self.s(h), "a": self.a(h), "hf": self.hf(h), "fl": self.fl(h),
            "il": self.il(h), "la": self.la(h), "sv1": self.sv1(h), "svc": self.svc(h)
        }


class MaskEncoderCNN(nn.Module):
    """
    Простой энкодер маски квартала: (B,1,H,W) -> (B, mask_dim).
    Адаптивный пуллинг позволяет работать с любым HxW.
    """
    def __init__(self, in_ch: int = 1, mask_dim: int = 64):
        super().__init__()
        c1, c2, c3 = 16, 32, 64
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/2, W/2

            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # H/4, W/4

            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # -> (B,c3,1,1)
        )
        self.proj = nn.Sequential(
            nn.Flatten(),               # (B,c3)
            nn.Linear(c3, mask_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,H,W) или (B,H,W) — второй вариант добавим канал внутри GraphModel
        h = self.net(x)
        z = self.proj(h)  # (B,mask_dim)
        return z

class GraphModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_services: int,
        use_mask: bool = True,
        mask_dim: int = 64,
    ):
        super().__init__()
        self.use_pyg = _PYG_OK
        self.use_mask = bool(use_mask)
        self.mask_dim = int(mask_dim) if self.use_mask else 0

        self.mask_cnn = MaskEncoderCNN(in_ch=1, mask_dim=self.mask_dim) if self.use_mask else None

        gnn_in = in_dim + self.mask_dim
        if self.use_pyg:
            self.gnn = GraphSAGE(in_channels=gnn_in, hidden_channels=hidden, num_layers=3, out_channels=hidden)
            head_in = hidden
        else:
            self.gnn = nn.Sequential(nn.Linear(gnn_in, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
            head_in = hidden

        self.head = NodeHead(head_in, hidden, num_services)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        mask_img: torch.Tensor | None = None,
    ):
        if self.use_mask and (mask_img is not None) and (batch_index is not None):
            # PyG может склеить (1,H,W) по dim=0 → (B,H,W). Добавим канал при необходимости.
            if mask_img.ndim == 3:
                mask_img = mask_img.unsqueeze(1)  # (B,1,H,W)
            z = self.mask_cnn(mask_img)          # (B,mask_dim)
            per_node = z[batch_index]            # (N,mask_dim)
            x = torch.cat([x, per_node], dim=1)
        else:
            if self.mask_dim > 0:
                x = torch.cat([x, torch.zeros((x.size(0), self.mask_dim), device=x.device, dtype=x.dtype)], dim=1)

        if self.use_pyg:
            h = self.gnn(x, edge_index)
        else:
            h = self.gnn(x)
        return self.head(h)

# ----------------------
# Лоссы
# ----------------------
class LossComputer:
    def __init__(self, w: Dict[str, float],
                 posw: Dict[str, float] | None = None,
                 ce_weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.05):
        self.w = w
        self.posw = posw or {"e": 1.0, "il": 1.0, "hf": 1.0}
        self.ce_weight = ce_weight
        self.label_smoothing = float(label_smoothing)
        self.l1 = nn.L1Loss(reduction='none')

    def _block_sum(self, v: torch.Tensor, batch_index: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Сумма по блокам: v — (N, D) или (N,); mask — (N,1)/(N,D) или None."""
        if v.ndim == 1:
            v = v.view(-1,1)
        if mask is not None:
            v = v * mask
        B = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 1
        out = torch.zeros((B, v.shape[1]), device=v.device, dtype=v.dtype)
        out.index_add_(0, batch_index, v)
        return out  # (B,D)

    def _collision_loss(self, pred: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor],
                        batch_index: torch.Tensor | None) -> torch.Tensor:
        """
        Простой penalty за перекрытие прямоугольников активных узлов в каждом блоке.
        overlap_area = ReLU((sx_i+sx_j)/2 - |dx|) * ReLU((sy_i+sy_j)/2 - |dy|)
        Нормируем на число пар.
        """
        if batch_index is None:
            return torch.zeros((), device=pred["e"].device)

        e_mask = (y["e"] > 0.5).view(-1)  # (N,)
        if e_mask.sum() <= 1:
            return torch.zeros((), device=pred["e"].device)

        pos = pred["pos"]        # (N,2)
        sz  = pred["sz"]         # (N,2)
        dev = pos.device

        loss = torch.zeros((), device=dev)
        B = int(batch_index.max().item()) + 1
        for b in range(B):
            m = (batch_index == b) & e_mask
            idx = torch.nonzero(m, as_tuple=False).view(-1)
            if idx.numel() <= 1:
                continue
            # ограничим максимумом, чтобы не взорваться по памяти
            if idx.numel() > 128:
                idx = idx[torch.randperm(idx.numel(), device=dev)[:128]]

            P = pos[idx]  # (K,2)
            S = sz[idx]   # (K,2)
            dx = (P[:,0].unsqueeze(1) - P[:,0].unsqueeze(0)).abs()
            dy = (P[:,1].unsqueeze(1) - P[:,1].unsqueeze(0)).abs()
            w  = (S[:,0].unsqueeze(1) + S[:,0].unsqueeze(0)) * 0.5
            h  = (S[:,1].unsqueeze(1) + S[:,1].unsqueeze(0)) * 0.5
            ovx = torch.relu(w - dx)
            ovy = torch.relu(h - dy)
            area = ovx * ovy
            # нулим диагональ
            area = area - torch.diag_embed(torch.diag(area))
            denom = max(1.0, (area.numel() - area.shape[0]))
            loss = loss + area.sum() / denom
        return loss

    def __call__(self, pred: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor],
                 batch_index: torch.Tensor | None = None) -> Tuple[torch.Tensor, Dict[str,float]]:
        device = pred["e"].device

        # Маски
        e_mask = (y["e"] > 0.5).float()                 # (N,1)
        denom_nodes = e_mask.sum() + 1e-6

        # ----- BCE с pos_weight -----
        pw_e  = torch.tensor(self.posw.get("e", 1.0),  device=device)
        pw_il = torch.tensor(self.posw.get("il", 1.0), device=device)
        pw_hf = torch.tensor(self.posw.get("hf", 1.0), device=device)

        e_loss_t  = F.binary_cross_entropy_with_logits(pred["e"],  y["e"],  pos_weight=pw_e,  reduction='mean')
        il_loss_t = (F.binary_cross_entropy_with_logits(pred["il"], y["il"], pos_weight=pw_il, reduction='none') * e_mask).sum() / denom_nodes
        hf_loss_t = (F.binary_cross_entropy_with_logits(pred["hf"], y["hf"], pos_weight=pw_hf, reduction='none') * e_mask).sum() / denom_nodes

        # ----- Регрессии (по активным) -----
        l1 = self.l1
        pos_loss_t = (l1(pred["pos"], y["pos"]) * e_mask).sum() / denom_nodes
        sz_loss_t  = (l1(pred["sz"],  y["sz"])  * e_mask).sum() / denom_nodes
        phi_loss_t = (l1(pred["phi"], y["phi"]) * e_mask).sum() / denom_nodes
        a_loss_t   = (l1(pred["a"],   y["a"])   * e_mask).sum() / denom_nodes

        # ----- Жилая площадь (по жилым), уже нормирована -----
        il_mask = y["il"]
        denom_living = il_mask.sum() + 1e-6
        la_loss_t = (l1(pred["la"], y["la"]) * il_mask).sum() / denom_living

        # ----- Этажность -----
        fl_valid = (y["fl"].view(-1,1) >= 0).float()
        fl_mask  = (y["hf"] > 0.5).float() * fl_valid
        denom_fl = fl_mask.sum() + 1e-6
        fl_tgt   = y["fl"].clamp(min=0).float().view(-1,1)
        fl_pred  = pred["fl"].view(-1,1)
        fl_loss_t = (l1(fl_pred, fl_tgt) * fl_mask).sum() / denom_fl

        # ----- Классы формы -----
        ce_w = self.ce_weight.to(device) if self.ce_weight is not None else None
        s_loss_vec = F.cross_entropy(pred["s"], y["s"].long(),
                                     weight=ce_w, reduction='none',
                                     label_smoothing=self.label_smoothing)
        s_loss_t = (s_loss_vec * e_mask.view(-1)).sum() / denom_nodes

        # ----- Сервисы (узловые) -----
        if y["sv1"].numel() > 0:
            svc_mask = y.get("svc_mask", torch.zeros_like(y["sv1"]))
            denom_svc = svc_mask.sum() + 1e-6
            if denom_svc.item() > 0:
                svc_loss_t = (l1(pred["svc"], y["svc"]) * svc_mask).sum() / denom_svc
            else:
                svc_loss_t = torch.zeros((), device=device)
            sv1_loss_t = (F.binary_cross_entropy_with_logits(pred["sv1"], y["sv1"], reduction='none') * e_mask).sum() / denom_nodes
        else:
            sv1_loss_t = torch.zeros((), device=device)
            svc_loss_t = torch.zeros((), device=device)

        # ----- Блочные лоссы -----
        la_block_t = torch.zeros((), device=device)
        svc_block_t = torch.zeros((), device=device)
        if batch_index is not None:
            # суммируем по блокам предсказанное (нормированное), сопоставляем gt (нормир.)
            # la: берём только жилые
            la_pred = pred["la"].view(-1,1)
            il_mask_b = il_mask  # (N,1)
            sum_la_by_block = self._block_sum(la_pred, batch_index, il_mask_b)  # (B,1)
            if "la_block" in y or "y_la_block" in y:
                y_lab = y.get("la_block", y.get("y_la_block"))
                la_block_t = torch.mean(torch.abs(sum_la_by_block - y_lab.to(device)))

            # svc: учитываем только присутствующие типы на узле
            if y["sv1"].numel() > 0:
                svc_pred = pred["svc"]  # (N,S)
                svc_mask = y.get("svc_mask", torch.zeros_like(svc_pred))
                sum_svc_by_block = self._block_sum(svc_pred, batch_index, svc_mask)  # (B,S)
                if ("svc_block" in y) or ("y_svc_block" in y):
                    y_svcb = y.get("svc_block", y.get("y_svc_block")).to(device)  # (B,S)
                    # усредняем L1 по активным сервисам в таргете, чтобы пустые блоки не доминировали
                    active = (y_svcb > 0).float()
                    denom = active.sum() + 1e-6
                    svc_block_t = (torch.abs(sum_svc_by_block - y_svcb) * torch.where(denom>0, active, active)).sum() / (denom + 1e-6)

        # ----- Коллизии (перекрытия) -----
        coll_loss_t = torch.zeros((), device=device)
        if self.w.get("coll", 0.0) > 0.0:
            coll_loss_t = self._collision_loss(pred, y, batch_index)

        # Сборка и взвешивание
        t_losses = {
            "e": e_loss_t, "pos": pos_loss_t, "sz": sz_loss_t, "phi": phi_loss_t,
            "s": s_loss_t, "a": a_loss_t, "fl": fl_loss_t, "hf": hf_loss_t,
            "il": il_loss_t, "la": la_loss_t, "sv1": sv1_loss_t, "svc": svc_loss_t,
            "la_block": la_block_t, "svc_block": svc_block_t, "coll": coll_loss_t
        }
        total_t = torch.zeros((), device=device)
        for k, t in t_losses.items():
            if k in self.w and self.w[k] != 0:
                total_t = total_t + float(self.w[k]) * t

        parts = {k: float(v.detach().item()) for k, v in t_losses.items()}
        return total_t, parts

# ----------------------
# Тренировка
# ----------------------

def train_loop(model: nn.Module, train_loader, val_loader, device: str, cfg: Dict[str,Any],
               save_ckpt_path: str, writer: SummaryWriter,
               loss_comp: LossComputer,
               normalizer: "TargetNormalizer | None" = None):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

    global_step = 0
    best_val = float('inf')

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        log.info(f"[Epoch {epoch}] — обучение…")
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(train_loader, total=len(train_loader), leave=False)
        epoch_loss = 0.0

        for batch in pbar:
            opt.zero_grad()
            if _PYG_OK:
                x  = batch.x.to(device)
                ei = batch.edge_index.to(device)
                bi = getattr(batch, "batch", None)
                mi = getattr(batch, "mask_img", None)
                if bi is not None: bi = bi.to(device)
                if mi is not None: mi = mi.to(device)

                y = {
                    "e":   batch.y_e.to(device),
                    "pos": batch.y_pos.to(device),
                    "sz":  batch.y_sz.to(device),
                    "phi": batch.y_phi.to(device),
                    "s":   batch.y_s.to(device).long().view(-1),
                    "a":   batch.y_a.to(device),
                    "hf":  batch.y_hf.to(device),
                    "fl":  batch.y_fl.to(device).long().view(-1),
                    "il":  batch.y_il.to(device),
                    "la":  batch.y_la.to(device),
                    "sv1": batch.y_sv1.to(device),
                    "svc": batch.y_svc.to(device),
                    "la_block":  batch.y_la_block.to(device),   # (B=1, 1)
                    "svc_block": batch.y_svc_block.to(device),  # (B=1, S)
                }
            else:
                x  = batch["x"].to(device)
                ei = batch["edge_index"].to(device)
                bi = None
                mi = batch.get("mask_img", None)
                if mi is not None: mi = mi.to(device)

                y = {k: v.to(device) for k,v in batch["y"].items()}
                if y["pos"].ndim == 2 and y["pos"].shape[0] == 2 and y["pos"].shape[1] != 2:
                    y["pos"] = y["pos"].T
                if y["sz"].ndim == 2 and y["sz"].shape[0] == 2 and y["sz"].shape[1] != 2:
                    y["sz"] = y["sz"].T

            with torch.no_grad():
                e_mean  = float((y["e"] > 0.5).float().mean().item())
                e_mask  = (y["e"] > 0.5).float()
                denom   = e_mask.sum() + 1e-6
                il_mean = float(((y["il"] > 0.5).float() * e_mask).sum().item() / denom.item())
                hf_mean = float(((y["hf"] > 0.5).float() * e_mask).sum().item() / denom.item())
                writer.add_scalar("batch_pos/e",  e_mean,  global_step)
                writer.add_scalar("batch_pos/il", il_mean, global_step)
                writer.add_scalar("batch_pos/hf", hf_mean, global_step)

            pred = model(x, ei, bi, mi)
            loss, parts = loss_comp(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += float(loss.item())

            writer.add_scalar("train/total", float(loss.item()), global_step)
            for k,v in parts.items():
                writer.add_scalar(f"train/{k}", float(v), global_step)
            global_step += 1
            pbar.set_description(f"loss={loss.item():.4f}")
        epoch_loss /= max(1, len(train_loader))

        # Валидация
        model.eval()
        with torch.no_grad():
            val_tot = 0.0; val_cnt = 0
            _dbg_la_logged = False
            for batch in val_loader:
                if _PYG_OK:
                    x  = batch.x.to(device)
                    ei = batch.edge_index.to(device)
                    bi = getattr(batch, "batch", None)
                    mi = getattr(batch, "mask_img", None)
                    if bi is not None: bi = bi.to(device)
                    if mi is not None: mi = mi.to(device)

                    y = {
                        "e":   batch.y_e.to(device),
                        "pos": batch.y_pos.to(device),
                        "sz":  batch.y_sz.to(device),
                        "phi": batch.y_phi.to(device),
                        "s":   batch.y_s.to(device).long().view(-1),
                        "a":   batch.y_a.to(device),
                        "hf":  batch.y_hf.to(device),
                        "fl":  batch.y_fl.to(device).long().view(-1),
                        "il":  batch.y_il.to(device),
                        "la":  batch.y_la.to(device),
                        "sv1": batch.y_sv1.to(device),
                        "svc": batch.y_svc.to(device),
                        "la_block":  batch.y_la_block.to(device),
                        "svc_block": batch.y_svc_block.to(device),
                    }
                else:
                    x  = batch["x"].to(device)
                    ei = batch["edge_index"].to(device)
                    bi = None
                    mi = batch.get("mask_img", None)
                    if mi is not None: mi = mi.to(device)

                    y = {k: v.to(device) for k,v in batch["y"].items()}
                    if y["pos"].ndim == 2 and y["pos"].shape[0] == 2 and y["pos"].shape[1] != 2:
                        y["pos"] = y["pos"].T
                    if y["sz"].ndim == 2 and y["sz"].shape[0] == 2 and y["sz"].shape[1] != 2:
                        y["sz"] = y["sz"].T

                pred = model(x, ei, bi, mi)
                l, _ = loss_comp(pred, y)

                if (epoch % 5 == 0) and (normalizer is not None) and (not _dbg_la_logged):
                    if _PYG_OK:
                        scale = batch.scale_l.to(device)
                        il_mask = y["il"].to(device)
                    else:
                        scale = batch["y"]["scale_l"].to(device)
                        il_mask = y["il"].to(device)

                    la_real = normalizer.decode_la(pred["la"], scale)
                    mask = (il_mask > 0.5).view(-1)
                    if mask.any():
                        vals = la_real.view(-1)[mask]
                        writer.add_scalar("debug/la_real_p10", torch.quantile(vals, 0.10).item(), epoch)
                        writer.add_scalar("debug/la_real_p50", torch.quantile(vals, 0.50).item(), epoch)
                        writer.add_scalar("debug/la_real_p90", torch.quantile(vals, 0.90).item(), epoch)
                    else:
                        writer.add_scalar("debug/la_real_p50", 0.0, epoch)
                    _dbg_la_logged = True

                val_tot += float(l.item()); val_cnt += 1
            val_loss = val_tot / max(1, val_cnt)
        writer.add_scalar("val/total", float(val_loss), epoch)
        log.info(f"[Epoch {epoch}] train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            _ensure_dir(save_ckpt_path)
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg,
                "val_loss": best_val,
                "timestamp": time.time()
            }, save_ckpt_path)
            log.info(f"Сохранён лучший чекпойнт → {save_ckpt_path}")

    return best_val

# ----------------------
# HF upload
# ----------------------

def push_to_hf(repo_id: str, token: str, folder: str, private: bool = True):
    if not _HF_OK:
        log.warning("huggingface_hub не установлен — пропускаю загрузку на HF")
        return
    if not repo_id:
        log.warning("Не указан --hf-repo-id — пропускаю загрузку на HF")
        return
    HfFolder.save_token(token or os.environ.get("HF_TOKEN", ""))
    try:
        create_repo(repo_id, token=token, exist_ok=True, private=private)
    except Exception:
        pass
    log.info(f"Загружаю артефакты в HF: repo_id={repo_id}, path={folder}")
    upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=folder,
        path_in_repo="",
        token=token,
        ignore_patterns=["*.png", "*.jpg", "*.jpeg", "*.zip"]
    )
    log.info("Готово: артефакты загружены на HF")

# ----------------------
# main
# ----------------------

def main():
    cfg = _CFG
    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        log.warning("CUDA недоступна — использую CPU")

    # общие пути
    data_dir = args.data_dir
    ckpt_path = args.model_ckpt
    zones_json = args.zones_json
    services_json = args.services_json

    # режим инференса?
    if args.mode == "infer":
        # загрузка артефактов
        aux_dir = os.path.join(os.path.dirname(ckpt_path) or data_dir, "artifacts")
        z_path = zones_json if os.path.exists(zones_json) else os.path.join(aux_dir, "zones.json")
        s_path = services_json if os.path.exists(services_json) else os.path.join(aux_dir, "services.json")
        norm_path = os.path.join(aux_dir, "target_normalizer.json")

        with open(z_path, "r", encoding="utf-8") as f:
            zone2id = json.load(f)
        with open(s_path, "r", encoding="utf-8") as f:
            service2id = json.load(f)
        normalizer = load_target_normalizer(norm_path)

        # восстановление модели
        # создадим временный датасет, чтобы узнать in_dim/S
        mask_size = tuple([int(_CFG["model"].get("mask_size", 128))]*2)
        # in_dim = 1 (e_stub) + |zones| + 1(scale_l) + 1(la_block_norm) + S + S
        in_dim = 1 + len(zone2id) + 1 + 1 + len(service2id) + len(service2id)
        model = GraphModel(
            in_dim=in_dim,
            hidden=int(_CFG["model"]["hidden"]),
            num_services=len(service2id),
            use_mask=bool(_CFG["model"].get("use_mask_cnn", True)),
            mask_dim=int(_CFG["model"].get("mask_dim", 64)),
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state"], strict=False)
        log.info(f"Loaded checkpoint: {ckpt_path}")

        # проброс аргументов инференса
        if not args.infer_geojson_in or not args.infer_out:
            raise ValueError("--infer-geojson-in и --infer-out обязательны в режиме --mode infer")
        if not args.zone:
            raise ValueError("--zone обязательно (например: residential)")

        run_infer(args, model, normalizer, zone2id, service2id)
        return 0

    # === TRAIN PATH ===
    log.info("Готовлю датасеты…")
    mask_size = tuple([int(_CFG["model"].get("mask_size", 128))]*2)
    mask_root = args.mask_root

    ds_train = HCanonGraphDataset(data_dir, zones_json, services_json,
                                  split="train", split_ratio=0.9,
                                  mask_size=mask_size, mask_root=mask_root)
    ds_val   = HCanonGraphDataset(data_dir, zones_json, services_json,
                                  split="val", split_ratio=0.9,
                                  mask_size=mask_size, mask_root=mask_root)
    
    log_service_presence_stats(ds_train)
    log_service_presence_stats(ds_val)

    posw = ds_train.get_pos_weights()
    ce_w = ds_train.get_s_class_weights()
    loss_comp = LossComputer(
        cfg["training"]["loss_weights"],
        posw=posw, ce_weight=ce_w, label_smoothing=0.05
    )
    if float(_CFG["training"]["loss_weights"].get("coll", 0.0)) <= 0.0:
        log.warning("[loss] collision loss weight is 0 — график train/coll будет ровной линией (0)")

    if _PYG_OK:
        if args.sampler_mode == "two_stream":
            maj = MajoritySampler(
                ds_train, tau=float(args.tau_majority),
                residential_stratify=not args.no_res_stratify, use_clusters=not args.no_res_clusters
            )
            mino = MinoritySampler(ds_train)
            batch_sampler = TwoStreamBatchSampler(
                ds_train, batch_size=int(args.batch_size), majority_frac=float(args.majority_frac),
                maj_sampler=maj, min_sampler=mino,
                epoch_len_batches=_CFG["sampler"].get("epoch_len_batches"),
            )
            train_loader = GeomLoader(ds_train, batch_sampler=batch_sampler)
        else:
            train_loader = GeomLoader(ds_train, batch_size=args.batch_size, shuffle=True)
        val_loader   = GeomLoader(ds_val,   batch_size=args.batch_size, shuffle=False)
        in_dim = ds_train[0].x.shape[1]
    else:
        def _collate(batch_list):
            xs = []; ys = {}
            for b in batch_list:
                xs.append(b["x"])
                for k, v in b["y"].items():
                    ys.setdefault(k, []).append(v)
            x = torch.cat(xs, dim=0)
            for k in ys: ys[k] = torch.cat(ys[k], dim=0)
            ei = torch.zeros((2,0), dtype=torch.long)
            return {"x": x, "edge_index": ei, "y": ys}
        if args.sampler_mode == "two_stream":
            maj = MajoritySampler(
                ds_train, tau=float(args.tau_majority),
                residential_stratify=not args.no_res_stratify, use_clusters=not args.no_res_clusters
            )
            mino = MinoritySampler(ds_train)
            batch_sampler = TwoStreamBatchSampler(
                ds_train, batch_size=int(args.batch_size), majority_frac=float(args.majority_frac),
                maj_sampler=maj, min_sampler=mino,
                epoch_len_batches=_CFG["sampler"].get("epoch_len_batches"),
            )
            train_loader = DataLoader(ds_train, batch_sampler=batch_sampler, collate_fn=_collate)
        else:
            train_loader = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=_collate)
        val_loader   = DataLoader(ds_val,   batch_size=1, shuffle=False, collate_fn=_collate)
        in_dim = ds_train[0]["x"].shape[1]

    log.info("Создаю модель…")
    model = GraphModel(
        in_dim=in_dim,
        hidden=int(_CFG["model"]["hidden"]),
        num_services=ds_train.num_services,
        use_mask=bool(_CFG["model"].get("use_mask_cnn", True)),
        mask_dim=int(_CFG["model"].get("mask_dim", 64)),
    ).to(device)
    log.info(f"Mask CNN enabled={_CFG['model'].get('use_mask_cnn', True)}; mask_dim={_CFG['model'].get('mask_dim', 64)}")

    # TensorBoard
    run_name = f"{cfg.get('experiment','exp')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runs_dir = os.path.join(os.path.dirname(ckpt_path) or data_dir, "runs", run_name)
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=runs_dir)
    log.info(f"TensorBoard лог-директория: {runs_dir}")

    best_val = train_loop(
        model, train_loader, val_loader, device, cfg, ckpt_path, writer,
        loss_comp=loss_comp, normalizer=getattr(ds_train, "_normalizer", None)
    )
    writer.close()

    # вспомогательные файлы
    aux_dir = os.path.join(os.path.dirname(ckpt_path) or data_dir, "artifacts")
    os.makedirs(aux_dir, exist_ok=True)
    with open(os.path.join(aux_dir, "zones.json"), "w", encoding="utf-8") as f:
        json.dump(ds_train.zone2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(aux_dir, "services.json"), "w", encoding="utf-8") as f:
        json.dump(ds_train.service2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(aux_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    norm_path = os.path.join(aux_dir, "target_normalizer.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(ds_train._normalizer.to_dict(), f, ensure_ascii=False, indent=2)

    if args.hf_push:
        repo_id = args.hf_repo_id
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if token:
            try:
                import shutil
                if not os.path.dirname(ckpt_path).startswith(aux_dir):
                    shutil.copy2(ckpt_path, os.path.join(aux_dir, os.path.basename(ckpt_path)))
            except Exception:
                pass
            push_to_hf(repo_id=repo_id, token=token, folder=aux_dir, private=bool(args.hf_private))
        else:
            log.warning("HF token не найден — пропускаю загрузку")

    log.info("Обучение завершено.")

def load_target_normalizer(path: str) -> TargetNormalizer:
    with open(path, "r", encoding="utf-8") as f:
        return TargetNormalizer.from_dict(json.load(f))

@torch.no_grad()
def denorm_predictions(batch, pred: Dict[str, torch.Tensor], normalizer: TargetNormalizer) -> Dict[str, torch.Tensor]:
    """
    Принимает батч и предсказания в НОРМ-формате, возвращает дикт с 'la_real', 'svc_real'.
    Требует наличие batch.scale_l (PyG) или batch["y"]["scale_l"] (fallback).
    ВАЖНО: обеспечивает неотрицательность жилой площади.
    """
    if hasattr(batch, "scale_l"):
        scale = batch.scale_l.to(pred["la"].device)  # (N,1)
    else:
        scale = batch["y"]["scale_l"].to(pred["la"].device)

    out = {}

    # ---- living area: clamp в норм-пространстве и ReLU в реальном ----
    if "la" in pred:
        la_norm = torch.clamp_min(pred["la"], 0.0)              # НЕ допускаем отрицательную la_norm
        la_real = normalizer.decode_la(la_norm, scale)          # ≥ 0 в реальных единицах
        la_real = torch.clamp_min(la_real, 0.0)                 # страховка
        out["la_real"] = la_real

    # ---- services: как раньше, но с маской присутствия ----
    if "svc" in pred:
        svc_real = normalizer.decode_svc(pred["svc"])           # может быть >0
        out["svc_real"] = svc_real

    if "sv1" in pred and "svc_real" in out:
        sv1_prob = torch.sigmoid(pred["sv1"])
        sv1_bin = (sv1_prob > 0.5).float()
        out["svc_real"] = out["svc_real"] * sv1_bin

    return out


if __name__ == "__main__":
    sys.exit(main())
