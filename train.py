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

# основной парсер
parser = argparse.ArgumentParser(parents=[_p0])
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
#sampler
parser.add_argument("--sampler-mode", choices=["two_stream","vanilla"], default=_CFG["sampler"]["mode"])
parser.add_argument("--majority-frac", type=float, default=_CFG["sampler"]["majority_frac"])
parser.add_argument("--tau-majority", type=float, default=_CFG["sampler"]["tau_majority"])
parser.add_argument("--no-res-stratify", action="store_true", help="Отключить стратификацию внутри residential")
parser.add_argument("--no-res-clusters", action="store_true", help="Отключить k-means кластеры внутри residential")
# hf
parser.add_argument("--hf-push", action="store_true", default=_CFG.get("hf",{}).get("push", False))
parser.add_argument("--hf-repo-id", default=_CFG.get("hf",{}).get("repo_id"))
parser.add_argument("--hf-token", default=_CFG.get("hf",{}).get("token"))
parser.add_argument("--hf-private", action="store_true", default=_CFG.get("hf",{}).get("private", True))
# misc
parser.add_argument("--log-level", default="INFO")
args = parser.parse_args()

setup_logger(args.log_level)
log.info(f"Sampler: mode={args.sampler_mode}, batch_size={args.batch_size}, "
         f"majority_frac={args.majority_frac}, tau={args.tau_majority}, "
         f"res_stratify={not args.no_res_stratify}, res_clusters={not args.no_res_clusters}")
log.info("Запуск train.py (YAML+CLI) …")

# ----------------------
# Утилиты
# ----------------------
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
        self.block_zone = {r.block_id: r.zone for r in self.blocks.itertuples(index=False)}
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
            r.block_id: float(r.scale_l) if pd.notna(r.scale_l) else 0.0
            for r in self.blocks.itertuples(index=False)
        }
        self._normalizer = TargetNormalizer(eps=1e-6)

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

        blk_id_val = df_nodes_block["block_id"].iloc[0]
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

        x_in = torch.cat([
            torch.as_tensor(nodes_b[["e_i"]].values, dtype=torch.float32),
            torch.tile(zone_onehot.view(1,-1), (len(nodes_b), 1)),
        ], dim=1)

        targets = self._node_targets(nodes_b)

        # путь к маске: сначала берём из blocks.parquet (mask_path), иначе — {mask_root}/{block_id}.png
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

            data.mask_img = mask_img  # (1,H,W)
            return data
        else:
            return {
                "x": x_in, "edge_index": edge_index, "y": targets,
                "num_nodes": x_in.shape[0], "block_id": blk_id, "zone_id": zone_id,
                "mask_img": mask_img,
            }
        
    def _compute_label_stats(self):
        """Подсчёт частот на текущем split для pos_weight (e/il/hf) и class_weight (s)."""
        nodes = self.nodes
        mask_split = nodes["block_id"].astype(str).isin(self.block_ids)
        df = nodes[mask_split].copy()

        # e
        e = pd.to_numeric(df["e_i"], errors="coerce").fillna(0.0) > 0.5
        p_e = float(e.mean()) if len(df) else 0.0

        # активные узлы
        df_act = df[e]

        # il, hf на активных
        if len(df_act):
            il = df_act["is_living"].astype(bool)
            hf = df_act["has_floors"].astype(bool)
            p_il = float(il.mean())
            p_hf = float(hf.mean())
        else:
            p_il, p_hf = 0.0, 0.0

        # классы формы (на активных)
        if "s_i" in df_act.columns and len(df_act):
            s_vals = pd.to_numeric(df_act["s_i"], errors="coerce").fillna(0).astype(int)
            counts = s_vals.value_counts().reindex([0,1,2,3], fill_value=0).astype(int)
        else:
            counts = pd.Series([0,0,0,0], index=[0,1,2,3])

        self.label_stats = {
            "p_e": max(min(p_e, 0.999999), 1e-6),
            "p_il": max(min(p_il, 0.999999), 1e-6),
            "p_hf": max(min(p_hf, 0.999999), 1e-6),
            "s_counts": counts.to_dict(),  # {0:c0,1:c1,2:c2,3:c3}
        }
    def _one_hot(self, idx: int, K: int) -> np.ndarray: 
        v = np.zeros((K,), dtype=np.float32) 
        if idx is not None and 0 <= idx < K: 
            v[idx] = 1.0 
        return v
    
    def _services_vecs(self, present_raw, cap_raw) -> Tuple[np.ndarray, np.ndarray]: 
        present = _maybe_json_to_list(present_raw) 
        cap = _maybe_json_to_list(cap_raw) 
        # present → multi-hot 
        mhot = np.zeros((self.num_services,), dtype=np.float32) 
        for s in present: 
            sid = self.service2id.get(str(s)) 
            if sid is not None: 
                mhot[sid] = 1.0 
                # capacity → aligned vector (если cap был списком float/ints) 
        cap_vec = np.zeros((self.num_services,), dtype=np.float32) 
        if isinstance(cap, list): # допускаем cap в формате [{name:..., value:...}] или [v0, v1, ...] 
            if cap and isinstance(cap[0], dict): 
                for it in cap: sid = self.service2id.get(str(it.get("name"))) 
                if sid is not None: 
                    try: cap_vec[sid] = float(it.get("value", 0.0)) 
                    except Exception: 
                        pass 
                    else: # если просто список значений, но без имён — запишем в первые позиции 
                        for i,v in enumerate(cap[:self.num_services]): 
                            try: 
                                cap_vec[i] = float(v) 
                            except Exception: 
                                pass 
        return mhot, cap_vec

    def get_pos_weights(self) -> Dict[str, float]:
        """pos_weight = neg/pos для BCE."""
        ls = getattr(self, "label_stats", None) or {}
        def _pw(p):  # p = P(y=1)
            p = max(min(float(p), 0.999999), 1e-6)
            return (1.0 - p) / p
        return {
            "e": _pw(ls.get("p_e", 0.01)),
            "il": _pw(ls.get("p_il", 0.1)),
            "hf": _pw(ls.get("p_hf", 0.5)),
        }

    def get_s_class_weights(self) -> torch.Tensor:
        """Обратные частоты классов для CE (Rect/X/U/L -> 0/1/2/3)."""
        ls = getattr(self, "label_stats", None) or {}
        counts = ls.get("s_counts", {0:1,1:1,2:1,3:1})
        arr = np.array([counts.get(k, 1) for k in [0,1,2,3]], dtype=np.float64)
        arr = np.where(arr > 0, arr, 1.0)
        inv = 1.0 / arr
        inv = inv / inv.mean()  # нормируем, чтобы средний вес ~1
        return torch.tensor(inv, dtype=torch.float32)
        
    def _compute_block_stats_and_strata(self, nbins: int = 4, k_clusters: int = 8):
        """Считает метрики на уровне квартала и подготавливает страты/кластеры для семплинга."""
        # агрегаты по узлам
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

            # гео-статы
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
        # присоединяем scale_l и zone
        meta["scale_l"] = self.blocks.set_index("block_id").reindex(self.block_ids)["scale_l"].values
        meta["zone"]    = self.blocks.set_index("block_id").reindex(self.block_ids)["zone"].astype(str).values

        # страты (квантили)
        meta["bin_occ"]   = _quantile_bins(meta["occupancy"].values, nbins)
        meta["bin_lshare"]= _quantile_bins(meta["living_share_active"].values, nbins)
        meta["bin_floor"] = _quantile_bins(pd.to_numeric(meta["floors_mean"], errors="coerce").fillna(-1).values, nbins)
        meta["bin_scale"] = _quantile_bins(pd.to_numeric(meta["scale_l"],  errors="coerce").fillna(-1).values, nbins)

        # k-means внутри residential (опц.)
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

        self.block_meta = meta  # DataFrame
        # списки индексов по зонам
        self.zone2indices = {}
        for z, grp in meta.groupby("zone"):
            self.zone2indices[str(z)] = [self.block2idx[b] for b in grp.index.tolist() if b in self.block2idx]

    def get_zone_indices(self, zone_label: str) -> List[int]:
        return self.zone2indices.get(str(zone_label), [])

    def get_block_meta(self, idx: int) -> Dict[str, Any]:
        blk = self.idx2block[idx]
        r = self.block_meta.loc[blk]
        return r.to_dict()

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
        # длина эпохи: по умолчанию ≈ покрытие датасета 1 раз
        self._n_batches = int(math.ceil(len(ds) / max(1, self.batch_size))) if epoch_len_batches is None else int(epoch_len_batches)

    def __iter__(self):
        for _ in range(self._n_batches):
            idxs = self.maj.sample(self.k_maj) + self.min.sample(self.k_min)
            random.shuffle(idxs)
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

    def __call__(self, pred: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str,float]]:
        device = pred["e"].device

        # Маски
        e_mask = (y["e"] > 0.5).float()                 # (N,1) — активные узлы
        denom_nodes = e_mask.sum() + 1e-6               # скаляр

        # ----- BCE с pos_weight -----
        pw_e  = torch.tensor(self.posw.get("e", 1.0),  device=device)
        pw_il = torch.tensor(self.posw.get("il", 1.0), device=device)
        pw_hf = torch.tensor(self.posw.get("hf", 1.0), device=device)

        e_loss_t  = F.binary_cross_entropy_with_logits(pred["e"],  y["e"],  pos_weight=pw_e,  reduction='mean')
        il_loss_t = (F.binary_cross_entropy_with_logits(pred["il"], y["il"], pos_weight=pw_il, reduction='none') * e_mask).sum() / denom_nodes
        hf_loss_t = (F.binary_cross_entropy_with_logits(pred["hf"], y["hf"], pos_weight=pw_hf, reduction='none') * e_mask).sum() / denom_nodes

        # ----- Регрессии (по активным) -----
        pos_loss_t = (self.l1(pred["pos"], y["pos"]) * e_mask).sum() / denom_nodes
        sz_loss_t  = (self.l1(pred["sz"],  y["sz"])  * e_mask).sum() / denom_nodes
        phi_loss_t = (self.l1(pred["phi"], y["phi"]) * e_mask).sum() / denom_nodes
        a_loss_t   = (self.l1(pred["a"],   y["a"])   * e_mask).sum() / denom_nodes

        # ----- Жилая площадь (по жилым), уже нормирована -----
        il_mask = y["il"]
        denom_living = il_mask.sum() + 1e-6
        la_loss_t = (self.l1(pred["la"], y["la"]) * il_mask).sum() / denom_living

        # ----- Этажность: корректная маска has_floors==1 & fl>=0 -----
        fl_valid = (y["fl"].view(-1,1) >= 0).float()
        fl_mask  = (y["hf"] > 0.5).float() * fl_valid
        denom_fl = fl_mask.sum() + 1e-6
        fl_tgt   = y["fl"].clamp(min=0).float().view(-1,1)
        fl_pred  = pred["fl"].view(-1,1)
        fl_loss_t = (self.l1(fl_pred, fl_tgt) * fl_mask).sum() / denom_fl

        # ----- Классы формы (CE с весами, по активным) -----
        ce_w = self.ce_weight.to(device) if self.ce_weight is not None else None
        s_loss_vec = F.cross_entropy(pred["s"], y["s"].long(),
                                     weight=ce_w, reduction='none',
                                     label_smoothing=self.label_smoothing)
        s_loss_t = (s_loss_vec * e_mask.view(-1)).sum() / denom_nodes

        # ----- Сервисы: как прежде (но учёт маски для capacity) -----
        if y["sv1"].numel() > 0:
            svc_mask = y.get("svc_mask", torch.zeros_like(y["sv1"]))
            denom_svc = svc_mask.sum() + 1e-6
            if denom_svc.item() > 0:
                svc_loss_t = (self.l1(pred["svc"], y["svc"]) * svc_mask).sum() / denom_svc
            else:
                svc_loss_t = torch.zeros((), device=device)
            sv1_loss_t = (F.binary_cross_entropy_with_logits(pred["sv1"], y["sv1"], reduction='none') * e_mask).sum() / denom_nodes
        else:
            sv1_loss_t = torch.zeros((), device=device)
            svc_loss_t = torch.zeros((), device=device)

        # Сборка и взвешивание
        t_losses = {
            "e": e_loss_t, "pos": pos_loss_t, "sz": sz_loss_t, "phi": phi_loss_t,
            "s": s_loss_t, "a": a_loss_t, "fl": fl_loss_t, "hf": hf_loss_t,
            "il": il_loss_t, "la": la_loss_t, "sv1": sv1_loss_t, "svc": svc_loss_t
        }
        total_t = torch.zeros((), device=device)
        for k, t in t_losses.items():
            if k in self.w:
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
    # пути
    data_dir = args.data_dir
    ckpt_path = args.model_ckpt
    zones_json = args.zones_json
    services_json = args.services_json

    device = args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        log.warning("CUDA недоступна — использую CPU")

    # датасеты
    log.info("Готовлю датасеты…")
    mask_size = tuple([int(_CFG["model"].get("mask_size", 128))]*2)
    mask_root = args.mask_root

    ds_train = HCanonGraphDataset(data_dir, zones_json, services_json,
                                split="train", split_ratio=0.9,
                                mask_size=mask_size, mask_root=mask_root)
    ds_val   = HCanonGraphDataset(data_dir, zones_json, services_json,
                                split="val", split_ratio=0.9,
                                mask_size=mask_size, mask_root=mask_root)

    # Балансировки лоссов из train-сплита
    posw = ds_train.get_pos_weights()
    ce_w = ds_train.get_s_class_weights()
    loss_comp = LossComputer(
        cfg["training"]["loss_weights"],
        posw=posw,
        ce_weight=ce_w,
        label_smoothing=0.05,  # можно 0.0, если не хочешь сглаживание
    )

    if _PYG_OK:
        if args.sampler_mode == "two_stream":
            maj = MajoritySampler(
                ds_train,
                tau=float(args.tau_majority),
                residential_stratify=not args.no_res_stratify,
                use_clusters=not args.no_res_clusters,
            )
            mino = MinoritySampler(ds_train)
            batch_sampler = TwoStreamBatchSampler(
                ds_train,
                batch_size=int(args.batch_size),
                majority_frac=float(args.majority_frac),
                maj_sampler=maj, min_sampler=mino,
                epoch_len_batches=_CFG["sampler"].get("epoch_len_batches"),
            )
            # ВАЖНО: при использовании batch_sampler не задаём batch_size/shuffle
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
                ds_train,
                tau=float(args.tau_majority),
                residential_stratify=not args.no_res_stratify,
                use_clusters=not args.no_res_clusters,
            )
            mino = MinoritySampler(ds_train)
            batch_sampler = TwoStreamBatchSampler(
                ds_train,
                batch_size=int(args.batch_size),
                majority_frac=float(args.majority_frac),
                maj_sampler=maj, min_sampler=mino,
                epoch_len_batches=_CFG["sampler"].get("epoch_len_batches"),
            )
            train_loader = DataLoader(ds_train, batch_sampler=batch_sampler, collate_fn=_collate)
        else:
            train_loader = DataLoader(ds_train, batch_size=1, shuffle=True, collate_fn=_collate)

        val_loader   = DataLoader(ds_val,   batch_size=1, shuffle=False, collate_fn=_collate)
        in_dim = ds_train[0]["x"].shape[1]

    # модель
    log.info("Создаю модель…")
    use_mask  = bool(_CFG["model"].get("use_mask_cnn", True))
    mask_dim  = int(_CFG["model"].get("mask_dim", 64))

    model = GraphModel(
        in_dim=in_dim,
        hidden=int(_CFG["model"]["hidden"]),
        num_services=ds_train.num_services,
        use_mask=use_mask,
        mask_dim=mask_dim,
    )
    log.info(f"Mask CNN enabled={use_mask}; mask_dim={mask_dim}; mask_root={mask_root}; mask_size={mask_size}")

    # TensorBoard
    run_name = f"{cfg.get('experiment','exp')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    runs_dir = os.path.join(os.path.dirname(ckpt_path) or data_dir, "runs", run_name)
    os.makedirs(runs_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=runs_dir)
    log.info(f"TensorBoard лог-директория: {runs_dir}")

    # обучение
    best_val = train_loop(
        model, train_loader, val_loader, device, cfg, ckpt_path, writer,
        loss_comp=loss_comp,
        normalizer=getattr(ds_train, "_normalizer", None)
    )
    writer.close()

    # сохраняем вспомог. файлы рядом с чекпойнтом
    aux_dir = os.path.join(os.path.dirname(ckpt_path) or data_dir, "artifacts")
    os.makedirs(aux_dir, exist_ok=True)
    with open(os.path.join(aux_dir, "zones.json"), "w", encoding="utf-8") as f:
        json.dump(ds_train.zone2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(aux_dir, "services.json"), "w", encoding="utf-8") as f:
        json.dump(ds_train.service2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(aux_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log.info(f"Вспомогательные файлы сохранены → {aux_dir}")

    norm_path = os.path.join(aux_dir, "target_normalizer.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(ds_train._normalizer.to_dict(), f, ensure_ascii=False, indent=2)
    log.info(f"Нормировщик таргетов сохранён → {norm_path}")

    # Загрузка на HF (директория с чекпойнтом и артефактами)
    if args.hf_push:
        repo_id = args.hf_repo_id
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not token:
            log.warning("HF token не найден ни в --hf-token, ни в $HF_TOKEN — пропускаю загрузку")
        else:
            to_push_dir = os.path.commonpath([os.path.abspath(aux_dir), os.path.abspath(ckpt_path)])
            # если чекпойнт вне aux_dir — положим его в artifacts и пушнем целиком
            if not os.path.dirname(ckpt_path).startswith(aux_dir):
                try:
                    import shutil
                    shutil.copy2(ckpt_path, os.path.join(aux_dir, os.path.basename(ckpt_path)))
                except Exception:
                    pass
            push_to_hf(repo_id=repo_id, token=token, folder=aux_dir, private=bool(args.hf_private))

    log.info("Обучение завершено.")

def load_target_normalizer(path: str) -> TargetNormalizer:
    with open(path, "r", encoding="utf-8") as f:
        return TargetNormalizer.from_dict(json.load(f))

@torch.no_grad()
def denorm_predictions(batch, pred: Dict[str, torch.Tensor], normalizer: TargetNormalizer) -> Dict[str, torch.Tensor]:
    """
    Принимает батч и предсказания в НОРМ-формате, возвращает дикт с 'la_real', 'svc_real'.
    Требует наличие batch.scale_l (PyG) или batch["y"]["scale_l"] (fallback).
    """
    if hasattr(batch, "scale_l"):
        scale = batch.scale_l.to(pred["la"].device)  # (N,1)
    else:
        scale = batch["y"]["scale_l"].to(pred["la"].device)

    out = {}
    if "la" in pred:
        la_real = normalizer.decode_la(pred["la"], scale)  # (N,1)
        out["la_real"] = la_real

    if "svc" in pred:
        svc_real = normalizer.decode_svc(pred["svc"])      # (N,S)
        out["svc_real"] = svc_real

    # По желанию: бинаризуем sv1 и зануляем capacities, где presence==0
    if "sv1" in pred and "svc_real" in out:
        sv1_prob = torch.sigmoid(pred["sv1"])
        sv1_bin = (sv1_prob > 0.5).float()
        out["svc_real"] = out["svc_real"] * sv1_bin

    return out


if __name__ == "__main__":
    sys.exit(main())
