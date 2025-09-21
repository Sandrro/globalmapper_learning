#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
per_block_worker.py — обработка одного квартала (block_id) и запись per-block parquet

Что делает:
- Рисует маску квартала 64×64, оценивает масштаб scale_l.
- Строит ветви (straight skeleton → ветви; fallback — главная ось MRR),
  опционально фильтрует/упрощает ветви.
- Канонизирует здания относительно ближайшей ветви (pos/size/phi/aspect),
  вычисляет s_i (тип формы) и a_i (occupancy_ratio).
- Раскладывает N слотов по ветвям и равномерно выбирает здания вдоль ветви.
- Пишет per-block parquet: blocks.parquet, branches.parquet, nodes_fixed.parquet, edges.parquet.
- Обогащает рёбра через KNN в канон-пространстве posx,posy (по умолчанию k=6).
- Мягкий UNIX-таймаут на обработку одного квартала.

Вход в виде словаря task (обязательные ключи):
  {
    "block_id": str,
    "zone": Any,
    "poly_wkb": bytes,                # геометрия квартала в WKB
    "bldgs_rows": List[Dict[str,Any]],# список зданий: geometry(WKB), floors_num, living_area, ...
    "N": int,                         # число слотов
    "mask_size": int,
    "mask_dir": str,
    "timeout_sec": int,
    "out_dir": str,
    # опционально
    "service_map": Dict[str,int],     # для совместимости; не используется напрямую здесь
    "min_branch_len": float,          # фильтр коротких ветвей
    "branch_simplify_tol": float,     # упрощение ветвей
    "knn_k": int,                     # k соседей для KNN-обогащения рёбер (по pos), по умолчанию 6
  }
"""
from __future__ import annotations

import math
import os
import signal
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Polygon

from geo_common import (
    save_block_mask_64,
    block_scale_l,
    geom_from_wkb,
)
from skeleton_branches import (
    build_straight_skeleton,
    skeleton_to_graph,
    extract_branches,
    simplify_branches_if_needed,
    filter_short_branches,
)
from canon_features import (
    canon_by_branch,
    classify_shape,
    occupancy_ratio,
)

# ----------------- timeout helper -----------------

@contextmanager
def time_limit(seconds: int):
    """Unix-only soft таймаут. Безопасно в воркере-процессе."""
    def _handler(signum, frame):  # noqa: ARG001
        raise TimeoutError("worker timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ----------------- helpers -----------------

import json
import numpy as np
import pandas as pd

def _as_list(val):
    """Аккуратно привести значение к python-списку."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (str, bytes)):
        s = val.decode("utf-8") if isinstance(val, bytes) else val
        s = s.strip()
        if s in ("", "[]", "{}"):
            return []
        try:
            parsed = json.loads(s)
            return _as_list(parsed)
        except Exception:
            return []
    # dict → возьмём значения >0 по возрастанию ключа
    if isinstance(val, dict):
        return [v for k, v in sorted(val.items())]
    return []

def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        geoms = [g for g in geom.geoms if (g is not None and not g.is_empty)]
        return max(geoms, key=lambda g: g.area) if geoms else None
    try:
        hull = geom.convex_hull
        return hull if isinstance(hull, Polygon) else None
    except Exception:
        return None


def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        try:
            return int(x.size)  # type: ignore[attr-defined]
        except Exception:
            return 0


def _block_dir(out_dir: str, blk_id: str) -> str:
    return os.path.join(out_dir, "by_block", str(blk_id))


def _write_block_parquets(
    out_dir: str,
    blk_id: str,
    block_row: Dict[str, Any],
    branches_rows: List[Dict[str, Any]],
    nodes_rows: List[Dict[str, Any]],
    edges_rows: List[Dict[str, Any]],
) -> None:
    bdir = _block_dir(out_dir, blk_id)
    os.makedirs(bdir, exist_ok=True)
    pd.DataFrame([block_row]).to_parquet(os.path.join(bdir, "blocks.parquet"), index=False)
    pd.DataFrame(branches_rows).to_parquet(os.path.join(bdir, "branches.parquet"), index=False)
    nodes_df = pd.DataFrame(nodes_rows)
    if "floors_num" in nodes_df.columns:
        nodes_df["floors_num"] = nodes_df["floors_num"].astype("Int64")
    nodes_df.to_parquet(os.path.join(bdir, "nodes_fixed.parquet"), index=False)
    pd.DataFrame(edges_rows).to_parquet(os.path.join(bdir, "edges.parquet"), index=False)


# ----------- slot allocation helpers -----------

def allocate_slots_per_branch(counts: List[int], N: int) -> List[int]:
    total = sum(counts)
    if total == 0:
        return [0] * len(counts)
    base = [max(1, int(round(c / total * N))) for c in counts]
    diff = N - sum(base)
    rema = [c / total * N - b for c, b in zip(counts, base)]
    order = np.argsort(rema)[::-1]
    i = 0
    while diff != 0 and len(order) > 0:
        idx = int(order[i % len(order)])
        if diff > 0:
            base[idx] += 1
            diff -= 1
        else:
            if base[idx] > 1:
                base[idx] -= 1
                diff += 1
        i += 1
    return base


def evenly_subsample(idx: np.ndarray, k: int) -> np.ndarray:
    if len(idx) <= k:
        return idx
    pos = np.linspace(0, len(idx) - 1, k).round().astype(int)
    return idx[pos]


# ----------------- main worker -----------------

def worker_process_block(task: Dict[str, Any]) -> Dict[str, Any]:
    blk_id: str = task["block_id"]
    zone = task["zone"]
    poly_wkb: bytes = task["poly_wkb"]
    rows_bldgs: List[Dict[str, Any]] = task["bldgs_rows"]
    N: int = int(task["N"])  # число слотов
    mask_size: int = int(task["mask_size"])
    mask_dir: str = task["mask_dir"]
    timeout_sec: int = int(task.get("timeout_sec", 300))
    out_dir: str = task["out_dir"]

    # доп. параметры
    min_branch_len: float = float(task.get("min_branch_len", 0.0))
    branch_simplify_tol: Optional[float] = task.get("branch_simplify_tol")
    if branch_simplify_tol is not None:
        branch_simplify_tol = float(branch_simplify_tol)
    knn_k: int = int(task.get("knn_k", 6))

    mask_path = os.path.join(mask_dir, f"{blk_id}.png")

    # геометрия квартала
    try:
        poly = geom_from_wkb(poly_wkb)
    except Exception:
        poly = None
    poly = _largest_polygon(poly)

    # mask
    ok_mask = False
    if poly is not None:
        try:
            ok_mask = bool(save_block_mask_64(poly, mask_path, size=mask_size))
        except Exception:
            ok_mask = False
    if not ok_mask:
        try:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            from PIL import Image  # локальный импорт, если Pillow не нужен выше

            Image.new("L", (mask_size, mask_size), 0).save(mask_path)
        except Exception:
            pass
        if poly is None:
            payload = _empty_block_payload(blk_id, zone, mask_path, N)
            _write_block_parquets(out_dir, blk_id, payload["block"], [], payload["nodes"], [])
            return {"block_id": blk_id, "status": "ok_empty"}

    if not rows_bldgs:
        scale_l = float(block_scale_l(poly, [])) if poly is not None else 0.0
        payload = _empty_block_payload(blk_id, zone, mask_path, N)
        payload["block"]["scale_l"] = scale_l
        _write_block_parquets(out_dir, blk_id, payload["block"], [], payload["nodes"], [])
        return {"block_id": blk_id, "status": "ok_empty"}

    try:
        with time_limit(timeout_sec):
            # 1) skeleton & branches
            try:
                skel = build_straight_skeleton(poly)
                G = skeleton_to_graph(skel)
                branches = extract_branches(G)
            except Exception:
                branches = []

            # упрощение/фильтрация ветвей, если заданы параметры
            branches = simplify_branches_if_needed(branches, branch_simplify_tol)
            branches = filter_short_branches(branches, min_branch_len)

            if not branches:
                # fallback — главная диагональ MRR
                mrr = poly.minimum_rotated_rectangle
                coords = list(mrr.exterior.coords)
                branches = [LineString([coords[0], coords[2]])]

            valid_branches = [
                (i, br) for i, br in enumerate(branches) if isinstance(br, LineString) and br.length > 0
            ]
            scale_l = float(block_scale_l(poly, [br for _, br in valid_branches]))

            # 2) buildings from WKB
            g_geoms, attrs = [], []
            for r in rows_bldgs:
                try:
                    g = geom_from_wkb(r["geometry"])  # type: ignore[index]
                except Exception:
                    g = None
                if g is None or g.is_empty:
                    continue
                g_geoms.append(g)
                a = dict(r)
                a.pop("geometry", None)
                attrs.append(a)
            if len(g_geoms) == 0:
                payload = _empty_block_payload(blk_id, zone, mask_path, N)
                payload["block"]["scale_l"] = scale_l
                _write_block_parquets(out_dir, blk_id, payload["block"], [], payload["nodes"], [])
                return {"block_id": blk_id, "status": "ok_empty"}

            df = pd.DataFrame(attrs)
            gdf = gpd.GeoDataFrame(df, geometry=g_geoms, crs=None)

            # 3) nearest branch by centroid
            cent = gdf.geometry.centroid
            nearest_idx: List[int] = []
            for p in cent:
                dists = [br.distance(p) for _, br in valid_branches]
                nearest_idx.append(int(np.argmin(dists)) if dists else 0)
            gdf["branch_local_id"] = nearest_idx

            # 4) per-branch features
            per_branch: Dict[int, Dict[str, Any]] = {}
            branches_rows = [
                {"block_id": blk_id, "branch_local_id": i, "length": float(br.length)} for i, br in valid_branches
            ]
            counts: List[int] = []
            branch_ids_sorted: List[int] = []
            for j, br in valid_branches:
                rows = gdf[gdf["branch_local_id"] == j].reset_index(drop=True)
                if rows.empty:
                    per_branch[j] = {
                        "rows": rows,
                        "pos": np.zeros((0, 2)),
                        "size": np.zeros((0, 2)),
                        "phi": np.zeros((0,)),
                        "aspect": np.zeros((0,)),
                        "s": np.zeros((0,), int),
                        "a": np.zeros((0,), float),
                    }
                    cnt = 0
                else:
                    pos, size, phi, aspect = canon_by_branch(list(rows.geometry.values), poly, br)
                    s_labels, a_vals = [], []
                    for geom in rows.geometry.values:
                        try:
                            s_labels.append(int(classify_shape(geom)))
                        except Exception:
                            s_labels.append(0)
                        try:
                            a_vals.append(float(occupancy_ratio(geom)))
                        except Exception:
                            a_vals.append(0.0)
                    per_branch[j] = {
                        "rows": rows,
                        "pos": pos,
                        "size": size,
                        "phi": phi,
                        "aspect": aspect,
                        "s": np.array(s_labels, int),
                        "a": np.array(a_vals, float),
                    }
                    cnt = len(rows)
                counts.append(cnt)
                branch_ids_sorted.append(j)

            # 5) allocate N slots & choose
            alloc = allocate_slots_per_branch(counts, N)
            slot_meta: List[Tuple[int, int, int]] = []  # (slot_id, branch_id, local_index)
            slot = 0
            for j, cnt, kslots in zip(branch_ids_sorted, counts, alloc):
                feat = per_branch[j]
                if cnt == 0 or kslots == 0:
                    continue
                order = np.argsort(feat["pos"][:, 0])
                chosen = evenly_subsample(order, kslots)
                posx = feat["pos"][chosen, 0]
                ord2 = np.argsort(posx)
                for k in chosen[ord2]:
                    slot_meta.append((slot, j, int(k)))
                    slot += 1
                    if slot >= N:
                        break
                if slot >= N:
                    break

            # 6) nodes
            nodes: List[Dict[str, Any]] = []
            for slot_id, j, k in slot_meta:
                feat = per_branch[j]
                r = feat["rows"].iloc[k]

                s_vec = feat.get("s", [])
                a_vec = feat.get("a", [])
                s_i = int(s_vec[k]) if isinstance(s_vec, (list, tuple, np.ndarray)) and _safe_len(s_vec) > k else 0
                a_i = float(a_vec[k]) if isinstance(a_vec, (list, tuple, np.ndarray)) and _safe_len(a_vec) > k else 0.0

                pres = _as_list(r.get("services_present", []))
                caps = _as_list(r.get("services_capacity", []))

                nodes.append(
                    {
                        "block_id": blk_id,
                        "slot_id": slot_id,
                        "e_i": 1,
                        "branch_local_id": int(j),
                        "posx": float(feat["pos"][k, 0]),
                        "posy": float(feat["pos"][k, 1]),
                        "size_x": float(feat["size"][k, 0]),
                        "size_y": float(feat["size"][k, 1]),
                        "phi_resid": float(feat["phi"][k]),
                        "s_i": s_i,
                        "a_i": a_i,
                        "floors_num": (pd.NA if pd.isna(r.get("floors_num")) else int(r.get("floors_num"))),
                        "living_area": float(r.get("living_area", 0.0)),
                        "is_living": bool(r.get("is_living", False)),
                        "has_floors": bool(r.get("has_floors", False)),
                        "services_present": pres,
                        "services_capacity": caps,
                        "aspect_ratio": float(feat["aspect"][k]) if _safe_len(feat.get("aspect", [])) > k else 0.0,
                    }
                )

            for slot_id in range(len(nodes), N):
                nodes.append(
                    {
                        "block_id": blk_id,
                        "slot_id": slot_id,
                        "e_i": 0,
                        "branch_local_id": -1,
                        "posx": 0.0,
                        "posy": 0.0,
                        "size_x": 0.0,
                        "size_y": 0.0,
                        "phi_resid": 0.0,
                        "s_i": 0,
                        "a_i": 0.0,
                        "floors_num": pd.NA,
                        "living_area": 0.0,
                        "is_living": False,
                        "has_floors": False,
                        "services_present": [],
                        "services_capacity": [],
                        "aspect_ratio": 0.0,
                    }
                )

            # 7) edges — внутри ветвей и между соседними ветвями
            edges: List[Dict[str, Any]] = []
            slots_by_branch: Dict[int, List[Tuple[int, int]]] = {}
            for slot_id, j, k in slot_meta:
                slots_by_branch.setdefault(j, []).append((slot_id, k))

            # внутри ветви — соседние по x
            for j, lst in slots_by_branch.items():
                feat = per_branch[j]
                lst_sorted = sorted(lst, key=lambda t: float(feat["pos"][t[1], 0]))
                for a in range(len(lst_sorted) - 1):
                    s1 = lst_sorted[a][0]
                    s2 = lst_sorted[a + 1][0]
                    edges.append({"block_id": blk_id, "src_slot": s1, "dst_slot": s2})
                    edges.append({"block_id": blk_id, "src_slot": s2, "dst_slot": s1})

            # между ветвями — по рангу вдоль ветви
            for idx in range(len(branch_ids_sorted) - 1):
                j1 = branch_ids_sorted[idx]
                j2 = branch_ids_sorted[idx + 1]
                if j1 not in slots_by_branch or j2 not in slots_by_branch:
                    continue
                f1 = per_branch[j1]
                f2 = per_branch[j2]
                A = sorted(slots_by_branch[j1], key=lambda t: float(f1["pos"][t[1], 0]))
                B = sorted(slots_by_branch[j2], key=lambda t: float(f2["pos"][t[1], 0]))
                m = min(len(A), len(B))
                for rnk in range(m):
                    a = int(round((len(A) - 1) * rnk / max(m - 1, 1)))
                    b = int(round((len(B) - 1) * rnk / max(m - 1, 1)))
                    s1 = A[a][0]
                    s2 = B[b][0]
                    edges.append({"block_id": blk_id, "src_slot": s1, "dst_slot": s2})
                    edges.append({"block_id": blk_id, "src_slot": s2, "dst_slot": s1})

            # 8) KNN-обогащение рёбер по канон-пространству posx,posy
            try:
                active = [
                    (n["slot_id"], float(n["posx"]), float(n["posy"])) for n in nodes if int(n.get("e_i", 0)) == 1
                ]
                if len(active) >= 2 and int(knn_k) > 0:
                    k = int(min(knn_k, len(active) - 1))
                    slot_ids = np.asarray([a[0] for a in active], dtype=int)
                    P = np.asarray([[a[1], a[2]] for a in active], dtype=float)
                    D2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(axis=2)
                    np.fill_diagonal(D2, np.inf)
                    neigh_idx = np.argpartition(D2, kth=k, axis=1)[:, :k]
                    existing = set((e["src_slot"], e["dst_slot"]) for e in edges)
                    for i, row in enumerate(neigh_idx):
                        src = int(slot_ids[i])
                        for j in row:
                            dst = int(slot_ids[int(j)])
                            if src == dst:
                                continue
                            if (src, dst) not in existing:
                                edges.append({"block_id": blk_id, "src_slot": src, "dst_slot": dst})
                                existing.add((src, dst))
                            if (dst, src) not in existing:
                                edges.append({"block_id": blk_id, "src_slot": dst, "dst_slot": src})
                                existing.add((dst, src))
            except Exception:
                pass

            block_row = {
                "block_id": blk_id,
                "zone": zone,
                "n_buildings": int(len(gdf)),
                "n_branches": int(len(valid_branches)),
                "scale_l": scale_l,
                "mask_path": mask_path,
            }

            _write_block_parquets(out_dir, blk_id, block_row, branches_rows, nodes, edges)
            return {"block_id": blk_id, "status": "ok"}

    except TimeoutError:
        return {
            "status": "timeout",
            "block_id": blk_id,
            "zone": zone,
            "poly_wkb": poly_wkb,
            "bldgs_rows": rows_bldgs,
        }


# ----------------- empty payload -----------------

def _empty_block_payload(blk_id: str, zone, mask_path: str, N: int) -> Dict[str, Any]:
    block_row = {
        "block_id": blk_id,
        "zone": zone,
        "n_buildings": 0,
        "n_branches": 0,
        "scale_l": 0.0,
        "mask_path": mask_path,
    }
    nodes = [
        {
            "block_id": blk_id,
            "slot_id": i,
            "e_i": 0,
            "branch_local_id": -1,
            "posx": 0.0,
            "posy": 0.0,
            "size_x": 0.0,
            "size_y": 0.0,
            "phi_resid": 0.0,
            "s_i": 0,
            "a_i": 0.0,
            "floors_num": pd.NA,
            "living_area": 0.0,
            "is_living": False,
            "has_floors": False,
            "services_present": [],
            "services_capacity": [],
            "aspect_ratio": 0.0,
        }
        for i in range(N)
    ]
    return {"block": block_row, "branches": [], "nodes": nodes, "edges": []}
