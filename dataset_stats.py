#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Stats CLI — собирает релевантную для обучения статистику по данным
из иерархической канонизации (transform.py) и сохраняет в один CSV.

Пример запуска:
  python dataset_stats.py \
    --data-dir ./dataset \
    --out ./dataset/stats_train.csv \
    --split train --split-ratio 0.9 --log-level INFO

CSV-формат: колонки [section, name, metric, value].
  - section: "global" | "zones" | "services" | "shapes" | "geom" | "floors" | "files"
  - name:    ключ для секции (например, zone label или service name), иначе "all"
  - metric:  конкретная метрика
  - value:   числовое значение (float)
"""
from __future__ import annotations
import os, sys, json, math, argparse, logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from services_processing import infer_service_schema_from_nodes

try:
    from tqdm.auto import tqdm
    TQDM = True
except Exception:
    TQDM = False
    def tqdm(x, **k):
        return x

log = logging.getLogger("ds_stats")

def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------- helpers ----------

def _maybe_json_to_list(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return []
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return []

# ---------- core ----------

def main():
    ap = argparse.ArgumentParser("Dataset stats → CSV")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None, help="Путь к CSV-файлу (по умолчанию <data-dir>/stats_<split>.csv)")
    ap.add_argument("--split", choices=["train", "val", "all"], default="train")
    ap.add_argument("--split-ratio", type=float, default=0.9, help="Доля train в сплите train/val")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logger(args.log_level)

    data_dir = args.data_dir
    out_csv = args.out or os.path.join(data_dir, f"stats_{args.split}.csv")

    # --- load tables ---
    blocks_pq   = os.path.join(data_dir, "blocks.parquet")
    nodes_pq    = os.path.join(data_dir, "nodes_fixed.parquet")
    edges_pq    = os.path.join(data_dir, "edges.parquet")
    branches_pq = os.path.join(data_dir, "branches.parquet")

    if not os.path.exists(blocks_pq) or not os.path.exists(nodes_pq) or not os.path.exists(edges_pq):
        log.error("Не найдены обязательные parquet-файлы (blocks/nodes_fixed/edges). Проверьте --data-dir.")
        sys.exit(2)

    log.info("Читаю parquet…")
    blocks = pd.read_parquet(blocks_pq)
    nodes  = pd.read_parquet(nodes_pq)
    edges  = pd.read_parquet(edges_pq)
    branches = None
    if os.path.exists(branches_pq):
        try:
            branches = pd.read_parquet(branches_pq)
        except Exception:
            branches = None

    # --- split ---
    blk_ids = sorted(blocks["block_id"].astype(str).unique().tolist())
    n = len(blk_ids)
    n_train = int(round(n * float(args.split_ratio)))
    if args.split == "train":
        blk_ids = blk_ids[:n_train]
    elif args.split == "val":
        blk_ids = blk_ids[n_train:]
    # else: all
    blk_set = set(blk_ids)

    # фильтруем
    blocks = blocks[blocks["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)
    nodes  = nodes[nodes["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)
    edges  = edges[edges["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)
    if branches is not None:
        branches = branches[branches["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)

    # --- global basics ---
    rows: List[Dict[str, Any]] = []
    def add(section: str, name: str, metric: str, value: float):
        rows.append({"section": section, "name": name, "metric": metric, "value": float(value)})

    add("global", "all", "blocks", len(blocks))
    add("global", "all", "nodes", len(nodes))
    add("global", "all", "edges", len(edges))
    if branches is not None:
        add("global", "all", "branches", len(branches))

    # маски
    masks_ok = 0
    if "mask_path" in blocks.columns:
        for p in blocks["mask_path"].astype(str).values.tolist():
            try:
                if os.path.exists(p):
                    masks_ok += 1
            except Exception:
                pass
        add("files", "masks", "exists_count", masks_ok)

    # timeouts / orphans
    tout_p = os.path.join(data_dir, "timeout.geojson")
    if os.path.exists(tout_p):
        try:
            import geopandas as gpd
            gdf = gpd.read_file(tout_p)
            add("files", "timeout", "features", len(gdf))
            if "kind" in gdf.columns:
                add("files", "timeout", "blocks", int((gdf["kind"] == "block").sum()))
                add("files", "timeout", "buildings", int((gdf["kind"] == "building").sum()))
        except Exception:
            add("files", "timeout", "features", 0)
    for nm in ("orphaned_blocks.geojson", "orphaned_buildings.geojson"):
        p = os.path.join(data_dir, nm)
        if os.path.exists(p):
            try:
                import geopandas as gpd
                gdf = gpd.read_file(p)
                add("files", nm, "features", len(gdf))
            except Exception:
                add("files", nm, "features", 0)

    # --- per-block occupancy & graph ---
    grp = nodes.groupby("block_id")
    occ_rates = []
    degs = edges.groupby("block_id").size().reindex(blocks["block_id"], fill_value=0).values
    for bid, sub in grp:
        total = len(sub)
        active = int(np.nan_to_num(sub["e_i"].values, nan=0).sum())
        if total > 0:
            occ_rates.append(active / total)
    if occ_rates:
        occ = np.asarray(occ_rates, dtype=float)
        add("global", "all", "occupancy_mean", float(occ.mean()))
        add("global", "all", "occupancy_std", float(occ.std()))
        add("global", "all", "occupancy_p25", float(np.percentile(occ, 25)))
        add("global", "all", "occupancy_p50", float(np.percentile(occ, 50)))
        add("global", "all", "occupancy_p75", float(np.percentile(occ, 75)))
    if len(blocks):
        add("global", "all", "edges_per_block_mean", float(np.mean(degs)))

    # scale_l
    if "scale_l" in blocks.columns:
        sl = pd.to_numeric(blocks["scale_l"], errors="coerce").fillna(0.0).values
        add("global", "all", "scale_l_mean", float(np.mean(sl)))
        add("global", "all", "scale_l_std", float(np.std(sl)))

    # --- zones ---
    if "zone" in blocks.columns:
        zc = blocks["zone"].fillna("nan").astype(str).value_counts()
        for z,v in zc.items():
            add("zones", z, "blocks", int(v))
        # пер-зонная агрегация по узлам
        blk_zone = blocks.set_index("block_id")["zone"].to_dict()
        nodes_z = nodes.copy()
        nodes_z["zone"] = nodes_z["block_id"].map(blk_zone).fillna("nan").astype(str)
        for z, sub in nodes_z.groupby("zone"):
            e = np.nan_to_num(sub["e_i"].values, nan=0.0)
            add("zones", z, "active_nodes", float(e.sum()))
            # living
            il = np.nan_to_num(sub.get("is_living", pd.Series([0]*len(sub))).astype(bool).values, nan=0.0)
            add("zones", z, "living_share_active", float((il * e).sum() / (e.sum() + 1e-6)))
            # floors
            hf = np.nan_to_num(sub.get("has_floors", pd.Series([0]*len(sub))).astype(bool).values, nan=0.0)
            fl = pd.to_numeric(sub.get("floors_num", pd.Series([-1]*len(sub))), errors="coerce").fillna(-1).values
            mask = (hf > 0.5) & (e > 0.5) & (fl >= 0)
            if mask.any():
                add("zones", z, "floors_mean", float(fl[mask].mean()))
            # living area (только is_living & e_i)
            la = pd.to_numeric(sub.get("living_area", pd.Series([0.0]*len(sub))), errors="coerce").fillna(0.0).values
            add("zones", z, "living_area_sum", float((la * il * e).sum()))

    # --- shapes distribution (по активным узлам) ---
    if "s_i" in nodes.columns:
        active_mask = np.nan_to_num(nodes["e_i"].values, nan=0.0) > 0.5
        s_vals = pd.to_numeric(nodes.loc[active_mask, "s_i"], errors="coerce").fillna(-1).astype(int)
        vc = s_vals.value_counts()
        label_map = {0:"Rect", 1:"L", 2:"U", 3:"X", -1:"unknown"}
        for k,v in vc.items():
            add("shapes", label_map.get(int(k), str(int(k))), "count", int(v))

    # --- geometry stats (по активным узлам) ---
    if len(nodes):
        e = np.nan_to_num(nodes["e_i"].values, nan=0.0) > 0.5
        for col, mname in [("posx","posx"),("posy","posy"),("size_x","size_x"),("size_y","size_y"),("phi_resid","phi")]:
            if col in nodes.columns:
                v = pd.to_numeric(nodes.loc[e, col], errors="coerce").dropna().values
                if v.size:
                    add("geom", "all", f"{mname}_mean", float(v.mean()))
                    add("geom", "all", f"{mname}_std", float(v.std()))
                    add("geom", "all", f"{mname}_min", float(v.min()))
                    add("geom", "all", f"{mname}_max", float(v.max()))

    # --- floors stats ---
    if "has_floors" in nodes.columns:
        e = np.nan_to_num(nodes["e_i"].values, nan=0.0) > 0.5
        hf = np.nan_to_num(nodes.get("has_floors", pd.Series([0]*len(nodes))).astype(bool).values, nan=0.0)
        add("floors", "all", "has_floors_share_active", float((hf[e]).mean() if e.any() else 0.0))
        if "floors_num" in nodes.columns:
            fl = pd.to_numeric(nodes["floors_num"], errors="coerce").fillna(-1).astype(int)
            mask = (hf > 0.5) & e & (fl >= 0)
            if mask.any():
                add("floors", "all", "floors_mean", float(fl[mask].mean()))
                # топ-20 этажностей
                top = fl[mask].value_counts().head(20)
                for k,v in top.items():
                    add("floors", f"{int(k)}", "count", int(v))

    # --- services (если есть) ---
    schema = infer_service_schema_from_nodes(nodes)
    if schema:
        e_mask = np.nan_to_num(nodes["e_i"].values, nan=0.0) > 0.5
        active_nodes = nodes.loc[e_mask].copy()
        present_cnt: Dict[str, int] = {}
        cap_sum: Dict[str, float] = {}
        for item in schema:
            name = item.get("name")
            has_col = item.get("has_column")
            cap_col = item.get("capacity_column")
            if not has_col or has_col not in active_nodes.columns:
                continue
            has_vals = active_nodes[has_col].fillna(False).astype(bool)
            present_cnt[name] = int(has_vals.sum())
            if cap_col and cap_col in active_nodes.columns:
                caps = pd.to_numeric(active_nodes[cap_col], errors="coerce").fillna(0.0)
                cap_sum[name] = float(caps[has_vals].sum())
            else:
                cap_sum[name] = 0.0
        for s, c in present_cnt.items():
            add("services", s, "present_count", int(c))
        for s, v in cap_sum.items():
            add("services", s, "capacity_sum", float(v))

    # --- save CSV ---
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"Сохранено: {out_csv}  (строк: {len(df)})")

if __name__ == "__main__":
    sys.exit(main())
