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

Опции:
  --no-services   : пропустить секцию сервисов (экономит память/время)
  --batch-size    : размер батча при потоковой обработке сервисов (по умолчанию 200k)
"""

from __future__ import annotations
import os, sys, json, math, argparse, logging
from typing import Any, Dict, List, Tuple
from collections import defaultdict

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

# ---------- logging ----------

def setup_logger(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ---------- parquet helpers ----------

def _get_parquet_columns(path: str) -> List[str]:
    """Считать список колонок из Parquet без загрузки данных."""
    try:
        import pyarrow.parquet as pq
        sch = pq.read_schema(path)
        return list(sch.names)
    except Exception:
        # Фолбэк: читаем только заголовок целиком (может быть дороже)
        try:
            df = pd.read_parquet(path, engine="pyarrow")
            return list(df.columns)
        except Exception:
            df = pd.read_parquet(path)  # последний шанс (fastparquet)
            return list(df.columns)

def _read_parquet(path: str, columns: List[str] | None = None) -> pd.DataFrame:
    """Чтение Parquet с Arrow-бэкендом, если доступен; иначе обычное чтение."""
    try:
        return pd.read_parquet(path, columns=columns, dtype_backend="pyarrow")
    except Exception:
        return pd.read_parquet(path, columns=columns)

def _to_float_np(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy()

def _to_bool_np(series: pd.Series) -> np.ndarray:
    # гарантированное преобразование в булев тип без object
    return series.fillna(False).astype("boolean").to_numpy()

# ---------- CSV row helper ----------

def _rows_add(rows: List[Dict[str, Any]], section: str, name: str, metric: str, value: float):
    rows.append({"section": section, "name": name, "metric": metric, "value": float(value)})

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser("Dataset stats → CSV")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=None, help="Путь к CSV-файлу (по умолчанию <data-dir>/stats_<split>.csv)")
    ap.add_argument("--split", choices=["train", "val", "all"], default="train")
    ap.add_argument("--split-ratio", type=float, default=0.9, help="Доля train в сплите train/val")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--no-services", action="store_true", help="Не считать метрики по сервисам")
    ap.add_argument("--batch-size", type=int, default=200_000, help="Размер батча при потоковой обработке сервисов")
    args = ap.parse_args()

    setup_logger(args.log_level)

    data_dir = args.data_dir
    out_csv = args.out or os.path.join(data_dir, f"stats_{args.split}.csv")

    # --- paths ---
    blocks_pq   = os.path.join(data_dir, "blocks.parquet")
    nodes_pq    = os.path.join(data_dir, "nodes_fixed.parquet")
    edges_pq    = os.path.join(data_dir, "edges.parquet")
    branches_pq = os.path.join(data_dir, "branches.parquet")

    if not os.path.exists(blocks_pq) or not os.path.exists(nodes_pq) or not os.path.exists(edges_pq):
        log.error("Не найдены обязательные parquet-файлы (blocks/nodes_fixed/edges). Проверьте --data-dir.")
        return 2

    # --- available columns & selective reading ---
    blocks_cols_avail = set(_get_parquet_columns(blocks_pq))
    nodes_cols_avail  = set(_get_parquet_columns(nodes_pq))
    edges_cols_avail  = set(_get_parquet_columns(edges_pq))
    branches_cols_avail = set(_get_parquet_columns(branches_pq)) if os.path.exists(branches_pq) else set()

    # минимальные колонки для метрик (без сервисов!)
    BLOCKS_NEED = [c for c in ["block_id", "zone", "scale_l", "mask_path"] if c in blocks_cols_avail]
    NODES_NEED  = [c for c in [
        "block_id", "e_i", "s_i",
        "posx", "posy", "size_x", "size_y", "phi_resid",
        "has_floors", "floors_num",
        "is_living", "living_area",
    ] if c in nodes_cols_avail]
    EDGES_NEED  = [c for c in ["block_id"] if c in edges_cols_avail]
    BRANCHES_NEED = [c for c in ["block_id"] if c in branches_cols_avail]

    log.info("Читаю parquet (выборочно по колонкам)…")
    blocks = _read_parquet(blocks_pq, BLOCKS_NEED)
    nodes  = _read_parquet(nodes_pq,  NODES_NEED)
    edges  = _read_parquet(edges_pq,  EDGES_NEED)
    branches = None
    if os.path.exists(branches_pq):
        try:
            branches = _read_parquet(branches_pq, BRANCHES_NEED)
        except Exception:
            branches = None

    # --- split by blocks ---
    blk_ids_all = sorted(blocks["block_id"].astype(str).unique().tolist())
    n = len(blk_ids_all)
    n_train = int(round(n * float(args.split_ratio)))
    if args.split == "train":
        blk_ids = blk_ids_all[:n_train]
    elif args.split == "val":
        blk_ids = blk_ids_all[n_train:]
    else:
        blk_ids = blk_ids_all
    blk_set = set(blk_ids)

    # фильтруем без копий по возможности
    blocks = blocks[blocks["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)
    nodes  = nodes[nodes["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)
    edges  = edges[edges["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)
    if branches is not None:
        branches = branches[branches["block_id"].astype(str).isin(blk_set)].reset_index(drop=True)

    # --- accumulate rows ---
    rows: List[Dict[str, Any]] = []
    add = lambda section, name, metric, value: _rows_add(rows, section, name, metric, value)

    # --- global basics ---
    add("global", "all", "blocks", len(blocks))
    add("global", "all", "nodes", len(nodes))
    add("global", "all", "edges", len(edges))
    if branches is not None:
        add("global", "all", "branches", len(branches))

    # маски
    if "mask_path" in blocks.columns:
        masks_ok = 0
        # избегаем try/except в цикле по возможности
        for p in blocks["mask_path"].astype(str).values.tolist():
            try:
                if os.path.exists(p):
                    masks_ok += 1
            except Exception:
                pass
        add("files", "masks", "exists_count", masks_ok)

    # timeouts / orphans (обычно маленькие)
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

    # --- per-block occupancy & edges per block ---
    if len(nodes):
        # аккуратно извлекаем e_i и block_id как массивы
        e_vals = _to_float_np(nodes["e_i"], default=0.0) if "e_i" in nodes.columns else np.zeros(len(nodes), dtype=float)
        block_ids_np = nodes["block_id"].astype(str).to_numpy()

        # считаем активность на блок без тяжелых подтаблиц
        df_e = pd.DataFrame({"block_id": block_ids_np, "e": e_vals})
        grp_e = df_e.groupby("block_id")["e"]
        active_sum = grp_e.sum()
        total_cnt  = grp_e.size()
        occ_series = (active_sum / total_cnt.replace(0, np.nan)).dropna()
        if not occ_series.empty:
            occ = occ_series.to_numpy(dtype=float)
            add("global", "all", "occupancy_mean", float(occ.mean()))
            add("global", "all", "occupancy_std", float(occ.std()))
            add("global", "all", "occupancy_p25", float(np.percentile(occ, 25)))
            add("global", "all", "occupancy_p50", float(np.percentile(occ, 50)))
            add("global", "all", "occupancy_p75", float(np.percentile(occ, 75)))

    if len(blocks) and len(edges):
        degs = edges.groupby("block_id").size().reindex(blocks["block_id"], fill_value=0).values
        add("global", "all", "edges_per_block_mean", float(np.mean(degs)) if degs.size else 0.0)

    # scale_l
    if "scale_l" in blocks.columns:
        sl = pd.to_numeric(blocks["scale_l"], errors="coerce").fillna(0.0).to_numpy()
        if sl.size:
            add("global", "all", "scale_l_mean", float(sl.mean()))
            add("global", "all", "scale_l_std", float(sl.std()))

    # --- zones ---
    if "zone" in blocks.columns and "block_id" in blocks.columns:
        zc = blocks["zone"].fillna("nan").astype(str).value_counts()
        for z, v in zc.items():
            add("zones", z, "blocks", int(v))

        # map block_id -> zone
        blk_zone = blocks.set_index("block_id")["zone"].astype(str).to_dict()
        zone_of_node = nodes["block_id"].map(blk_zone).fillna("nan").astype(str) if len(nodes) else pd.Series([], dtype=str)

        if len(nodes):
            e_arr  = _to_float_np(nodes["e_i"], default=0.0) if "e_i" in nodes.columns else np.zeros(len(nodes), dtype=float)
            il_arr = _to_float_np(nodes["is_living"], default=0.0) if "is_living" in nodes.columns else np.zeros(len(nodes), dtype=float)
            hf_arr = _to_bool_np(nodes["has_floors"]) if "has_floors" in nodes.columns else np.zeros(len(nodes), dtype=bool)
            fl_arr = pd.to_numeric(nodes["floors_num"], errors="coerce").fillna(-1).to_numpy(dtype=int) if "floors_num" in nodes.columns else np.full(len(nodes), -1, dtype=int)
            la_arr = _to_float_np(nodes["living_area"], default=0.0) if "living_area" in nodes.columns else np.zeros(len(nodes), dtype=float)

            for z in zone_of_node.unique():
                mask_z = (zone_of_node.values == z)
                if not mask_z.any():
                    continue
                e_z = e_arr[mask_z]
                add("zones", z, "active_nodes", float(e_z.sum()))
                # living share among active
                il_z = il_arr[mask_z]
                denom = e_z.sum() + 1e-6
                add("zones", z, "living_share_active", float((il_z * e_z).sum() / denom))
                # floors mean on active with floors_num>=0
                if fl_arr is not None and hf_arr is not None:
                    hf_z = hf_arr[mask_z]
                    fl_z = fl_arr[mask_z]
                    mask_f = (hf_z.astype(bool)) & (e_z > 0.5) & (fl_z >= 0)
                    if mask_f.any():
                        add("zones", z, "floors_mean", float(fl_z[mask_f].mean()))
                # living area sum only is_living & active
                la_z = la_arr[mask_z]
                add("zones", z, "living_area_sum", float((la_z * il_z * e_z).sum()))

    # --- shapes distribution (по активным узлам) ---
    if len(nodes) and "s_i" in nodes.columns and "e_i" in nodes.columns:
        e_mask = _to_float_np(nodes["e_i"], default=0.0) > 0.5
        if e_mask.any():
            s_vals = pd.to_numeric(nodes.loc[e_mask, "s_i"], errors="coerce").fillna(-1).astype(int)
            vc = s_vals.value_counts()
            label_map = {0: "Rect", 1: "L", 2: "U", 3: "X", -1: "unknown"}
            for k, v in vc.items():
                add("shapes", label_map.get(int(k), str(int(k))), "count", int(v))

    # --- geometry stats (по активным узлам) ---
    if len(nodes) and "e_i" in nodes.columns:
        e_mask = _to_float_np(nodes["e_i"], default=0.0) > 0.5
        if e_mask.any():
            for col, mname in [
                ("posx","posx"),("posy","posy"),
                ("size_x","size_x"),("size_y","size_y"),
                ("phi_resid","phi"),
            ]:
                if col in nodes.columns:
                    v = pd.to_numeric(nodes.loc[e_mask, col], errors="coerce").dropna().to_numpy()
                    if v.size:
                        add("geom", "all", f"{mname}_mean", float(v.mean()))
                        add("geom", "all", f"{mname}_std",  float(v.std()))
                        add("geom", "all", f"{mname}_min",  float(v.min()))
                        add("geom", "all", f"{mname}_max",  float(v.max()))

    # --- floors stats ---
    if len(nodes) and "has_floors" in nodes.columns and "e_i" in nodes.columns:
        e_mask = _to_float_np(nodes["e_i"], default=0.0) > 0.5
        hf = _to_bool_np(nodes["has_floors"])
        add("floors", "all", "has_floors_share_active", float(hf[e_mask].mean() if e_mask.any() else 0.0))
        if "floors_num" in nodes.columns:
            fl = pd.to_numeric(nodes["floors_num"], errors="coerce").fillna(-1).astype(int)
            mask = (hf.astype(bool)) & e_mask & (fl >= 0)
            if mask.any():
                fl_masked = fl[mask]
                add("floors", "all", "floors_mean", float(fl_masked.mean()))
                # топ-20 этажностей
                top = fl_masked.value_counts().head(20)
                for k, v in top.items():
                    add("floors", f"{int(k)}", "count", int(v))

    # --- services (потоковая обработка батчами) ---
    if not args.no_services:
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa

            pf = pq.ParquetFile(nodes_pq)

            # 1) Небольшая выборка (первая row-group) для определения схемы сервисов:
            sample_tbl = pf.read_row_group(0)
            # Преобразуем в pandas с ArrowDtype, если возможно
            try:
                sample_df = sample_tbl.to_pandas(types_mapper=pd.ArrowDtype)
            except Exception:
                sample_df = sample_tbl.to_pandas()

            schema = infer_service_schema_from_nodes(sample_df) or []
            if schema:
                svc_has_cols = [it.get("has_column") for it in schema if it.get("has_column")]
                svc_cap_cols = [it.get("capacity_column") for it in schema if it.get("capacity_column")]
                svc_has_cols = [c for c in svc_has_cols if c]  # чистим None/пустые
                svc_cap_cols = [c for c in svc_cap_cols if c]

                # читаем только e_i и нужные сервисные колонки
                cols = ["e_i"] + svc_has_cols + svc_cap_cols
                # пересекаем с реально доступными
                cols = [c for c in cols if c in nodes_cols_avail]
                if "e_i" not in cols:
                    cols = ["e_i"] + cols  # вдруг удалили

                present_cnt: Dict[str, int] = defaultdict(int)
                cap_sum: Dict[str, float] = defaultdict(float)

                for batch in pf.iter_batches(columns=cols, batch_size=args.batch_size):
                    try:
                        dfb = batch.to_pandas(types_mapper=pd.ArrowDtype)
                    except Exception:
                        dfb = batch.to_pandas()

                    e_mask_b = pd.to_numeric(dfb["e_i"], errors="coerce").fillna(0.0).to_numpy() > 0.5

                    for it in schema:
                        nm = it.get("name")
                        has_col = it.get("has_column")
                        cap_col = it.get("capacity_column")

                        if not has_col or has_col not in dfb.columns:
                            continue

                        has_vals = dfb[has_col].fillna(False).astype("boolean").to_numpy()
                        mask = has_vals & e_mask_b
                        present_cnt[nm] += int(mask.sum())

                        if cap_col and cap_col in dfb.columns:
                            caps = pd.to_numeric(dfb[cap_col], errors="coerce").fillna(0.0).to_numpy()
                            cap_sum[nm] += float(caps[mask].sum())

                for s, c in present_cnt.items():
                    add("services", s, "present_count", int(c))
                for s, v in cap_sum.items():
                    add("services", s, "capacity_sum", float(v))
            else:
                log.info("Схема сервисов не обнаружена — пропускаю секцию 'services'.")
        except Exception as ex:
            log.warning(f"Секция 'services' пропущена (ошибка потоковой обработки): {ex}")

    # --- save CSV ---
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info(f"Сохранено: {out_csv}  (строк: {len(df)})")

    return 0

if __name__ == "__main__":
    sys.exit(main())
