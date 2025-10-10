#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assign_blocks_cli.py — быстрое присвоение зданий кварталам (STRtree)

Функция:
  1) Загружает кварталы и здания (любой поддерживаемый GeoPandas формат или Parquet).
  2) Приводит к заданному метрическому CRS, чиниит геометрии.
  3) Назначает каждому зданию block_id: сначала centroid∈block (prepared.contains),
     затем fallback по максимальной площади пересечения.
  4) (Опционально) сохраняет «сирот» (здания без квартала) и кварталы без зданий.
  5) Сохраняет обновлённый слой зданий с колонкой block_id.

Зависимости: geopandas, shapely (1.x/2.x), numpy, pandas
Внутренние зависимости: geo_common.py (из текущего проекта).

Пример:
  python assign_blocks_cli.py \
    --blocks data/blocks.gpkg \
    --buildings data/buildings.parquet \
    --target-crs EPSG:3857 \
    --out-buildings out/buildings_assigned.parquet \
    --out-orphan-buildings out/orphaned_buildings.geojson \
    --out-orphan-blocks out/orphaned_blocks.geojson \
    --num-workers 8 --chunk-size 20000 --log-level INFO
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkb as _shp_wkb

from geo_common import (
    load_vector,
    ensure_valid_series,
    normalize_crs_str,
    to_metric_crs,
    geom_to_wkb,
)

log = logging.getLogger("assign_blocks")

# ----------------- LOGGER -----------------

def setup_logger(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ----------------- ПАРАЛЛЕЛЬНАЯ НАБОРНАЯ -----------------
# Используем общий шаблон: инициализация статического состояния в воркере +
# обработка чанков зданий. Это экономит память и ускоряет запросы в STRtree.

from shapely.prepared import prep as _prep

_ASSIGN_STATE: Dict[str, Any] = {
    "blk_ids": None,
    "blk_geoms": None,
    "blk_sindex": None,
    "blk_prepared": None,
}


def _init_assign_worker(blk_wkbs: List[bytes], blk_ids: List[str]) -> None:
    geoms = [_shp_wkb.loads(b) for b in blk_wkbs]
    gs = gpd.GeoSeries(geoms, crs=None)
    _ASSIGN_STATE["blk_ids"] = np.asarray(blk_ids)
    _ASSIGN_STATE["blk_geoms"] = geoms
    _ASSIGN_STATE["blk_sindex"] = gs.sindex
    _ASSIGN_STATE["blk_prepared"] = [_prep(g) for g in geoms]


def _assign_chunk(args: Tuple[np.ndarray, List[bytes]]) -> Tuple[np.ndarray, np.ndarray]:
    idx_global, wkbs = args
    geoms_b = [_shp_wkb.loads(b) for b in wkbs]
    centroids = [g.centroid if g and (not g.is_empty) else None for g in geoms_b]
    sindex = _ASSIGN_STATE["blk_sindex"]
    blk_geoms = _ASSIGN_STATE["blk_geoms"]
    blk_prepared = _ASSIGN_STATE["blk_prepared"]

    assigned = np.full(len(idx_global), -1, dtype=np.int64)

    # STEP1: centroid ∈ block
    for i, pt in enumerate(centroids):
        if pt is None:
            continue
        cand = list(sindex.intersection(pt.bounds))
        if not cand:
            continue
        hit = -1
        for ci in cand:
            if blk_prepared[ci].contains(pt):
                hit = ci
                break
        if hit >= 0:
            assigned[i] = hit

    # STEP2: fallback — intersects + max area
    for i, gi in enumerate(idx_global):
        if assigned[i] >= 0:
            continue
        g = geoms_b[i]
        if g is None or g.is_empty:
            continue
        cand = list(sindex.intersection(g.bounds))
        if not cand:
            continue
        best_a, best_ci = 0.0, -1
        for ci in cand:
            inter = g.intersection(blk_geoms[ci])
            a = inter.area if (inter and not inter.is_empty) else 0.0
            if a > best_a:
                best_a, best_ci = a, ci
        if best_ci >= 0:
            assigned[i] = best_ci

    return idx_global, assigned


# ----------------- ВЕРСИЯ БЕЗ ПАРАЛЛЕЛИЗМА -----------------

def fast_assign_blocks(
    bldgs: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    log_prefix: str = "",
    num_workers: int = 1,
    chunk_size: int = 20000,
) -> pd.Series:
    """Назначить block_id каждому зданию, используя STRtree и prepared.contains.

    Возвращает pd.Series dtype=object длиной len(bldgs) — присвоенный block_id либо <NA>.
    """
    t0 = time.perf_counter()

    valid = blocks.geometry.notna() & (~blocks.geometry.is_empty)
    blocks_v = blocks.loc[valid].copy()
    if blocks_v.empty:
        log.error(f"{log_prefix}нет валидных геометрий blocks")
        return pd.Series(pd.NA, index=bldgs.index, dtype="object")

    blk_ids_v = blocks_v["block_id"].astype(str).to_numpy()
    blk_wkbs = [geom_to_wkb(g) for g in blocks_v.geometry.values]

    n = len(bldgs)
    all_idx = np.arange(n, dtype=np.int64)
    b_wkbs = [geom_to_wkb(g) for g in bldgs.geometry.values]

    if num_workers <= 1:
        _init_assign_worker(blk_wkbs, list(blk_ids_v))
        assigned_ix = np.full(n, -1, dtype=np.int64)
        rng = range(0, n, chunk_size)
        for start in rng:
            end = min(start + chunk_size, n)
            ig, ax = _assign_chunk((all_idx[start:end], b_wkbs[start:end]))
            assigned_ix[ig] = ax
    else:
        import multiprocessing as mp

        assigned_ix = np.full(n, -1, dtype=np.int64)
        tasks: List[Tuple[np.ndarray, List[bytes]]] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            tasks.append((all_idx[start:end], b_wkbs[start:end]))
        with mp.get_context("spawn").Pool(
            processes=max(1, int(num_workers)),
            initializer=_init_assign_worker,
            initargs=(blk_wkbs, list(blk_ids_v)),
        ) as pool:
            for ig, ax in pool.imap_unordered(_assign_chunk, tasks, chunksize=1):
                assigned_ix[ig] = ax

    n_assigned = int((assigned_ix >= 0).sum())
    t1 = time.perf_counter()
    log.info(f"{log_prefix}assigned total: {n_assigned}/{n} in {t1 - t0:.2f}s")

    out = pd.Series(pd.NA, index=bldgs.index, dtype="object")
    ok = assigned_ix >= 0
    out.iloc[np.where(ok)[0]] = blk_ids_v[assigned_ix[ok]]
    return out


# ----------------- ORPHANS DUMPS -----------------

def dump_orphans(
    bldgs: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    out_bldgs_geojson: Optional[str] = None,
    out_blocks_geojson: Optional[str] = None,
) -> None:
    """Сохранить сирот: здания без block_id и кварталы без зданий."""
    if out_bldgs_geojson:
        orphan_b = bldgs[bldgs["block_id"].isna()].copy()
        if not orphan_b.empty:
            os.makedirs(os.path.dirname(out_bldgs_geojson), exist_ok=True)
            try:
                orphan_b.to_file(out_bldgs_geojson, driver="GeoJSON")
                log.info(f"orphaned_buildings: сохранено {len(orphan_b)} → {out_bldgs_geojson}")
            except Exception as e:
                log.warning(f"orphaned_buildings: не удалось сохранить GeoJSON: {e}")

    if out_blocks_geojson:
        used = set(bldgs["block_id"].dropna().unique().tolist())
        orphan_blk = blocks[~blocks["block_id"].isin(used)].copy()
        if not orphan_blk.empty:
            os.makedirs(os.path.dirname(out_blocks_geojson), exist_ok=True)
            try:
                orphan_blk.to_file(out_blocks_geojson, driver="GeoJSON")
                log.info(f"orphaned_blocks: сохранено {len(orphan_blk)} → {out_blocks_geojson}")
            except Exception as e:
                log.warning(f"orphaned_blocks: не удалось сохранить GeoJSON: {e}")


# ----------------- CLI -----------------

def main() -> int:
    ap = argparse.ArgumentParser("Быстрое присвоение зданий кварталам (STRtree)")
    ap.add_argument("--blocks", required=True, help="Путь к кварталам (vector/Parquet)")
    ap.add_argument("--buildings", required=True, help="Путь к зданиям (vector/Parquet)")
    ap.add_argument("--target-crs", required=True, help="Целевой метрический CRS, напр. EPSG:3857")
    ap.add_argument("--out-buildings", required=True, help="Куда сохранить здания с block_id (Parquet/GeoPackage)")
    ap.add_argument("--out-orphan-buildings", default=None, help="GeoJSON сирот-зданий")
    ap.add_argument("--out-orphan-blocks", default=None, help="GeoJSON кварталов без зданий")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--keep-unassigned", action="store_true", help="Не отбрасывать здания без block_id")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logger(args.log_level)

    target_crs = normalize_crs_str(args.target_crs)
    os.makedirs(os.path.dirname(args.out_buildings), exist_ok=True)

    log.info("Загрузка данных…")
    blocks = load_vector(args.blocks)
    bldgs = load_vector(args.buildings)

    log.info(f"Приведение CRS к {target_crs}…")
    blocks = to_metric_crs(blocks, target_crs)
    bldgs = to_metric_crs(bldgs, target_crs)

    log.info("Починка геометрий…")
    blocks["geometry"] = ensure_valid_series(blocks.geometry)
    bldgs["geometry"] = ensure_valid_series(bldgs.geometry)

    # Обязательные идентификаторы
    if "block_id" not in blocks.columns:
        blocks = blocks.reset_index(drop=True).copy()
        blocks["block_id"] = blocks.index.astype(str)
    else:
        blocks["block_id"] = blocks["block_id"].astype(str)

    if "building_id" not in bldgs.columns:
        bldgs = bldgs.reset_index(drop=True).copy()
        bldgs["building_id"] = bldgs.index.astype(str)

    log.info("Назначение зданий кварталам (STRtree)…")
    bldgs = bldgs.copy()
    bldgs["block_id"] = fast_assign_blocks(
        bldgs, blocks,
        log_prefix="assign: ",
        num_workers=max(1, int(args.num_workers)),
        chunk_size=max(1000, int(args.chunk_size)),
    )
    n_unassigned = int(bldgs["block_id"].isna().sum())
    if n_unassigned:
        log.warning(f"Неприсвоенных зданий: {n_unassigned}")

    # Сироты (опционально) + фильтрация
    dump_orphans(
        bldgs, blocks,
        out_bldgs_geojson=args.out_orphan_buildings,
        out_blocks_geojson=args.out_orphan_blocks,
    )

    if not args.keep_unassigned:
        bldgs = bldgs[~bldgs["block_id"].isna()].copy()

    # Сохранение
    out = args.out_buildings
    if out.lower().endswith(".parquet"):
        bldgs.to_parquet(out, index=False)
    else:
        # гео-формат (например, .gpkg)
        bldgs.to_file(out)
    log.info(f"[ok] saved assigned buildings → {out} (rows={len(bldgs)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
