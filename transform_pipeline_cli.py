#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transform_pipeline_cli.py — оркестратор пайплайна иерархической канонизации

Этапы:
  1) Загрузка кварталов/зданий/сервисов, CRS → метрический, починка геометрий.
  2) Быстрый assign зданий → кварталам (STRtree + prepared.contains + fallback по пересечению).
  3) Интеграция сервисов: агрегирование по зданиям, добавление синтетических зданий и генерация колонок
     has_service__*/service_capacity__* вместе со схемой services.json.
  4) Параллельная per-block обработка (ветви, канон-фичи, маски, parquet на блок).
  5) Сборка по дереву by_block/*/*.parquet → четыре глобальных parquet в out/.
  6) Дампы orphaned_buildings.geojson / orphaned_blocks.geojson и timeout.geojson.

Модули проекта:
  - geo_common.py (I/O, CRS, WKB, маски, масштаб)
  - assign_blocks_cli.py (fast_assign_blocks)
  - services_processing.py (агрегация сервисов и описание схемы)
  - per_block_worker.py (worker_process_block)

Пример запуска:
  python transform_pipeline_cli.py \
    --blocks data/blocks.gpkg \
    --buildings data/buildings.parquet \
    --target-crs EPSG:3857 \
    --out-dir out \
    --N 120 --mask-size 64 --num-workers 8 --block-timeout-sec 300 \
    --services data/services.geojson --out-services-json out/services.json \
    --min-branch-len 5.0 --branch-simplify-tol 0.2 \
    --log-level INFO
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from glob import glob
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from geo_common import (
    load_vector,
    ensure_valid_series,
    normalize_crs_str,
    to_metric_crs,
    geom_to_wkb,
    geom_from_wkb,
)
from assign_blocks_cli import fast_assign_blocks
from services_processing import (
    DEFAULT_CAPACITY_FIELDS,
    DEFAULT_EXCLUDED_SERVICES,
    attach_services_to_buildings,
    write_service_schema,
)
from per_block_worker import worker_process_block

log = logging.getLogger("transform_pipeline")

# ----------------- LOGGER -----------------

def setup_logger(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )

import numpy as np
import pandas as pd


class BlockProgressTracker:
    """Aggregate per-block statistics and log them periodically."""

    def __init__(
        self,
        total_blocks: int,
        log_every: int = 20,
        log_interval_sec: float = 30.0,
    ) -> None:
        self.total_blocks = max(0, int(total_blocks))
        self.log_every = max(1, int(log_every))
        self.log_interval_sec = float(log_interval_sec)
        self.processed = 0
        self.ok = 0
        self.ok_empty = 0
        self.timeouts = 0
        self.other = 0
        self.total_buildings = 0
        self.total_nodes = 0
        self.total_edges = 0
        self.total_elapsed = 0.0
        self.last_log_ts = time.perf_counter()

    def update(self, result: Optional[Dict[str, Any]]) -> None:
        if not result:
            return
        self.processed += 1
        status = result.get("status")
        if status == "ok":
            self.ok += 1
        elif status == "ok_empty":
            self.ok_empty += 1
        elif status == "timeout":
            self.timeouts += 1
        else:
            self.other += 1
        try:
            self.total_buildings += int(result.get("n_buildings") or 0)
        except Exception:
            pass
        try:
            self.total_nodes += int(result.get("n_nodes") or 0)
        except Exception:
            pass
        try:
            self.total_edges += int(result.get("n_edges") or 0)
        except Exception:
            pass
        elapsed = result.get("elapsed")
        if elapsed is not None:
            try:
                self.total_elapsed += float(elapsed)
            except Exception:
                pass
        self._maybe_log()

    def _maybe_log(self, force: bool = False) -> None:
        now = time.perf_counter()
        should_log = force or (self.processed % self.log_every == 0)
        if not should_log and (now - self.last_log_ts) < self.log_interval_sec:
            return
        avg_block_time = (self.total_elapsed / self.processed) if self.processed else 0.0
        log.info(
            "[progress] blocks %s/%s | ok=%s empty=%s timeout=%s other=%s | buildings=%s nodes=%s edges=%s | avg_block_time=%.2fs",
            self.processed,
            self.total_blocks,
            self.ok,
            self.ok_empty,
            self.timeouts,
            self.other,
            self.total_buildings,
            self.total_nodes,
            self.total_edges,
            avg_block_time,
        )
        self.last_log_ts = now

    def log(self, force: bool = False) -> None:
        self._maybe_log(force=force)


def dump_orphans_files(
    bldgs: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    orphan_b = bldgs[bldgs["block_id"].isna()].copy()
    if not orphan_b.empty:
        outb = os.path.join(out_dir, "orphaned_buildings.geojson")
        try:
            orphan_b.to_file(outb, driver="GeoJSON")
            log.info(f"orphaned_buildings: сохранено {len(orphan_b)} → {outb}")
        except Exception as e:
            log.warning(f"orphaned_buildings: не удалось сохранить GeoJSON: {e}")

    used = set(bldgs["block_id"].dropna().unique().tolist())
    orphan_blk = blocks[~blocks["block_id"].isin(used)].copy()
    if not orphan_blk.empty:
        outk = os.path.join(out_dir, "orphaned_blocks.geojson")
        try:
            orphan_blk.to_file(outk, driver="GeoJSON")
            log.info(f"orphaned_blocks: сохранено {len(orphan_blk)} → {outk}")
        except Exception as e:
            log.warning(f"orphaned_blocks: не удалось сохранить GeoJSON: {e}")


def dump_timeouts(
    timeout_packets: List[Dict[str, Any]], out_dir: str, crs, service_schema: Optional[List[Dict[str, str]]] = None
) -> None:
    if not timeout_packets:
        return
    rows: List[Dict[str, Any]] = []
    schema = service_schema or []
    for item in timeout_packets:
        blk_id = item.get("block_id")
        zone = item.get("zone")
        try:
            poly = geom_from_wkb(item.get("poly_wkb"))
        except Exception:
            poly = None
        if poly is not None and not poly.is_empty:
            rows.append({"kind": "block", "block_id": blk_id, "zone": zone, "geometry": poly})
        for r in (item.get("bldgs_rows") or []):
            try:
                g = geom_from_wkb(r.get("geometry"))
            except Exception:
                g = None
            if g is None or g.is_empty:
                continue
            row = {
                "kind": "building",
                "block_id": blk_id,
                "building_id": r.get("building_id"),
                "floors_num": (None if pd.isna(r.get("floors_num")) else int(r.get("floors_num"))),
                "living_area": float(r.get("living_area", 0.0)),
                "is_living": bool(r.get("is_living", False)),
                "has_floors": bool(r.get("has_floors", False)),
                "geometry": g,
            }
            for item_schema in schema:
                has_col = item_schema.get("has_column")
                cap_col = item_schema.get("capacity_column")
                if has_col:
                    row[has_col] = bool(r.get(has_col, False))
                if cap_col:
                    try:
                        row[cap_col] = float(r.get(cap_col, 0.0))
                    except Exception:
                        row[cap_col] = 0.0
            rows.append(row)
    if not rows:
        return
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=crs)
    outp = os.path.join(out_dir, "timeout.geojson")
    try:
        gdf.to_file(outp, driver="GeoJSON")
        log.info(f"timeout.geojson: сохранено {len(gdf)} объектов (блоки+здания) → {outp}")
    except Exception as e:
        log.warning(f"timeout.geojson: не удалось сохранить GeoJSON: {e}")


# ----------------- MERGE per-block parquet -----------------

def merge_parquet_tree(by_block_root: str, rel_name: str, out_path: str) -> int:
    t0 = time.perf_counter()
    paths = sorted(glob(os.path.join(by_block_root, "*", rel_name)))
    if not paths:
        pd.DataFrame([]).to_parquet(out_path, index=False)
        log.warning(f"нет файлов для {rel_name} — создан пустой parquet → {out_path}")
        return 0

    dfs: List[pd.DataFrame] = []
    bad = 0
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:
            bad += 1
            log.warning(f"пропускаю '{p}': {e}")

    if not dfs:
        pd.DataFrame([]).to_parquet(out_path, index=False)
        log.warning(f"все входные файлы для {rel_name} повреждены/недоступны → пустой parquet")
        return 0

    df = pd.concat(dfs, ignore_index=True)
    if rel_name == "nodes_fixed.parquet" and "floors_num" in df.columns:
        try:
            df["floors_num"] = df["floors_num"].astype("Int64")
        except Exception:
            pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    n = len(df)
    t1 = time.perf_counter()
    msg = f"[{rel_name}] merged {n} rows → {out_path} in {t1 - t0:.2f}s"
    if bad:
        msg += f" (skipped {bad} bad files)"
    log.info(msg)
    return n


# ----------------- PIPELINE -----------------

def main() -> int:
    ap = argparse.ArgumentParser("Оркестратор: assign → per-block → merge")
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--target-crs", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument(
    "--zone-column", type=str, default=None,
    help="Имя колонки с лейблом зоны в кварталах (по умолчанию авто-детект, в т.ч. functional_zone_type_name)"
)

    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--mask-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--block-timeout-sec", type=int, default=300)

    ap.add_argument("--services", type=str, default=None, help="GeoJSON с сервисами")
    ap.add_argument(
        "--services-capacity-fields",
        type=str,
        default=",".join(DEFAULT_CAPACITY_FIELDS),
        help="Поля ёмкости в GeoJSON сервисов (через запятую)",
    )
    ap.add_argument(
        "--services-exclude",
        type=str,
        default=",".join(DEFAULT_EXCLUDED_SERVICES),
        help="Названия сервисов, которые нужно игнорировать (через запятую)",
    )
    ap.add_argument("--out-services-json", type=str, default=None, help="Куда сохранить services.json")

    ap.add_argument("--min-branch-len", type=float, default=0.0)
    ap.add_argument("--branch-simplify-tol", type=float, default=0.0)

    ap.add_argument("--knn-k", type=int, default=6,
                    help="K соседей для KNN-обогащения рёбер (в пространстве posx,posy); 0 — выкл.")

    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    setup_logger(args.log_level)

    target_crs = normalize_crs_str(args.target_crs)
    os.makedirs(args.out_dir, exist_ok=True)

    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    by_block_root = os.path.join(args.out_dir, "by_block")
    os.makedirs(by_block_root, exist_ok=True)

    # 1) Загрузка
    log.info("Загрузка данных…")
    blocks = load_vector(args.blocks)
    bldgs = load_vector(args.buildings)
    services = load_vector(args.services) if args.services else None

    # 2) CRS
    log.info(f"Приведение CRS к {target_crs}…")
    blocks = to_metric_crs(blocks, target_crs)
    bldgs = to_metric_crs(bldgs, target_crs)
    if services is not None:
        services = to_metric_crs(services, target_crs)

    # 3) Валидность геометрий
    log.info("Починка геометрий…")
    blocks["geometry"] = ensure_valid_series(blocks.geometry)
    bldgs["geometry"] = ensure_valid_series(bldgs.geometry)
    if services is not None:
        services["geometry"] = ensure_valid_series(services.geometry)

    # Идентификаторы
    blocks = blocks.reset_index(drop=True).copy()
    if "block_id" not in blocks.columns:
        blocks["block_id"] = blocks.index.astype(str)
    else:
        blocks["block_id"] = blocks["block_id"].astype(str)

    bldgs = bldgs.reset_index(drop=True).copy()
    service_schema: List[Dict[str, str]] = []
    if services is not None and not services.empty:
        cap_fields = [c.strip() for c in str(args.services_capacity_fields).split(",") if c.strip()]
        exclude_names = [c.strip() for c in str(args.services_exclude).split(",") if c.strip()]
        bldgs, service_schema = attach_services_to_buildings(
            bldgs,
            services.reset_index(drop=True).copy(),
            capacity_fields=cap_fields,
            exclude_names=exclude_names,
        )
    bldgs = bldgs.reset_index(drop=True).copy()
    bldgs["building_id"] = bldgs.index.astype(str)

    zone_col = args.zone_column
    if zone_col is None:
        candidates = [
            "functional_zone_type_name",  # ваш кейс
            "zone", "func_zone", "functional_zone", "zone_name", "zoning", "class", "type"
        ]
        zone_col = next((c for c in candidates if c in blocks.columns), None)

    if zone_col and zone_col in blocks.columns:
        blocks["zone"] = blocks[zone_col].astype(str)
    else:
        if "zone" not in blocks.columns:
            log.warning(f"Нет зон")
            blocks["zone"] = pd.NA

    # 4) Assign
    log.info("Быстрое присвоение зданий кварталам (STRtree)…")
    t0 = time.perf_counter()
    bldgs["block_id"] = fast_assign_blocks(
        bldgs,
        blocks,
        log_prefix="assign: ",
        num_workers=max(1, int(args.num_workers)),
        chunk_size=20000,
    )
    n_unassigned = int(bldgs["block_id"].isna().sum())
    if n_unassigned:
        log.warning(f"Неприсвоенных зданий: {n_unassigned} (будут пропущены)")
        dump_orphans_files(bldgs, blocks, args.out_dir)
        bldgs = bldgs[~bldgs["block_id"].isna()].copy()
    t1 = time.perf_counter()
    log.info(f"assign total time: {t1 - t0:.2f}s")

    # 5) Нормализация атрибутов зданий
    if "floors_num" not in bldgs.columns and "storeys_count" in bldgs.columns:
        bldgs["floors_num"] = bldgs["storeys_count"]

    for k, default in {
        "floors_num": pd.NA,
        "living_area": 0.0,
        "is_living": False,
        "has_floors": False,
    }.items():
        if k not in bldgs.columns:
            bldgs[k] = default

    floors_numeric = pd.to_numeric(bldgs["floors_num"], errors="coerce")
    storeys_numeric = None
    if "storeys_count" in bldgs.columns:
        storeys_numeric = pd.to_numeric(bldgs["storeys_count"], errors="coerce")

    if storeys_numeric is not None:
        has_floors_mask = storeys_numeric.notna() & (storeys_numeric > 0)
    else:
        has_floors_mask = floors_numeric.notna() & (floors_numeric > 0)

    bldgs["floors_num"] = floors_numeric.astype("Int64")
    bldgs["living_area"] = pd.to_numeric(bldgs["living_area"], errors="coerce").fillna(0.0).astype(float)
    bldgs["is_living"] = bldgs["is_living"].astype(bool)
    bldgs["has_floors"] = has_floors_mask.fillna(False).astype(bool)

    if service_schema:
        out_json_path = args.out_services_json or os.path.join(args.out_dir, "services.json")
        write_service_schema(service_schema, out_json_path)
        log.info(f"[ok] services.json → {out_json_path} (|types|={len(service_schema)})")

    # 6) Формирование задач per-block
    log.info("Формирование заданий по кварталам…")
    block_records = blocks[["block_id", "zone", "geometry"]].copy()
    block_records["poly_wkb"] = block_records.geometry.apply(geom_to_wkb)
    block_records = block_records.drop(columns=["geometry"])

    groups = bldgs.groupby("block_id")
    tasks: List[Dict[str, Any]] = []
    n_bldgs_with_services = 0

    for _, row in block_records.iterrows():
        blk_id = str(row["block_id"])
        sub = groups.get_group(blk_id) if blk_id in groups.groups else pd.DataFrame(columns=bldgs.columns)

        rows_bldgs: List[Dict[str, Any]] = []
        if not sub.empty:
            for r in sub.itertuples(index=False):
                row_dict: Dict[str, Any] = {
                    "building_id": r.building_id,
                    "geometry": geom_to_wkb(getattr(r, "geometry")),
                    "floors_num": getattr(r, "floors_num", pd.NA),
                    "living_area": getattr(r, "living_area", 0.0),
                    "is_living": bool(getattr(r, "is_living", False)),
                    "has_floors": bool(getattr(r, "has_floors", False)),
                }
                has_service = False
                for item_schema in service_schema:
                    has_col = item_schema.get("has_column")
                    cap_col = item_schema.get("capacity_column")
                    if has_col:
                        has_val = bool(getattr(r, has_col, False))
                        row_dict[has_col] = has_val
                    else:
                        has_val = False
                    if cap_col:
                        try:
                            cap_val = float(getattr(r, cap_col, 0.0)) if has_val else 0.0
                        except Exception:
                            cap_val = 0.0
                        row_dict[cap_col] = cap_val
                    if has_val:
                        has_service = True
                if has_service:
                    n_bldgs_with_services += 1
                rows_bldgs.append(row_dict)
        tasks.append(
            {
                "block_id": blk_id,
                "zone": row["zone"],
                "poly_wkb": row["poly_wkb"],
                "bldgs_rows": rows_bldgs,
                "N": int(args.N),
                "mask_size": int(args.mask_size),
                "mask_dir": mask_dir,
                "timeout_sec": int(args.block_timeout_sec),
                "out_dir": args.out_dir,
                "service_schema": service_schema,
                "min_branch_len": float(args.min_branch_len),
                "branch_simplify_tol": float(args.branch_simplify_tol) if args.branch_simplify_tol else None,
                "knn_k": int(args.knn_k),
            }
        )

    log.info(
        f"[services] типов={len(service_schema)}; зданий с непустыми сервисами: {n_bldgs_with_services:,}"
    )

    # 7) Параллельная обработка блоков
    num_workers = max(1, int(args.num_workers))
    log.info(f"Параллельная обработка блоков: workers={num_workers}, blocks={len(tasks)}")
    log_every = max(1, min(50, (len(tasks) // 10) or 1))
    tracker = BlockProgressTracker(total_blocks=len(tasks), log_every=log_every)
    results: List[Dict[str, Any]] = []
    if num_workers <= 1:
        for task in tasks:
            res = worker_process_block(task)
            results.append(res)
            tracker.update(res)
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            for res in pool.imap_unordered(worker_process_block, tasks):
                results.append(res)
                tracker.update(res)
    tracker.log(force=True)

    timeouts = [r for r in results if r and r.get("status") == "timeout"]
    dump_timeouts(timeouts, args.out_dir, crs=blocks.crs, service_schema=service_schema)

    # 8) Сборка parquet
    log.info("Сборка результатов и запись Parquet…")
    out_root = args.out_dir

    total_blocks = merge_parquet_tree(by_block_root, "blocks.parquet", os.path.join(out_root, "blocks.parquet"))
    total_branches = merge_parquet_tree(by_block_root, "branches.parquet", os.path.join(out_root, "branches.parquet"))
    total_nodes = merge_parquet_tree(by_block_root, "nodes_fixed.parquet", os.path.join(out_root, "nodes_fixed.parquet"))
    total_edges = merge_parquet_tree(by_block_root, "edges.parquet", os.path.join(out_root, "edges.parquet"))

    log.info(f"[ok] saved {os.path.join(out_root, 'blocks.parquet')}     rows={total_blocks}")
    log.info(f"[ok] saved {os.path.join(out_root, 'branches.parquet')}   rows={total_branches}")
    log.info(f"[ok] saved {os.path.join(out_root, 'nodes_fixed.parquet')} rows={total_nodes}")
    log.info(f"[ok] saved {os.path.join(out_root, 'edges.parquet')}      rows={total_edges}")

    # Доп. orphaned_blocks в корне (на случай изменений после assign)
    try:
        used_blocks = set(bldgs["block_id"].dropna().unique().tolist())
        orphan_blk = blocks[~blocks["block_id"].isin(used_blocks)].copy()
        if not orphan_blk.empty:
            outk = os.path.join(out_root, "orphaned_blocks.geojson")
            try:
                orphan_blk.to_file(outk, driver="GeoJSON")
                log.info(f"orphaned_blocks: сохранено {len(orphan_blk)} → {outk}")
            except Exception as e:
                log.warning(f"orphaned_blocks: не удалось сохранить GeoJSON: {e}")
    except Exception:
        pass

    # Итог
    if timeouts:
        unique_timeout_blocks = sorted({t.get("block_id") for t in timeouts if t})
        log.warning(f"Timeout on {len(unique_timeout_blocks)} blocks → see timeout.geojson")

    log.info("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
