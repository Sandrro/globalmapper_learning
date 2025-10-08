#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_infer_blocks_clustered.py

Актуализированная утилита для пакетного инференса кварталов.
Инференс и подготовка точек совпадают с исходной версией: вызывается
модель, центроиды при необходимости смещаются ко второму кольцу.
Далее строится регулярная сетка квадратов 15×15 м, обрезанная по
внутренней границе первого кольца, в ячейках подсчитывается статистика
по точкам и выполняется кластеризация по плотности и вытянутости.

Выходные данные:
  --out-buildings : квадраты сетки (Polygon) с агрегированными атрибутами.
  --out-centroids : (опц.) итоговые (слитые и смещённые) точки зданий (Point).

"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

try:  # tqdm опционален
    from tqdm import tqdm
except Exception:  # pragma: no cover - резерв
    tqdm = lambda x, **k: x

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    print("[FATAL] Требуется NumPy: {}".format(exc), file=sys.stderr)
    raise

try:
    from shapely.geometry import Point, Polygon, box, mapping, shape
    from shapely.prepared import prep
except Exception:  # pragma: no cover
    print("[FATAL] Требуется shapely и её зависимости (GEOS).", file=sys.stderr)
    raise

from batch_infer_blocks import (  # noqa: E402
    build_rings,
    get_people,
    infer_slots_from_block_bbox,
    lines_of,
    load_json_maybe,
    load_services_vocab_from_artifacts,
    nearest_on_lines,
    normalize_targets_map,
    pick_service_keys,
    read_geojson,
    write_geojson,
    zone_label_of,
    _aggregate_points,
)


def _generate_square_grid(area: Polygon, cell_size: float) -> List[Polygon]:
    if area is None or area.is_empty or cell_size <= 0:
        return []

    minx, miny, maxx, maxy = area.bounds
    if maxx - minx <= 0 or maxy - miny <= 0:
        return []

    start_x = math.floor(minx / cell_size) * cell_size
    start_y = math.floor(miny / cell_size) * cell_size
    end_x = math.ceil(maxx / cell_size) * cell_size
    end_y = math.ceil(maxy / cell_size) * cell_size

    grid: List[Polygon] = []
    y = start_y
    while y < end_y - 1e-9:
        x = start_x
        while x < end_x - 1e-9:
            cell = box(x, y, x + cell_size, y + cell_size)
            try:
                clipped = cell.intersection(area)
            except Exception:
                clipped = cell
            if clipped.is_empty:
                x += cell_size
                continue
            try:
                clipped = clipped.buffer(0)
            except Exception:
                pass
            if not clipped.is_empty and clipped.area > 1e-4:
                grid.append(clipped)
            x += cell_size
        y += cell_size
    return grid


def _points_in_polygon(points: Sequence[Point], polygon: Polygon) -> List[int]:
    if not points:
        return []
    try:
        prepared = prep(polygon)
    except Exception:
        prepared = None
    indices: List[int] = []
    for idx, point in enumerate(points):
        inside = False
        try:
            if prepared is not None:
                inside = prepared.contains(point) or prepared.intersects(point)
            else:
                inside = polygon.contains(point) or polygon.touches(point)
        except Exception:
            inside = polygon.distance(point) <= 1e-6
        if inside:
            indices.append(idx)
    return indices


def _compute_elongation(coords: np.ndarray) -> Tuple[float, float | None]:
    if coords.shape[0] < 2:
        return 0.0, None
    centered = coords - coords.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return 0.0, None
    cov = np.cov(centered, rowvar=False)
    if cov.shape != (2, 2):
        return 0.0, None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    major = float(max(eigvals[order[0]], 0.0))
    minor = float(max(eigvals[order[-1]], 0.0))
    elongation = 1.0
    if minor <= 1e-9 and major <= 1e-9:
        elongation = 1.0
    else:
        elongation = (major + 1e-9) / (minor + 1e-9)
    direction = eigvecs[:, order[0]]
    angle_rad = math.atan2(direction[1], direction[0])
    angle_deg = math.degrees(angle_rad)
    # ориентация без направления: нормируем к [0, 180)
    if angle_deg < 0:
        angle_deg += 180.0
    angle_deg = angle_deg % 180.0
    return float(elongation), float(angle_deg)


def _cluster_cells(vectors: Sequence[Sequence[float]], max_clusters: int = 4,
                   max_iter: int = 50) -> List[int]:
    data = np.array(vectors, dtype=float)
    n = data.shape[0]
    if n == 0:
        return []
    k = min(max_clusters, n)
    if k <= 1:
        return [0] * n

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-9] = 1.0
    normed = (data - mean) / std

    indices = np.linspace(0, n - 1, k).astype(int)
    centroids = normed[indices].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(normed[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        for ci in range(k):
            if not np.any(new_labels == ci):
                farthest = int(np.argmax(distances.min(axis=1)))
                new_labels[farthest] = ci

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for ci in range(k):
            members = normed[labels == ci]
            if members.size == 0:
                continue
            centroids[ci] = members.mean(axis=0)

    return labels.tolist()


# ---- Основной цикл ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer + regular grid clustering")
    ap.add_argument("--blocks", required=True, help="GeoJSON FeatureCollection кварталов")
    ap.add_argument("--model-ckpt", required=True, help="Путь к чекпойнту модели (.pt)")
    ap.add_argument("--train-script", default="./train.py", help="Путь к train.py")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--zone-attr", default="zone")
    ap.add_argument("--zones-json", default=None)
    ap.add_argument("--services-json", default=None)
    ap.add_argument("--targets-by-zone", default=None)
    ap.add_argument("--la-by-zone", default=None)
    ap.add_argument("--people", type=int, default=1000)
    ap.add_argument("--min-services", dest="min_services", action="store_true", default=True)
    ap.add_argument("--no-min-services", dest="min_services", action="store_false")
    ap.add_argument("--infer-slots", type=int, default=None,
                    help="Число слотов инференса. По умолчанию вычисляется по bbox квартала")
    ap.add_argument("--infer-knn", type=int, default=8)
    ap.add_argument("--infer-e-thr", type=float, default=0.5)
    ap.add_argument("--infer-il-thr", type=float, default=0.5)
    ap.add_argument("--infer-sv1-thr", type=float, default=0.5)
    ap.add_argument("--out-buildings", required=True, help="Выход GeoJSON квадратов")
    ap.add_argument("--out-centroids", default=None, help="Выход GeoJSON центроидов")
    ap.add_argument("--out-epsg", type=int, default=32636)
    ap.add_argument("--grid-size", type=float, default=15.0,
                    help="Размер стороны квадрата сетки в метрах")
    ap.add_argument("--temp-dir", default=None)
    args = ap.parse_args()

    blocks = read_geojson(args.blocks)
    features = blocks.get("features", [])
    targets_map = normalize_targets_map(load_json_maybe(args.targets_by_zone))
    la_by_zone = normalize_targets_map(load_json_maybe(args.la_by_zone))
    services_vocab = load_services_vocab_from_artifacts(args.model_ckpt)
    service_keys = pick_service_keys(services_vocab)

    out_square_features: List[Dict[str, Any]] = []
    out_centroids_features: List[Dict[str, Any]] = []

    temp_root = Path(args.temp_dir or "./_infer_tmp")
    temp_root.mkdir(parents=True, exist_ok=True)

    for bi, feat in enumerate(tqdm(features, desc="blocks")):
        geom = shape(feat.get("geometry"))
        if geom is None or geom.is_empty:
            continue
        zone_label = zone_label_of(feat, args.zone_attr) or "unknown"
        block_idx = bi
        infer_slots = args.infer_slots or infer_slots_from_block_bbox(geom, cell_size_m=100.0)

        people = get_people(feat.get("properties") or {}, args.people)
        services_target = None
        if args.min_services:
            services_target = {
                k: people for k in service_keys.values()
            }
        la_target = None
        floors_avg = None
        targets = targets_map.get(zone_label)
        if targets:
            la_target = targets.get("la")
            floors_avg = targets.get("floors_avg")
        elif zone_label in la_by_zone:
            la_target = la_by_zone[zone_label].get("la")

        in_path = temp_root / f"block_{bi:05d}_in.geojson"
        out_path = temp_root / f"block_{bi:05d}_out.geojson"
        write_geojson(str(in_path), {
            "type": "FeatureCollection",
            "features": [feat],
        })

        cmd = [
            sys.executable,
            args.train_script,
            "--mode", "infer",
            "--model-ckpt", args.model_ckpt,
            "--infer-geojson-in", str(in_path),
            "--infer-out", str(out_path),
            "--infer-knn", str(args.infer_knn),
            "--infer-e-thr", str(args.infer_e_thr),
            "--infer-il-thr", str(args.infer_il_thr),
            "--infer-sv1-thr", str(args.infer_sv1_thr),
            "--infer-slots", str(infer_slots),
            "--zone", str(zone_label),
        ]
        if args.config:
            cmd.extend(["--config", args.config])
        if args.device:
            cmd.extend(["--device", args.device])
        if services_target:
            cmd.extend(["--services-target", json_dumps(services_target)])
        if la_target is not None:
            cmd.extend(["--la-target", str(la_target)])
        if floors_avg is not None:
            cmd.extend(["--floors-avg", str(floors_avg)])

        subprocess.run(cmd, check=True)
        infer_out = read_geojson(str(out_path))

        centroids_raw: List[Point] = []
        cprops_raw: List[Dict[str, Any]] = []
        infer_uid = 0
        for block in infer_out.get("features", []):
            try:
                g = shape(block.get("geometry"))
            except Exception:
                continue
            if g is None or g.is_empty:
                continue
            if g.geom_type == "MultiPolygon":
                geoms = [poly for poly in g.geoms if not poly.is_empty]
                if not geoms:
                    continue
                g = max(geoms, key=lambda poly: poly.area)
            if g.geom_type != "Polygon" or g.area <= 0:
                continue
            raw_props = dict(block.get("properties") or {})
            raw_props.setdefault(args.zone_attr, zone_label)
            raw_props.setdefault("_source_block_index", block_idx)
            infer_uid += 1
            raw_props["infer_fid"] = infer_uid
            centroids_raw.append(g.centroid)
            cprops_raw.append(raw_props)

        if not centroids_raw:
            continue

        merged_points = list(centroids_raw)
        merged_props = list(cprops_raw)

        ring1, ring2, midline = build_rings(geom)
        mid_lines = lines_of(midline)

        moved_points: List[Point] = []
        moved_props: List[Dict[str, Any]] = []
        for point, props in zip(merged_points, merged_props):
            q = point
            status = "none"
            props_copy = dict(props)
            if ring2 and not ring2.is_empty and mid_lines:
                in_r1 = ring1.contains(point) or ring1.touches(point)
                in_r2 = ring2.contains(point) or ring2.touches(point)
                if in_r1 or in_r2:
                    status = "first" if in_r1 and not in_r2 else "second"
                    li, q_on, _ = nearest_on_lines(mid_lines, point)
                    if li >= 0:
                        q = q_on
                    props_copy["ring_snap"] = "to_ring2_midline"
                    props_copy["ring_zone"] = status
            if "ring_snap" not in props_copy:
                props_copy["ring_snap"] = "none"
                props_copy["ring_zone"] = status
            moved_points.append(q)
            moved_props.append(props_copy)

        for point, props in zip(moved_points, moved_props):
            out_centroids_features.append({
                "type": "Feature",
                "geometry": mapping(point),
                "properties": props,
            })

        coords = np.array([[p.x, p.y] for p in moved_points], dtype=float)
        grid_area = None
        try:
            inner_area = geom.buffer(-10.0)
            if inner_area is not None and not inner_area.is_empty:
                grid_area = inner_area
        except Exception:
            grid_area = None
        if grid_area is None:
            grid_area = geom

        grid_cells = _generate_square_grid(grid_area, max(0.1, float(args.grid_size)))
        if not grid_cells:
            continue

        cell_vectors: List[List[float]] = []
        cell_infos: List[Dict[str, Any]] = []

        for cell_idx, cell in enumerate(grid_cells):
            indices = _points_in_polygon(moved_points, cell)
            agg_attrs = _aggregate_points(indices, moved_props) if indices else {}
            point_count = len(indices)
            area_m2 = float(cell.area) if cell.area else 0.0
            density = point_count / area_m2 if area_m2 > 1e-9 else 0.0
            if indices:
                cell_coords = coords[indices]
            else:
                cell_coords = np.empty((0, 2), dtype=float)
            elongation, direction = _compute_elongation(cell_coords) if indices else (0.0, None)
            cell_vectors.append([float(point_count), float(density), float(elongation)])
            cell_infos.append({
                "geometry": cell,
                "indices": indices,
                "agg": agg_attrs,
                "point_count": point_count,
                "point_density": density,
                "elongation": elongation,
                "direction": direction,
                "area": area_m2,
                "grid_idx": cell_idx,
            })

        cluster_ids = _cluster_cells(cell_vectors, max_clusters=4)

        for info, cluster_id in zip(cell_infos, cluster_ids):
            props = {
                "block_index": block_idx,
                "zone": zone_label,
                "grid_index": int(info["grid_idx"]),
                "grid_area": float(info["area"]),
                "point_count": int(info["point_count"]),
                "point_density": float(info["point_density"]),
                "elongation": float(info["elongation"]),
                "cluster_id": int(cluster_id),
            }
            if info["direction"] is not None:
                props["elongation_dir_deg"] = float(info["direction"])
            props.update(info["agg"])
            out_square_features.append({
                "type": "Feature",
                "geometry": mapping(info["geometry"]),
                "properties": props,
            })

    if out_square_features:
        write_geojson(args.out_buildings, {
            "type": "FeatureCollection",
            "features": out_square_features,
        }, epsg=args.out_epsg)

    if args.out_centroids:
        write_geojson(args.out_centroids, {
            "type": "FeatureCollection",
            "features": out_centroids_features,
        }, epsg=args.out_epsg)

def json_dumps(obj: Any) -> str:
    try:
        import json

        return json.dumps(obj)
    except Exception:  # pragma: no cover
        raise


if __name__ == "__main__":  # pragma: no cover
    main()