#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_infer_blocks_clustered.py

Актуализированная утилита для пакетного инференса кварталов.
Этап подготовки точек совпадает с исходной версией: вызывается модель,
центроиды объединяются и при необходимости смещаются ко второму кольцу.
Далее квартал делится на сегменты с помощью quad-tree, и в итоговых
квадратах вычисляется агрегированная статистика по точкам.

Выходные данные:
  --out-buildings : квадраты quad-tree (Polygon) c агрегированными атрибутами.
  --out-centroids : (опц.) итоговые (слитые и смещённые) точки зданий (Point).

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
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
    merge_centroids,
    nearest_on_lines,
    normalize_targets_map,
    pick_service_keys,
    read_geojson,
    write_geojson,
    zone_label_of,
    _aggregate_points,
)


@dataclass
class QuadTreeParams:
    max_depth: int
    min_points: int
    min_size: float


@dataclass
class QuadTreeLeaf:
    polygon: Polygon
    indices: List[int]
    path: str
    depth: int
    bounds: Tuple[float, float, float, float]


class QuadTreeSegmenter:
    """Делит квартал на квадраты quad-tree по точкам."""

    def __init__(self, params: QuadTreeParams):
        self.params = params

    def segment(self, coords: np.ndarray, block_geom: Polygon) -> List[QuadTreeLeaf]:
        if len(coords) == 0:
            return []

        minx, miny, maxx, maxy = block_geom.bounds
        size = max(maxx - minx, maxy - miny)
        if size <= 0:
            size = max(1.0, self.params.min_size)
        cx = (minx + maxx) / 2.0
        cy = (miny + maxy) / 2.0
        half = size / 2.0
        root_bounds = (cx - half, cy - half, cx + half, cy + half)

        leaves: List[QuadTreeLeaf] = []

        def subdivide(bounds: Tuple[float, float, float, float], depth: int,
                      indices: Sequence[int], path: str) -> None:
            if not indices:
                return
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            stop = (
                depth >= self.params.max_depth
                or len(indices) <= self.params.min_points
                or width <= self.params.min_size
                or height <= self.params.min_size
            )
            polygon = box(*bounds).intersection(block_geom)
            if polygon.is_empty:
                return
            polygon = polygon.buffer(0)
            if stop:
                leaves.append(
                    QuadTreeLeaf(
                        polygon=polygon,
                        indices=list(indices),
                        path=path,
                        depth=depth,
                        bounds=bounds,
                    )
                )
                return

            midx = (bounds[0] + bounds[2]) / 2.0
            midy = (bounds[1] + bounds[3]) / 2.0
            children = [
                (bounds[0], midy, midx, bounds[3]),  # северо-запад
                (midx, midy, bounds[2], bounds[3]),  # северо-восток
                (midx, bounds[1], bounds[2], midy),  # юго-восток
                (bounds[0], bounds[1], midx, midy),  # юго-запад
            ]
            child_indices: List[List[int]] = [[] for _ in range(4)]
            eps = 1e-9
            for idx in indices:
                x, y = coords[idx]
                assigned = False
                for ci, child in enumerate(children):
                    if (
                        x >= child[0] - eps
                        and x <= child[2] + eps
                        and y >= child[1] - eps
                        and y <= child[3] + eps
                    ):
                        child_indices[ci].append(idx)
                        assigned = True
                        break
                if not assigned:
                    # на случай численных артефактов
                    closest = min(
                        range(4),
                        key=lambda ci: _distance_to_bounds(x, y, children[ci]),
                    )
                    child_indices[closest].append(idx)

            for ci, child_bounds in enumerate(children):
                child_path = f"{path}{ci}"
                subdivide(child_bounds, depth + 1, child_indices[ci], child_path)

        subdivide(root_bounds, 0, list(range(len(coords))), "0")
        return leaves


def _distance_to_bounds(x: float, y: float,
                        bounds: Tuple[float, float, float, float]) -> float:
    minx, miny, maxx, maxy = bounds
    dx = max(minx - x, 0.0, x - maxx)
    dy = max(miny - y, 0.0, y - maxy)
    return dx * dx + dy * dy


def aggregate_segment_attributes(indices: List[int], props: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = _aggregate_points(indices, props)
    agg["point_count"] = len(indices)
    return agg


# ---- Основной цикл ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer + quad-tree segmentation")
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
    ap.add_argument("--merge-centroids-radius-m", type=float, default=10.0)
    ap.add_argument("--prob-field", default="e")
    ap.add_argument("--out-buildings", required=True, help="Выход GeoJSON квадратов")
    ap.add_argument("--out-centroids", default=None, help="Выход GeoJSON центроидов")
    ap.add_argument("--out-epsg", type=int, default=32636)
    ap.add_argument("--quad-max-depth", type=int, default=6, help="Максимальная глубина quad-tree")
    ap.add_argument("--quad-min-points", type=int, default=10,
                    help="Минимальное число точек в квадрате перед остановкой деления")
    ap.add_argument("--quad-min-size", type=float, default=30.0,
                    help="Минимальный размер стороны квадрата в метрах")
    ap.add_argument("--temp-dir", default=None)
    args = ap.parse_args()

    qt_params = QuadTreeParams(
        max_depth=max(0, args.quad_max_depth),
        min_points=max(1, args.quad_min_points),
        min_size=max(0.1, float(args.quad_min_size)),
    )
    segmenter = QuadTreeSegmenter(qt_params)

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

        merged_points, merged_props = merge_centroids(
            points=centroids_raw,
            props_list=cprops_raw,
            radius_m=float(args.merge_centroids_radius_m),
            prob_field=str(args.prob_field),
        )

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
        leaves = segmenter.segment(coords, geom)
        if not leaves:
            continue

        for leaf in leaves:
            attrs = aggregate_segment_attributes(leaf.indices, moved_props)
            props = {
                "block_index": block_idx,
                "zone": zone_label,
                "quad_path": leaf.path,
                "quad_depth": leaf.depth,
                "quad_bounds": list(leaf.bounds),
                "quad_area": float(leaf.polygon.area),
            }
            props.update(attrs)
            out_square_features.append({
                "type": "Feature",
                "geometry": mapping(leaf.polygon),
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
