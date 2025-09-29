#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_infer_blocks_clustered.py
Альтернативная версия batch_infer_blocks.py, которая повторяет весь пайплайн
инференса, объединения и возможного смещения центроидов зданий внутри квартала,
но вместо построения гекс-сетки и графа соседства формирует кластеры точек
и строит полигоны зданий по эвристикам.

Основные этапы:
1. Вызов train.py --mode infer по каждому кварталу (аналог batch_infer_blocks).
2. Слияние центроидов в радиусе и «прилипание» к срединной линии второго кольца.
3. Пространственная кластеризация точек (hdbscan, при недоступности — DBSCAN).
4. Построение геометрии зданий для каждого кластера с учётом параметров из YAML.
5. Агрегация атрибутов точек (e_max, is_living_max, living_area_sum, floors_avg_round).

Требования:
- Shapely < 2.0 (проверено на 1.7/1.8).
- PyYAML для чтения параметров.
- Опционально hdbscan, иначе sklearn.cluster.DBSCAN.

Выходы:
  --out-buildings        : полигоны зданий (Polygon) с агрегированными атрибутами.
  --out-clusters-centers : (опц.) точки-кластеры (Point) с параметрами кластеров.
  --out-centroids        : (опц.) итоговые (слитые и смещённые) точки зданий (Point).

Параметры геометрии и кластеризации читаются из YAML-файла (см. building_shape_params.yaml).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # tqdm опционален
    from tqdm import tqdm
except Exception:  # pragma: no cover - резерв
    tqdm = lambda x, **k: x

try:
    import yaml
except Exception as exc:  # pragma: no cover - критично
    print("[FATAL] Требуется PyYAML для загрузки параметров: {}".format(exc), file=sys.stderr)
    raise

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    print("[FATAL] Требуется NumPy: {}".format(exc), file=sys.stderr)
    raise

try:
    from shapely import affinity
    from shapely.geometry import (LineString, MultiPoint, Point, Polygon, mapping,
                                  shape)
except Exception:  # pragma: no cover
    print("[FATAL] Требуется shapely и её зависимости (GEOS).", file=sys.stderr)
    raise

# ---- Переиспользование функций исходного пайплайна ----

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
class ClusterParams:
    algorithm: str
    min_cluster_size: int
    min_samples: int
    dbscan_eps: float
    dbscan_min_samples: int


@dataclass
class GeometryParams:
    round_ratio_max: float
    elongated_ratio_min: float
    narrow_width_max: float
    wall_thickness: float
    min_courtyard_area: float
    center_distance_min: float
    corner_distance_max: float
    circle_segments: int
    bend_angle_min: float
    bend_angle_max: float
    h_bar_gap: float
    h_bar_length_factor: float


def load_params(path: str) -> Tuple[ClusterParams, GeometryParams]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data = data or {}
    cl = data.get("clustering") or {}
    geom = data.get("geometry") or {}
    cparams = ClusterParams(
        algorithm=str(cl.get("algorithm", "auto")),
        min_cluster_size=int(cl.get("min_cluster_size", 8)),
        min_samples=int(cl.get("min_samples", 5)),
        dbscan_eps=float(((cl.get("dbscan") or {}).get("eps", 35.0))),
        dbscan_min_samples=int(((cl.get("dbscan") or {}).get("min_samples", 5))),
    )
    gparams = GeometryParams(
        round_ratio_max=float(geom.get("round_ratio_max", 1.5)),
        elongated_ratio_min=float(geom.get("elongated_ratio_min", 2.0)),
        narrow_width_max=float(geom.get("narrow_width_max", 70.0)),
        wall_thickness=float(geom.get("wall_thickness", 15.0)),
        min_courtyard_area=float(geom.get("min_courtyard_area", 100.0)),
        center_distance_min=float(geom.get("center_distance_min", 45.0)),
        corner_distance_max=float(geom.get("corner_distance_max", 35.0)),
        circle_segments=int(geom.get("circle_segments", 32)),
        bend_angle_min=float(geom.get("bend_angle_min", 90.0)),
        bend_angle_max=float(geom.get("bend_angle_max", 170.0)),
        h_bar_gap=float(geom.get("h_bar_gap", 20.0)),
        h_bar_length_factor=float(geom.get("h_bar_length_factor", 0.8)),
    )
    return cparams, gparams


# ---- Кластеризация ----

class Clusterer:
    def __init__(self, params: ClusterParams):
        self.params = params
        self._algo = None
        self._init_algorithm()

    def _init_algorithm(self) -> None:
        algo = self.params.algorithm.lower()
        if algo not in {"auto", "hdbscan", "dbscan"}:
            algo = "auto"
        self._algo = algo
        if algo in ("auto", "hdbscan"):
            try:
                import hdbscan  # type: ignore

                self._hdbscan_cls = hdbscan.HDBSCAN
                self._algo = "hdbscan"
                return
            except Exception:
                if algo == "hdbscan":
                    print("[WARN] hdbscan не доступен, переключаемся на DBSCAN.", file=sys.stderr)
        try:
            from sklearn.cluster import DBSCAN  # type: ignore

            self._dbscan_cls = DBSCAN
            self._algo = "dbscan"
        except Exception as exc:
            raise RuntimeError("Не удалось инициализировать алгоритм кластеризации: {}".format(exc))

    def fit_predict(self, coords: np.ndarray) -> np.ndarray:
        if len(coords) == 0:
            return np.empty((0,), dtype=int)
        if getattr(self, "_algo", None) == "hdbscan":
            clusterer = self._hdbscan_cls(
                min_cluster_size=self.params.min_cluster_size,
                min_samples=self.params.min_samples,
                allow_single_cluster=False,
            )
            labels = clusterer.fit_predict(coords)
        else:
            clusterer = self._dbscan_cls(
                eps=self.params.dbscan_eps,
                min_samples=self.params.dbscan_min_samples,
            )
            labels = clusterer.fit_predict(coords)
        return labels


# ---- Геометрические утилиты ----

def _rectangle_dimensions(rect: Polygon) -> Tuple[float, float, float]:
    coords = list(rect.exterior.coords)
    if len(coords) < 4:
        return 0.0, 0.0, 0.0
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    angle = math.atan2(dy, dx)
    lengths = []
    for i in range(4):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % 4]
        lengths.append(math.hypot(x2 - x1, y2 - y1))
    if not lengths:
        return 0.0, 0.0, angle
    lengths.sort(reverse=True)
    long_side, short_side = lengths[0], lengths[2]
    return long_side, short_side, angle


def _cluster_roundness(long_side: float, short_side: float) -> float:
    if short_side <= 0:
        return float("inf")
    return long_side / short_side


def _is_central(centroid: Point, block: Polygon, params: GeometryParams) -> bool:
    dist = block.boundary.distance(centroid)
    return dist >= params.center_distance_min


def _is_corner(centroid: Point, block: Polygon, params: GeometryParams) -> bool:
    verts = list(block.exterior.coords)[:-1]
    if not verts:
        return False
    dist = min(Point(v).distance(centroid) for v in verts)
    return dist <= params.corner_distance_max


def _convex_hull_angles(hull: Polygon) -> List[float]:
    angles: List[float] = []
    coords = list(hull.exterior.coords)[:-1]
    if len(coords) < 3:
        return angles
    for i in range(len(coords)):
        prev_pt = coords[(i - 1) % len(coords)]
        cur_pt = coords[i]
        next_pt = coords[(i + 1) % len(coords)]
        v1 = (prev_pt[0] - cur_pt[0], prev_pt[1] - cur_pt[1])
        v2 = (next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1])
        ang = _angle_between(v1, v2)
        if ang is not None:
            angles.append(ang)
    return angles


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> Optional[float]:
    x1, y1 = v1
    x2, y2 = v2
    norm1 = math.hypot(x1, y1)
    norm2 = math.hypot(x2, y2)
    if norm1 <= 1e-9 or norm2 <= 1e-9:
        return None
    dot = x1 * x2 + y1 * y2
    cos_val = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_val))


def _square_polygon(size: float, angle: float, center: Point,
                    params: GeometryParams, add_courtyard: bool) -> Polygon:
    half = size / 2.0
    square_coords = [
        (-half, -half),
        (half, -half),
        (half, half),
        (-half, half),
        (-half, -half),
    ]
    square = Polygon(square_coords)
    if add_courtyard:
        inner_side = size - 2.0 * params.wall_thickness
        if inner_side > 0 and inner_side * inner_side >= params.min_courtyard_area:
            hole_half = inner_side / 2.0
            hole = [
                (-hole_half, -hole_half),
                (hole_half, -hole_half),
                (hole_half, hole_half),
                (-hole_half, hole_half),
                (-hole_half, -hole_half),
            ]
            square = Polygon(square_coords, holes=[hole])
    rotated = affinity.rotate(square, math.degrees(angle), origin=(0, 0))
    moved = affinity.translate(rotated, xoff=center.x, yoff=center.y)
    return moved.buffer(0)


def _h_shape_polygon(long_side: float, short_side: float, angle: float, center: Point,
                     params: GeometryParams) -> Polygon:
    width = max(short_side, params.wall_thickness * 3)
    height = max(long_side, width)
    leg_width = params.wall_thickness
    gap = params.h_bar_gap
    leg_length = height * params.h_bar_length_factor
    top = leg_length / 2.0
    base = -leg_length / 2.0
    # Ножки
    left_leg = Polygon([
        (-width / 2.0, base),
        (-width / 2.0 + leg_width, base),
        (-width / 2.0 + leg_width, top),
        (-width / 2.0, top),
        (-width / 2.0, base),
    ])
    right_leg = affinity.translate(left_leg, xoff=width - leg_width)
    # Перемычки
    bar_half = leg_width
    top_bar = Polygon([
        (-gap / 2.0, top - bar_half),
        (gap / 2.0, top - bar_half),
        (gap / 2.0, top),
        (-gap / 2.0, top),
        (-gap / 2.0, top - bar_half),
    ])
    bottom_bar = affinity.translate(top_bar, yoff=-(leg_length - leg_width))
    middle_bar = Polygon([
        (-gap / 2.0, -bar_half / 2.0),
        (gap / 2.0, -bar_half / 2.0),
        (gap / 2.0, bar_half / 2.0),
        (-gap / 2.0, bar_half / 2.0),
        (-gap / 2.0, -bar_half / 2.0),
    ])
    h_poly = left_leg.union(right_leg).union(top_bar).union(bottom_bar).union(middle_bar)
    rotated = affinity.rotate(h_poly, math.degrees(angle), origin=(0, 0))
    moved = affinity.translate(rotated, xoff=center.x, yoff=center.y)
    return moved.buffer(0)


def _circle_polygon(radius: float, center: Point, params: GeometryParams) -> Polygon:
    return Point(center.x, center.y).buffer(radius, resolution=max(8, params.circle_segments))


def _rectangle_with_optional_courtyard(length: float, width: float, angle: float,
                                       center: Point, params: GeometryParams,
                                       with_courtyard: bool) -> Polygon:
    length = max(length, width)
    width = max(width, params.wall_thickness * 2)
    rect_coords = [
        (-length / 2.0, -width / 2.0),
        (length / 2.0, -width / 2.0),
        (length / 2.0, width / 2.0),
        (-length / 2.0, width / 2.0),
        (-length / 2.0, -width / 2.0),
    ]
    rect = Polygon(rect_coords)
    if with_courtyard:
        inner_len = length - 2.0 * params.wall_thickness
        inner_w = width - 2.0 * params.wall_thickness
        if inner_len > 0 and inner_w > 0 and inner_len * inner_w >= params.min_courtyard_area:
            hole = [
                (-inner_len / 2.0, -inner_w / 2.0),
                (inner_len / 2.0, -inner_w / 2.0),
                (inner_len / 2.0, inner_w / 2.0),
                (-inner_len / 2.0, inner_w / 2.0),
                (-inner_len / 2.0, -inner_w / 2.0),
            ]
            rect = Polygon(rect_coords, holes=[hole])
    rotated = affinity.rotate(rect, math.degrees(angle), origin=(0, 0))
    moved = affinity.translate(rotated, xoff=center.x, yoff=center.y)
    return moved.buffer(0)


def _rectangle_for_angle(points: List[Tuple[float, float]], width: float,
                         params: GeometryParams) -> Polygon:
    if len(points) < 3:
        return MultiPoint(points).buffer(width / 2.0)
    line = LineString(points)
    return line.buffer(width / 2.0, cap_style=2, join_style=2)


def build_geometry_for_cluster(points: List[Point], block_geom: Polygon,
                               params: GeometryParams) -> Tuple[Polygon, Dict[str, Any]]:
    multipoint = MultiPoint([p for p in points if not p.is_empty])
    centroid = multipoint.centroid
    hull = multipoint.convex_hull
    if hull.is_empty:
        return centroid.buffer(params.wall_thickness), {"shape": "point"}
    rect = hull.minimum_rotated_rectangle
    if rect.geom_type != "Polygon":
        rect = hull.envelope
    long_side, short_side, angle = _rectangle_dimensions(rect)
    if long_side <= 0 or short_side <= 0:
        size = max(math.sqrt(hull.area), params.wall_thickness * 2)
        return _square_polygon(size, 0.0, centroid, params, add_courtyard=False), {
            "shape": "fallback",
            "roundness": 1.0,
            "long_side": size,
            "short_side": size,
        }
    roundness = _cluster_roundness(long_side, short_side)
    geom_info: Dict[str, Any] = {
        "roundness": roundness,
        "long_side": long_side,
        "short_side": short_side,
    }

    add_courtyard_square = False
    shape_type = "square"

    if roundness <= params.round_ratio_max:
        # Кластер похож на круг
        add_courtyard_square = True
        if _is_central(centroid, block_geom, params):
            shape_type = "h"
            geom = _h_shape_polygon(long_side, short_side, angle, centroid, params)
        elif _is_corner(centroid, block_geom, params):
            shape_type = "circle"
            radius = max(short_side, params.wall_thickness * 2) / 2.0
            geom = _circle_polygon(radius, centroid, params)
        else:
            size = max(short_side, params.wall_thickness * 3)
            geom = _square_polygon(size, angle, centroid, params, add_courtyard_square)
    else:
        elongated = (long_side / max(short_side, 1e-6)) >= params.elongated_ratio_min
        angles = _convex_hull_angles(hull)
        bend_angles = [a for a in angles if params.bend_angle_min <= a <= params.bend_angle_max]
        is_bent = bool(bend_angles)
        shape_type = "rectangle"
        with_courtyard = elongated and short_side >= params.narrow_width_max
        width = short_side
        if short_side < params.wall_thickness * 2:
            width = params.wall_thickness * 2
        if short_side < params.narrow_width_max:
            width = max(params.wall_thickness, min(30.0, short_side))
            with_courtyard = False
            shape_type = "rectangle_narrow"
        if is_bent and len(bend_angles) > 0:
            shape_type = "rectangle_bent"
            bend_vertex = _pick_bend_vertex(hull, params)
            bend_pts = _bend_polyline(hull, bend_vertex)
            geom = _rectangle_for_angle(bend_pts, width, params)
        else:
            geom = _rectangle_with_optional_courtyard(long_side, width, angle, centroid,
                                                      params, with_courtyard)

    geom_info["shape"] = shape_type
    return geom, geom_info


def _pick_bend_vertex(hull: Polygon, params: GeometryParams) -> Tuple[int, Tuple[float, float]]:
    coords = list(hull.exterior.coords)[:-1]
    best_idx = 0
    best_angle = 0.0
    for i in range(len(coords)):
        prev_pt = coords[(i - 1) % len(coords)]
        cur_pt = coords[i]
        next_pt = coords[(i + 1) % len(coords)]
        ang = _angle_between((prev_pt[0] - cur_pt[0], prev_pt[1] - cur_pt[1]),
                             (next_pt[0] - cur_pt[0], next_pt[1] - cur_pt[1]))
        if ang is None:
            continue
        if params.bend_angle_min <= ang <= params.bend_angle_max and ang > best_angle:
            best_angle = ang
            best_idx = i
    return best_idx, coords[best_idx]


def _bend_polyline(hull: Polygon, bend_info: Tuple[int, Tuple[float, float]]) -> List[Tuple[float, float]]:
    coords = list(hull.exterior.coords)[:-1]
    if not coords:
        return []
    idx, vertex = bend_info
    prev_pt = coords[(idx - 1) % len(coords)]
    next_pt = coords[(idx + 1) % len(coords)]
    return [prev_pt, vertex, next_pt]


def aggregate_cluster_attributes(indices: List[int], props: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg = _aggregate_points(indices, props)
    agg["cluster_size"] = len(indices)
    return agg


# ---- Основной цикл ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer + clustering-based building polygons")
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
    ap.add_argument("--out-buildings", required=True, help="Выход GeoJSON полигонов зданий")
    ap.add_argument("--out-centroids", default=None, help="Выход GeoJSON центроидов")
    ap.add_argument("--out-clusters-centers", default=None,
                    help="Выход GeoJSON центров кластеров")
    ap.add_argument("--out-epsg", type=int, default=32636)
    ap.add_argument("--shape-params", default="building_shape_params.yaml",
                    help="YAML с параметрами геометрии зданий")
    ap.add_argument("--temp-dir", default=None)
    args = ap.parse_args()

    cparams, gparams = load_params(args.shape_params)
    clusterer = Clusterer(cparams)

    blocks = read_geojson(args.blocks)
    features = blocks.get("features", [])
    targets_map = normalize_targets_map(load_json_maybe(args.targets_by_zone))
    la_by_zone = normalize_targets_map(load_json_maybe(args.la_by_zone))
    services_vocab = load_services_vocab_from_artifacts(args.model_ckpt)
    service_keys = pick_service_keys(services_vocab)

    out_building_features: List[Dict[str, Any]] = []
    out_centroids_features: List[Dict[str, Any]] = []
    out_cluster_centers: List[Dict[str, Any]] = []

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
        if zone_label in targets_map:
            la_target = targets_map[zone_label].get("la")
            floors_avg = targets_map[zone_label].get("floors_avg")
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
            "--zone", str(zone_label),
            "--infer-slots", str(infer_slots),
            "--infer-knn", str(args.infer_knn),
            "--infer-e-thr", str(args.infer_e_thr),
            "--infer-il-thr", str(args.infer_il_thr),
            "--infer-sv1-thr", str(args.infer_sv1_thr),
        ]
        if args.device:
            cmd += ["--device", args.device]
        if args.config:
            cmd += ["--config", args.config]
        if args.zones_json:
            cmd += ["--zones-json", args.zones_json]
        if args.services_json:
            cmd += ["--services-json", args.services_json]
        if la_target is not None:
            cmd += ["--la-target", str(la_target)]
        if floors_avg is not None:
            cmd += ["--floors-avg", str(floors_avg)]
        if services_target:
            cmd += ["--services-target", json.dumps(services_target, ensure_ascii=False)]

        try:
            subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] infer failed for block #{bi}: {exc}", file=sys.stderr)
            continue

        try:
            infer_output = read_geojson(str(out_path))
        except Exception as exc:
            print(f"[WARN] failed to read infer output for block #{bi}: {exc}", file=sys.stderr)
            continue

        centroids_raw: List[Point] = []
        cprops_raw: List[Dict[str, Any]] = []
        infer_uid = 0
        for b in infer_output.get("features", []):
            try:
                g = shape(b.get("geometry"))
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
            raw_props = dict(b.get("properties") or {})
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
        for p, pr in zip(merged_points, merged_props):
            q = p
            status = "none"
            props_copy = dict(pr)
            if ring2 and not ring2.is_empty and mid_lines:
                in_r1 = ring1.contains(p) or ring1.touches(p)
                in_r2 = ring2.contains(p) or ring2.touches(p)
                if in_r1 or in_r2:
                    status = "first" if in_r1 and not in_r2 else "second"
                    li, q_on, _m = nearest_on_lines(mid_lines, p)
                    if li >= 0:
                        q = q_on
                    props_copy["ring_snap"] = "to_ring2_midline"
                    props_copy["ring_zone"] = status
            if "ring_snap" not in props_copy:
                props_copy["ring_snap"] = "none"
                props_copy["ring_zone"] = status
            moved_points.append(q)
            moved_props.append(props_copy)

        for p, pr in zip(moved_points, moved_props):
            out_centroids_features.append({
                "type": "Feature",
                "geometry": mapping(p),
                "properties": pr,
            })

        coords = np.array([[p.x, p.y] for p in moved_points], dtype=float)
        labels = clusterer.fit_predict(coords)
        cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            if label < 0:
                continue
            cluster_to_indices[label].append(idx)

        if not cluster_to_indices:
            continue

        for cluster_id, indices in cluster_to_indices.items():
            pts = [moved_points[i] for i in indices]
            geom_cluster, geom_info = build_geometry_for_cluster(pts, geom, gparams)
            attrs = aggregate_cluster_attributes(indices, moved_props)
            props = {
                "block_index": block_idx,
                "zone": zone_label,
                "cluster_id": int(cluster_id),
            }
            props.update(attrs)
            props.update(geom_info)

            out_building_features.append({
                "type": "Feature",
                "geometry": mapping(geom_cluster),
                "properties": props,
            })

            out_cluster_centers.append({
                "type": "Feature",
                "geometry": mapping(MultiPoint(pts).centroid),
                "properties": {
                    "block_index": block_idx,
                    "zone": zone_label,
                    "cluster_id": int(cluster_id),
                    "shape": geom_info.get("shape"),
                    "roundness": geom_info.get("roundness"),
                },
            })

    if out_building_features:
        write_geojson(args.out_buildings, {
            "type": "FeatureCollection",
            "features": out_building_features,
        }, epsg=args.out_epsg)

    if args.out_centroids:
        write_geojson(args.out_centroids, {
            "type": "FeatureCollection",
            "features": out_centroids_features,
        }, epsg=args.out_epsg)

    if args.out_clusters_centers:
        write_geojson(args.out_clusters_centers, {
            "type": "FeatureCollection",
            "features": out_cluster_centers,
        }, epsg=args.out_epsg)


if __name__ == "__main__":  # pragma: no cover
    main()
