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
import os, sys, json, argparse, subprocess, math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import geopandas as gpd

try:  # tqdm опционален
    from tqdm import tqdm
except Exception:  # pragma: no cover - резерв
    tqdm = lambda x, **k: x

try:
    from shapely.geometry import Point, Polygon, LineString, MultiLineString, mapping, shape
except Exception:  # pragma: no cover
    print("[FATAL] Требуется shapely и её зависимости (GEOS).", file=sys.stderr)
    raise

from postprocessing import DensityIsolines, GridGenerator, BuildingGenerator, BuildingAttributes


SYNONYMS = {
    "school":       ["school", "школа", "общеобразовательная школа", "образовательная организация"],
    "kindergarten": ["kindergarten", "детский сад", "детсад", "дошкольное", "д/с"],
    "polyclinic":   ["polyclinic", "clinic", "поликлиника", "амбулатория", "клиника"],
}


def build_rings(block: Polygon) -> Tuple[Any, Any, Any]:
    """Возвращает (ring1, ring2, midline) где:
       ring1 = 0–10 м, ring2 = 10–30 м, midline = граница buffer(-20) ∩ ring2 (MultiLineString/LineString).
       Если ring2 пуст, midline пуст.
    """
    inner10 = block.buffer(-10.0)
    ring1 = block.difference(inner10)
    inner30 = block.buffer(-30.0)
    ring2 = inner10.difference(inner30) if (inner30 and not inner30.is_empty) else inner10.difference(Polygon())
    # если после вычитания получилось пусто (узкий квартал) — обнулим
    if ring2 is None or ring2.is_empty:
        return ring1, ring2, LineString()
    mid20_poly = block.buffer(-20.0)
    midline_raw = mid20_poly.boundary
    midline = midline_raw.intersection(ring2)
    return ring1, ring2, midline

def get_people(props: Dict[str,Any], default_people: int) -> int:
    for k in ("population","people","POPULATION","num_people"):
        if k in props:
            try:
                v = int(float(props[k]))
                if v > 0: return v
            except Exception:
                pass
    return int(default_people)

def infer_slots_from_block_bbox(block_geom, cell_size_m: float = 100.0) -> int:
    """
    Возвращает число КВАДРАТОВ по стороне прямоугольной сетки (N),
    построенной по bbox квартала при размере ячейки cell_size_m (м).
    Используется как значение для --infer-slots (то есть N, а не N*N).

    ВАЖНО: CRS должен быть метрическим.
    """
    if block_geom is None or block_geom.is_empty:
        return 1
    minx, miny, maxx, maxy = block_geom.bounds
    w = float(maxx - minx)
    h = float(maxy - miny)
    if w <= 0.0 or h <= 0.0:
        return 1
    nx = int(math.ceil(w / cell_size_m))
    ny = int(math.ceil(h / cell_size_m))
    n_side = max(nx, ny)
    if max(1, n_side) > 5000:
        return 5000
    else:
        return max(1, n_side)
    
def lines_of(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    # boundary может вернуть LinearRing — он является LineString в геометриях Shapely
    return []

def load_json_maybe(path_or_json: str) -> Any:
    try:
        if path_or_json is None:
            return None
        if os.path.exists(path_or_json):
            with open(path_or_json, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return json.loads(path_or_json)
    except Exception as e:
        print(f"[WARN] failed to read JSON '{path_or_json}': {e}", file=sys.stderr)
        return None
    
def load_services_vocab_from_artifacts(model_ckpt: str) -> Dict[str, int] | None:
    aux_dir = os.path.join(os.path.dirname(model_ckpt) or ".", "artifacts")
    sj = os.path.join(aux_dir, "services.json")
    if os.path.exists(sj):
        try:
            with open(sj, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def lines_of(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    # boundary может вернуть LinearRing — он является LineString в геометриях Shapely
    return []



def nearest_on_lines(lines: List[LineString], p: Point) -> Tuple[int, Point, float]:
    """Находит ближайший компонент линии и возвращает (index, point_on_line, m_along)."""
    best_i = -1
    best_q = None
    best_d = 1e100
    best_m = 0.0
    for i, ln in enumerate(lines):
        # ближайшая точка на конкретной линии
        m = ln.project(p)
        q = ln.interpolate(m)
        d = q.distance(p)
        if d < best_d:
            best_d = d; best_q = q; best_i = i; best_m = m
    if best_q is None:
        return -1, p, 0.0
    return best_i, best_q, best_m

def normalize_targets_map(raw: Any) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, dict):
        return out
    for zone, val in raw.items():
        if not isinstance(zone, str) or not isinstance(val, dict): 
            continue
        la = None; fl = None
        for k in ("la", "living_area", "la_target"):
            if k in val and val[k] is not None:
                try:
                    la = float(val[k]); break
                except Exception:
                    pass
        for k in ("floors_avg", "floors"):
            if k in val and val[k] is not None:
                try:
                    fl = float(val[k]); break
                except Exception:
                    pass
        out[zone] = {"la": la, "floors_avg": fl}
    return out

def pick_service_keys(vocab: Dict[str,int] | None) -> Dict[str,str]:
    out = {"school":"school", "kindergarten":"kindergarten", "polyclinic":"polyclinic"}
    if not vocab:
        return out
    names = list(vocab.keys())
    lowered = [s.casefold() for s in names]
    for canon, syns in SYNONYMS.items():
        for syn in syns:
            cf = syn.casefold()
            if cf in lowered:
                out[canon] = names[lowered.index(cf)]; break
            idx = next((i for i,s in enumerate(lowered) if cf in s), None)
            if idx is not None:
                out[canon] = names[idx]; break
    return out

def read_geojson(path: str) -> Dict[str,Any]:
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    if gj.get("type") != "FeatureCollection":
        if gj.get("type") == "Feature":
            feats = [gj]
        else:
            feats = [{"type":"Feature","geometry":gj,"properties":{}}]
        gj = {"type":"FeatureCollection", "features": feats}
    return gj

def write_geojson(path: str, fc: Dict[str,Any], epsg: Optional[int] = None) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    if epsg:
        fc = dict(fc)
        fc["crs"] = {"type": "name", "properties": {"name": f"EPSG:{int(epsg)}"}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

def zone_label_of(feat: Dict[str,Any], zone_attr: str) -> Optional[str]:
    props = feat.get("properties") or {}
    z = props.get(zone_attr)
    return z if isinstance(z, str) else None

def _to_float(x) -> Optional[float]:
    try:
        if isinstance(x, bool): return 1.0 if x else 0.0
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in ("true","yes","y","t","1"): return 1.0
            if xs in ("false","no","n","f","0"): return 0.0
            return float(x)
    except Exception:
        return None
    return None

def get_floors_value(props: Dict[str,Any]) -> Optional[float]:
    for k in ("floors", "floors_avg", "floors_num"):
        if k in props and props[k] is not None:
            v = _to_float(props[k])
            if v is not None and v > 0:
                return v
    return None

def get_is_living_value(props: Dict[str,Any]) -> Optional[float]:
    for k in ("is_living", "living", "is_residential"):
        if k in props and props[k] is not None:
            v = _to_float(props[k])
            if v is not None:
                return max(0.0, min(1.0, v))
    return None

def get_living_area_value(props: Dict[str,Any]) -> Optional[float]:
    for k in ("living_area", "la", "area_living", "la_target"):
        if k in props and props[k] is not None:
            v = _to_float(props[k])
            if v is not None and v >= 0:
                return v
    return None


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

        empty_grid = GridGenerator.make_grid_for_blocks(
            blocks_gdf=blocks,
            cell_size_m=args.grid_size, 
            midlines=gpd.GeoSeries([midline], crs=blocks.crs),
            block_id_col="block_id",
            offset_m=20.0
        )
        preicted_points = gpd.GeoDataFrame.from_features(out_centroids_features, crs="EPSG:32636")
        isolines = DensityIsolines.build(blocks, preicted_points, zone_id_col='zone')
        grid = GridGenerator.fit_transform(empty_grid, isolines)
        generation_result = BuildingGenerator.fit_transform(grid, blocks, zone_name_aliases=['zone'])
        buildings = generation_result["buildings_rects"]
        service_territories = generation_result["service_sites"]
        buildings = BuildingAttributes.fit_transform(buildings, blocks)["buildings"]

        preicted_points.to_file('centroids.geojson')
        isolines.to_file('isolines.geojson')
        grid.to_file('grid.geojson')
        service_territories.to_file('service_territories.geojson')
        buildings.to_file('buildings.geojson')

def json_dumps(obj: Any) -> str:
    try:
        import json

        return json.dumps(obj)
    except Exception:  # pragma: no cover
        raise


if __name__ == "__main__":  # pragma: no cover
    main()