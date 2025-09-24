#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_infer_blocks.py — пакетная генерация застройки через train.py --mode infer
с последующей нарезкой квартала на кольца (внутренние буферы) и сектора от направлений на точки.
Площадь сектора ∝ весу точки (living_area/floors, затем living_area, иначе 1).

Совместимо с Shapely < 2.0.
Зависимости: shapely (1.7/1.8), numpy, tqdm (опц.).

ВЫХОДЫ (4 шт):
  1) --out                 : полигоны-сектора (rings/sectors)
  2) --out-points          : исходные точки для разрезки (центроиды полигонов инференса + исходные Point)
  3) --out-infer-polys     : полигоны, выданные инференсом (Polygon/MultiPolygon как есть)
  4) --out-infer-centroids : центроиды полигонов инференса (Point), свойства 1:1 с полигоном

Для пары (полигон инференса, его центроид) свойства строго идентичны, добавлен общий ключ: infer_fid.
"""

from __future__ import annotations
import os, sys, json, argparse, tempfile, subprocess, shutil, math
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x

# --- Гео-зависимости (Shapely < 2.0 поддерживается) ---
try:
    from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon
    from shapely.ops import unary_union
except Exception:
    print("[FATAL] Требуется shapely и её зависимости (GEOS).", file=sys.stderr)
    raise

import numpy as np

# ---- I/O ----

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

def write_geojson(path: str, fc: Dict[str,Any]) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

def zone_label_of(feat: Dict[str,Any], zone_attr: str) -> Optional[str]:
    props = feat.get("properties") or {}
    z = props.get(zone_attr)
    return z if isinstance(z, str) else None

# ---- Старые утилиты для целей (оставлены для совместимости с твоими флагами) ----

SYNONYMS = {
    "school":       ["school", "школа", "общеобразовательная школа", "образовательная организация"],
    "kindergarten": ["kindergarten", "детский сад", "детсад", "дошкольное", "д/с"],
    "polyclinic":   ["polyclinic", "clinic", "поликлиника", "амбулатория", "клиника"],
}

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

def get_people(props: Dict[str,Any], default_people: int) -> int:
    for k in ("population","people","POPULATION","num_people"):
        if k in props:
            try:
                v = int(float(props[k]))
                if v > 0: return v
            except Exception:
                pass
    return int(default_people)

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

# ---- Гео-хелперы ----

def block_rings(block, step_m: float) -> List[Polygon]:
    """Возвращает список «колец» (Polygon) как разности последовательных внутренних буферов с шагом step_m.
       Последнее центральное «ядро» тоже добавляется (как кольцо меньшей ширины)."""
    if isinstance(block, MultiPolygon):
        block = unary_union(block)
    if block.is_empty:
        return []
    rings: List[Polygon] = []
    prev = block
    k = 1
    while True:
        next_poly = block.buffer(-step_m * k)
        ring = prev.difference(next_poly)
        if not ring.is_empty and ring.area > 0:
            if isinstance(ring, MultiPolygon):
                ring = unary_union(ring)
            if not ring.is_empty and ring.area > 0:
                rings.append(ring)
        if next_poly.is_empty or next_poly.area <= 0:
            if (not next_poly.is_empty) and (next_poly.area > 0):
                rings.append(next_poly if isinstance(next_poly, Polygon) else unary_union(next_poly))
            break
        prev = next_poly
        k += 1
    return rings

def sector_wedge(center: Tuple[float,float], theta0: float, theta1: float, radius: float, arc_pts: int=64) -> Polygon:
    """Строит «пирог» (сектор круга) от theta0 до theta1 радиуса radius с вершиной в center."""
    cx, cy = center
    while theta1 < theta0:
        theta1 += 2.0*math.pi
    ts = np.linspace(theta0, theta1, max(3, int(arc_pts * (theta1-theta0)/(2*math.pi))))
    pts = [(cx, cy)] + [(cx + radius*math.cos(t), cy + radius*math.sin(t)) for t in ts] + [(cx, cy)]
    return Polygon(pts)

def point_weight_from_props(props: Dict[str,Any]) -> float:
    """w = living_area/floors (если есть), иначе living_area, иначе 1."""
    la = None; fl = None
    for k in ("living_area", "la", "area_living", "la_target"):
        if k in props and props[k] is not None:
            try:
                la = float(props[k]); break
            except Exception:
                pass
    for k in ("floors", "floors_num", "floors_avg"):
        if k in props and props[k] is not None:
            try:
                fl = float(props[k]); break
            except Exception:
                pass
    if la is not None and fl and fl > 0:
        return max(1e-9, la / fl)
    if la is not None:
        return max(1e-9, la)
    return 1.0

# ---- Основной скрипт ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer + ring/sector partition (Shapely < 2.0)")
    ap.add_argument("--blocks", required=True, help="Входной GeoJSON (FeatureCollection) кварталов")
    ap.add_argument("--out", required=True, help="Выходной GeoJSON полигонов-секторов")
    ap.add_argument("--out-points", default=None, help="Выходной GeoJSON исходных точек для разрезки (по умолчанию <OUT>_points.geojson)")
    ap.add_argument("--out-infer-polys", default=None, help="Выход GeoJSON полигонов, выданных инференсом (Polygon/MultiPolygon)")
    ap.add_argument("--out-infer-centroids", default=None, help="Выход GeoJSON центроидов этих полигонов (Point), свойства 1:1")
    ap.add_argument("--train-script", default="./train.py", help="Путь к train.py")
    ap.add_argument("--model-ckpt", required=True, help="Путь к чекпойнту модели (.pt)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None, help="cuda|cpu")
    ap.add_argument("--zone-attr", default="zone", help="Имя свойства зоны в кварталах")
    # Совместимость со «старыми» флагами
    ap.add_argument("--zones-json", default=None)
    ap.add_argument("--services-json", default=None)
    ap.add_argument("--targets-by-zone", default=None)
    ap.add_argument("--la-by-zone", default=None)
    ap.add_argument("--people", type=int, default=1000)
    ap.add_argument("--min-services", dest="min_services", action="store_true", default=True)
    ap.add_argument("--no-min-services", dest="min_services", action="store_false")
    # Инференс
    ap.add_argument("--infer-slots", type=int, default=256)
    ap.add_argument("--infer-knn", type=int, default=8)
    ap.add_argument("--infer-e-thr", type=float, default=0.5)
    ap.add_argument("--infer-il-thr", type=float, default=0.5)
    ap.add_argument("--infer-sv1-thr", type=float, default=0.5)
    # Параметры разрезки
    ap.add_argument("--ring-step-m", type=float, default=60.0, help="Шаг внутренних буферов, м")
    ap.add_argument("--arc-resolution", type=int, default=96, help="Точность аппроксимации дуг")
    ap.add_argument("--origin", choices=["ring_centroid","block_centroid"], default="ring_centroid",
                    help="Центр разбиения на сектора (по умолчанию — центроид каждого кольца)")

    args = ap.parse_args()

    stem = os.path.splitext(args.out)[0]
    out_points_path = args.out_points or (stem + "_points.geojson")
    out_infer_polys_path = args.out_infer_polys or (stem + "_infer_polys.geojson")
    out_infer_centroids_path = args.out_infer_centroids or (stem + "_infer_centroids.geojson")

    # Цели по зонам (оставлено для совместимости / передачи в train.py)
    vocab = load_services_vocab_from_artifacts(args.model_ckpt)
    svc_keys = pick_service_keys(vocab)
    targets_by_zone: Dict[str, Dict[str, float]] = {}
    raw_targets = load_json_maybe(args.targets_by_zone) or load_json_maybe(args.la_by_zone)
    if raw_targets:
        if args.targets_by_zone:
            targets_by_zone = normalize_targets_map(raw_targets)
        else:
            if isinstance(raw_targets, dict):
                targets_by_zone = {z: {"la": (float(v) if v is not None else None), "floors_avg": None}
                                   for z, v in raw_targets.items()}

    fc_blocks = read_geojson(args.blocks)
    out_sector_feats: List[Dict[str,Any]] = []
    out_point_feats:  List[Dict[str,Any]] = []
    out_infer_poly_feats: List[Dict[str,Any]] = []
    out_infer_centroid_feats: List[Dict[str,Any]] = []
    infer_uid = 0  # общий идентификатор пар (полигон<->центроид)

    tmpdir = tempfile.mkdtemp(prefix="ked_infer_")
    try:
        for bi, feat in enumerate(tqdm(fc_blocks.get("features", []), desc="Blocks")):
            z = zone_label_of(feat, args.zone_attr)
            if not z:
                print(f"[WARN] feature #{bi}: нет properties['{args.zone_attr}'] — пропуск", file=sys.stderr)
                continue

            block_geom = shape(feat["geometry"])
            if block_geom.is_empty:
                continue

            # Временные файлы для инференса
            in_path  = os.path.join(tmpdir, f"blk_{bi:06d}.geojson")
            out_path = os.path.join(tmpdir, f"blk_{bi:06d}_out.geojson")
            write_geojson(in_path, {"type":"FeatureCollection","features":[feat]})

            # Цели (если нужны train.py)
            la_target = None; floors_avg = None; services_target = {}
            tz = targets_by_zone.get(z or "", {})
            la_target = tz.get("la")
            floors_avg = tz.get("floors_avg")

            if (la_target is None) and (str(z).casefold() == "residential"):
                people = get_people(feat.get("properties") or {}, args.people)
                la_target = float(15.0 * people)
                if args.min_services and vocab:
                    services_target[svc_keys["school"]]       = 1.0
                    services_target[svc_keys["polyclinic"]]   = 1.0
                    services_target[svc_keys["kindergarten"]] = 1.0

            # Инференс одного квартала
            cmd = [
                sys.executable, args.train_script,
                "--mode", "infer",
                "--model-ckpt", args.model_ckpt,
                "--infer-geojson-in", in_path,
                "--infer-out", out_path,
                "--zone", str(z),
                "--infer-slots", str(args.infer_slots),
                "--infer-knn",   str(args.infer_knn),
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
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] infer failed for block #{bi} (zone={z}): {e}", file=sys.stderr)
                continue

            # Читаем результат квартала и вытаскиваем полигоны/точки + веса
            try:
                blk_out = read_geojson(out_path)
            except Exception as e:
                print(f"[WARN] failed to read output for block #{bi}: {e}", file=sys.stderr)
                continue

            pts: List[Point] = []
            pprops_for_weights: List[Dict[str,Any]] = []
            weights: List[float] = []

            for b in blk_out.get("features", []):
                try:
                    g = shape(b["geometry"])
                    raw_props = dict(b.get("properties") or {})

                    # Нормализованные общие свойства, которые хотим иметь и у полигона, и у его центроида
                    shared_props = dict(raw_props)
                    shared_props.setdefault(args.zone_attr, z)
                    shared_props.setdefault("_source_block_index", bi)

                    if isinstance(g, (Polygon, MultiPolygon)) and (not g.is_empty) and (g.area > 0):
                        # 1) Сохраняем полигон инференса
                        infer_uid += 1
                        shared_props["infer_fid"] = infer_uid

                        out_infer_poly_feats.append({
                            "type":"Feature",
                            "geometry": mapping(g),
                            "properties": dict(shared_props)  # строго те же свойства
                        })

                        # 2) Его центроид (точка) — со строго идентичными свойствами
                        c = g.centroid
                        out_infer_centroid_feats.append({
                            "type":"Feature",
                            "geometry": mapping(c),
                            "properties": dict(shared_props)
                        })

                        # 3) Точка для разрезки/веса — центроид с тех. полями (может отличаться набором полей)
                        props_for_point = dict(shared_props)
                        props_for_point["_ptx"] = c.x
                        props_for_point["_pty"] = c.y
                        pts.append(c)
                        pprops_for_weights.append(props_for_point)
                        weights.append(point_weight_from_props(shared_props))

                    elif isinstance(g, Point):
                        # Исходный Point из инференса — идёт только в "points" (для разрезки/весов)
                        p = g
                        props_for_point = dict(shared_props)
                        props_for_point["_ptx"] = p.x
                        props_for_point["_pty"] = p.y
                        pts.append(p)
                        pprops_for_weights.append(props_for_point)
                        weights.append(point_weight_from_props(shared_props))
                    else:
                        # Другие типы геометрий игнорируем
                        continue
                except Exception:
                    continue

            # Сохраняем точки как есть (для последующего анализа и как «источники направлений»)
            for p, props in zip(pts, pprops_for_weights):
                out_point_feats.append({
                    "type":"Feature",
                    "geometry": mapping(p),
                    "properties": props
                })

            if not pts:
                # Нечего нарезать — следующий блок
                continue

            # Режем квартал на кольца
            rings = block_rings(block_geom, step_m=float(args.ring_step_m))
            if not rings:
                rings = [block_geom]

            # Общая нормировка весов
            W = float(sum(max(0.0, w) for w in weights))
            if W <= 0:
                weights = [1.0 for _ in weights]
                W = float(len(weights))

            # Подготовим порядок точек по углу (стабильность разметки)
            cblk = block_geom.centroid
            ang_order = np.argsort([math.atan2(p.y - cblk.y, p.x - cblk.x) for p in pts])
            pts_ord   = [pts[i] for i in ang_order]
            props_ord = [pprops_for_weights[i] for i in ang_order]
            w_ord     = [weights[i] for i in ang_order]

            # Угловые доли для каждой точки: 2π * w / W
            theta_sizes = [2.0*math.pi * (w / W) for w in w_ord]

            # Для каждого кольца делаем сектора
            for ri, ring in enumerate(rings):
                if ring.is_empty or ring.area <= 0:
                    continue
                # Центр разбиения
                if args.origin == "block_centroid":
                    c = (cblk.x, cblk.y)
                else:
                    cr = ring.centroid
                    c = (cr.x, cr.y)

                # Радиус для «пирога»
                minx, miny, maxx, maxy = ring.bounds
                R = 1.5 * math.hypot(maxx - minx, maxy - miny)

                # Начальный угол — по первой точке
                theta0 = math.atan2(pts_ord[0].y - c[1], pts_ord[0].x - c[0])

                for j, (prop_j, dth) in enumerate(zip(props_ord, theta_sizes)):
                    theta1 = theta0 + dth
                    wedge = sector_wedge(c, theta0, theta1, R, arc_pts=int(args.arc_resolution))
                    sect = ring.intersection(wedge)
                    theta0 = theta1  # следующий сектор начинается с конца предыдущего

                    if sect.is_empty:
                        continue

                    # Клон свойств точки + служебные поля сектора
                    pprops_dup = dict(prop_j)
                    pprops_dup.update({
                        "ring_index": int(ri),
                        "sector_index": int(j),
                        "sector_theta_start_deg": float((theta1 - dth) * 180.0 / math.pi),
                        "sector_theta_end_deg":   float(theta1 * 180.0 / math.pi),
                        "sector_area_target": float(ring.area * (dth / (2.0*math.pi))),
                        "sector_area_actual": float(sect.area),
                        "partition": "rings_sectors_v1",
                        "ring_step_m": float(args.ring_step_m),
                        "origin_mode": args.origin,
                    })

                    # sect может быть MultiPolygon — разобьём на компоненты
                    if isinstance(sect, MultiPolygon):
                        for k, gk in enumerate(sect.geoms):
                            if gk.is_empty or gk.area <= 0:
                                continue
                            pk = dict(pprops_dup)
                            pk["sector_part"] = k
                            out_sector_feats.append({
                                "type":"Feature",
                                "geometry": mapping(gk),
                                "properties": pk
                            })
                    else:
                        out_sector_feats.append({
                            "type":"Feature",
                            "geometry": mapping(sect),
                            "properties": pprops_dup
                        })

        # Запись результатов
        write_geojson(args.out, {"type":"FeatureCollection","features": out_sector_feats})
        write_geojson(out_points_path, {"type":"FeatureCollection","features": out_point_feats})
        write_geojson(out_infer_polys_path, {"type":"FeatureCollection","features": out_infer_poly_feats})
        write_geojson(out_infer_centroids_path, {"type":"FeatureCollection","features": out_infer_centroid_feats})

        print(f"[OK] sectors:         {len(out_sector_feats)} → {args.out}")
        print(f"[OK] weight points:   {len(out_point_feats)} → {out_points_path}")
        print(f"[OK] infer polygons:  {len(out_infer_poly_feats)} → {out_infer_polys_path}")
        print(f"[OK] infer centroids: {len(out_infer_centroid_feats)} → {out_infer_centroids_path}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())
