#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_infer_blocks.py — пакетная генерация застройки через train.py --mode infer
с последующим слиянием центроидов внутри кварталов и построением графа соседства.

Совместимо с Shapely < 2.0.
Зависимости: shapely (1.7/1.8), numpy, tqdm (опц.).

ВЫХОДЫ (3 шт):
  1) --out                 : граф как линии (LineString) между СЛИТЫМИ центроидами зданий
  2) --out-infer-polys     : полигоны, выданные инференсом (Polygon/MultiPolygon как есть)
  3) --out-infer-centroids : СЛИТЫЕ центроиды (Point) с агрегированными свойствами:
       - e            : из якоря кластера (наиболее вероятного)
       - floors       : ceil(среднего по кластеру)  (дополнительно floors_avg — само среднее)
       - is_living    : среднее (0..1)
       - living_area  : сумма
       - merged_from  : список infer_fid, вошедших в кластер
       - merged_count : размер кластера

Примечание: расстояния считаются в тех же единицах, что CRS входа; для «метров» используйте метрическую проекцию.
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
    from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon, LineString
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

# ---- Совместимость (targets) ----

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

# ---- Вспомогательные извлечения атрибутов ----

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

# ---- Слияние центроидов внутри квартала ----

def merge_centroids(points: List[Point],
                    props_list: List[Dict[str,Any]],
                    radius_m: float,
                    prob_field: str = "e") -> Tuple[List[Point], List[Dict[str,Any]]]:
    """
    Кластеризуем центроиды внутри квартала: сортировка по убыванию prob_field,
    якорь поглощает соседей в радиусе radius_m с меньшей/равной вероятностью.
    Геометрия кластера = точка якоря.
    """
    n = len(points)
    if n <= 1:
        return list(points), list(props_list)

    order = sorted(range(n), key=lambda i: _to_float(props_list[i].get(prob_field)) or 0.0, reverse=True)
    taken = [False]*n
    m_points: List[Point] = []
    m_props:  List[Dict[str,Any]] = []

    for idx in order:
        if taken[idx]:
            continue
        # создаём кластер с якорем idx
        anchor_i = idx
        anchor_p = points[anchor_i]
        anchor_pr = props_list[anchor_i]
        anchor_e = _to_float(anchor_pr.get(prob_field)) or 0.0

        cluster_ids = [anchor_i]
        taken[anchor_i] = True

        # присоединяем соседей
        for j in order:
            if taken[j]:
                continue
            pj = points[j]
            ej = _to_float(props_list[j].get(prob_field)) or 0.0
            if ej <= anchor_e + 1e-12 and anchor_p.distance(pj) <= radius_m:
                cluster_ids.append(j)
                taken[j] = True

        # агрегируем атрибуты
        floors_vals = [get_floors_value(props_list[k]) for k in cluster_ids]
        floors_vals = [v for v in floors_vals if v is not None]
        is_living_vals = [get_is_living_value(props_list[k]) for k in cluster_ids]
        is_living_vals = [v for v in is_living_vals if v is not None]
        living_area_vals = [get_living_area_value(props_list[k]) for k in cluster_ids]
        living_area_vals = [v for v in living_area_vals if v is not None]

        merged = dict(anchor_pr)  # e и прочие поля из якоря
        if floors_vals:
            mean_fl = float(sum(floors_vals)/len(floors_vals))
            merged["floors_avg"] = mean_fl
            merged["floors"] = int(math.ceil(mean_fl))
        if is_living_vals:
            merged["is_living"] = float(sum(is_living_vals)/len(is_living_vals))
        if living_area_vals:
            merged["living_area"] = float(sum(living_area_vals))

        # служебные поля слияния
        infer_ids = []
        for k in cluster_ids:
            fid = props_list[k].get("infer_fid")
            if fid is not None:
                try:
                    infer_ids.append(int(fid))
                except Exception:
                    pass
        infer_ids = sorted(set(infer_ids))
        merged["merged_from"] = infer_ids
        merged["merged_count"] = int(len(cluster_ids))

        # добавляем слитую точку
        m_points.append(anchor_p)
        m_props.append(merged)

    return m_points, m_props

# ---- Граф: построение ребер по радиусу ----

def build_gabriel_knn_edges(points: List[Point],
                            props_list: List[Dict[str,Any]],
                            knn_k: int,
                            block_index: int,
                            zone_label: str,
                            eps: float = 1e-9) -> List[Dict[str,Any]]:
    """
    Gabriel-graph с предварительным отбором кандидатов по kNN (k ближайших соседей).
    Кандидаты: для каждого i берём k ближайших j != i. Затем фильтруем по критерию Gabriel:
    диск с диаметром uv не должен содержать других точек.
    """
    edges: List[Dict[str,Any]] = []
    n = len(points)
    if n <= 1:
        return edges

    # Координаты в массив для быстрых дистанций
    coords = np.array([[p.x, p.y] for p in points], dtype=np.float64)

    # Соберём множество неориентированных кандидатных рёбер по kNN
    cand = set()
    for i in range(n):
        # расстояния от i до всех
        di = coords - coords[i]
        d2 = (di[:,0]**2 + di[:,1]**2)
        # исключаем самого себя
        d2[i] = np.inf
        # индексы k ближайших
        if knn_k >= n:
            nn_idx = np.argsort(d2)
        else:
            nn_idx = np.argpartition(d2, knn_k)[:knn_k]
        for j in nn_idx:
            a, b = (i, int(j)) if i < j else (int(j), i)
            cand.add((a, b))

    # Фильтрация по критерию Gabriel
    for (i, j) in cand:
        # середина и радиус (половина длины)
        mx = 0.5 * (coords[i,0] + coords[j,0])
        my = 0.5 * (coords[i,1] + coords[j,1])
        mid = Point(mx, my)
        dij = math.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
        r = 0.5 * dij

        if r <= eps:
            # точка-дубликат или почти совпадение — пропускаем ребро
            continue

        # Проверяем отсутствие других точек внутри диска диаметра (<= r)
        violates = False
        for k in range(n):
            if k == i or k == j:
                continue
            if mid.distance(points[k]) <= r - eps:
                violates = True
                break
        if violates:
            continue

        # Если прошёл — создаём ребро
        pi, pj = points[i], points[j]
        line = LineString([(pi.x, pi.y), (pj.x, pj.y)])
        length_m = float(pi.distance(pj))

        ui = (props_list[i].get("infer_fid"), props_list[i].get("_source_block_index", block_index))
        uj = (props_list[j].get("infer_fid"), props_list[j].get("_source_block_index", block_index))

        eprops = {
            "u": int(ui[0]) if ui[0] is not None else i,
            "v": int(uj[0]) if uj[0] is not None else j,
            "u_block": int(ui[1]),
            "v_block": int(uj[1]),
            "zone": zone_label,
            "length_m": length_m,
            "edge_type": "gabriel_knn",
            "knn_k": int(knn_k),
            "graph": "centroid_gabriel_v1",
            "block_index": int(block_index),
        }
        edges.append({"type":"Feature", "geometry": mapping(line), "properties": eprops})

    return edges

# ---- Основной скрипт ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer + centroid merge + graph (Shapely < 2.0)")
    ap.add_argument("--blocks", required=True, help="Входной GeoJSON (FeatureCollection) кварталов")
    ap.add_argument("--out", required=True, help="Выходной GeoJSON линий графа (LineString)")
    ap.add_argument("--out-infer-polys", default=None, help="Выход GeoJSON полигонов инференса (Polygon/MultiPolygon)")
    ap.add_argument("--out-infer-centroids", default=None, help="Выход GeoJSON СЛИТЫХ центроидов (Point)")
    ap.add_argument("--train-script", default="./train.py", help="Путь к train.py")
    ap.add_argument("--model-ckpt", required=True, help="Путь к чекпойнту модели (.pt)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None, help="cuda|cpu")
    ap.add_argument("--zone-attr", default="zone", help="Имя свойства зоны в кварталах")
    # targets совместимость
    ap.add_argument("--zones-json", default=None)
    ap.add_argument("--services-json", default=None)
    ap.add_argument("--targets-by-zone", default=None)
    ap.add_argument("--la-by-zone", default=None)
    ap.add_argument("--people", type=int, default=1000)
    ap.add_argument("--min-services", dest="min_services", action="store_true", default=True)
    ap.add_argument("--no-min-services", dest="min_services", action="store_false")
    # инференс
    ap.add_argument("--infer-slots", type=int, default=256)
    ap.add_argument("--infer-knn", type=int, default=8)
    ap.add_argument("--infer-e-thr", type=float, default=0.5)
    ap.add_argument("--infer-il-thr", type=float, default=0.5)
    ap.add_argument("--infer-sv1-thr", type=float, default=0.5)
    # слияние центроидов
    ap.add_argument("--merge-centroids-radius-m", type=float, default=10.0,
                    help="Радиус слияния центроидов внутри квартала (м)")
    ap.add_argument("--prob-field", default="e",
                    help="Имя поля вероятности для сортировки якорей (по умолчанию 'e')")
    # граф
    ap.add_argument("--graph-knn-k", type=int, default=4,
                    help="Число ближайших соседей для кандидатных рёбер в Gabriel-графе (по умолчанию 4)")

    args = ap.parse_args()

    stem = os.path.splitext(args.out)[0]
    out_infer_polys_path = args.out_infer_polys or (stem + "_infer_polys.geojson")
    out_infer_centroids_path = args.out_infer_centroids or (stem + "_infer_centroids.geojson")

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
    out_edge_feats: List[Dict[str,Any]] = []
    out_infer_poly_feats: List[Dict[str,Any]] = []
    out_infer_centroid_feats: List[Dict[str,Any]] = []
    infer_uid = 0  # общий идентификатор пар (полигон<->центроид до слияния)

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

            # Инференс квартала
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

            # Читаем результат квартала и вытаскиваем полигоны + центроиды (до слияния)
            try:
                blk_out = read_geojson(out_path)
            except Exception as e:
                print(f"[WARN] failed to read output for block #{bi}: {e}", file=sys.stderr)
                continue

            centroids_raw: List[Point] = []
            cprops_raw: List[Dict[str,Any]] = []

            for b in blk_out.get("features", []):
                try:
                    g = shape(b["geometry"])
                    raw_props = dict(b.get("properties") or {})
                    shared_props = dict(raw_props)
                    shared_props.setdefault(args.zone_attr, z)
                    shared_props.setdefault("_source_block_index", bi)

                    if isinstance(g, (Polygon, MultiPolygon)) and (not g.is_empty) and (g.area > 0):
                        infer_uid += 1
                        shared_props["infer_fid"] = infer_uid

                        # исходный полигон — как есть
                        out_infer_poly_feats.append({
                            "type":"Feature", "geometry": mapping(g), "properties": dict(shared_props)
                        })

                        # исходный центроид — идёт только во «внутренний» список для слияния
                        c = g.centroid
                        centroids_raw.append(c)
                        cprops_raw.append(shared_props)

                    else:
                        continue
                except Exception:
                    continue

            # Слияние центроидов внутри квартала
            if centroids_raw:
                m_points, m_props = merge_centroids(
                    points=centroids_raw,
                    props_list=cprops_raw,
                    radius_m=float(args.merge_centroids_radius_m),
                    prob_field=str(args.prob_field),
                )
                # пишем СЛИТЫЕ центроиды в выход
                for p, pr in zip(m_points, m_props):
                    out_infer_centroid_feats.append({
                        "type":"Feature", "geometry": mapping(p), "properties": pr
                    })
                # строим граф по слитым центроидам
                edges = build_gabriel_knn_edges(
                    points=m_points,
                    props_list=m_props,
                    knn_k=int(args.graph_knn_k),
                    block_index=bi,
                    zone_label=z,
                )
                out_edge_feats.extend(edges)

        # Запись результатов
        write_geojson(args.out, {"type":"FeatureCollection","features": out_edge_feats})
        write_geojson(out_infer_polys_path, {"type":"FeatureCollection","features": out_infer_poly_feats})
        write_geojson(out_infer_centroids_path, {"type":"FeatureCollection","features": out_infer_centroid_feats})

        print(f"[OK] graph edges:           {len(out_edge_feats)} → {args.out}")
        print(f"[OK] infer polygons:        {len(out_infer_poly_feats)} → {out_infer_polys_path}")
        print(f"[OK] merged infer centroids:{len(out_infer_centroid_feats)} → {out_infer_centroids_path}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())
