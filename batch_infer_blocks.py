#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_infer_blocks.py — пакетная генерация застройки через train.py --mode infer
с последующим слиянием центроидов внутри кварталов, переформатированием «прибрежной»
зоны квартала в сегменты второго кольца (10–30 м), и построением графа соседства.

Совместимо с Shapely < 2.0 (проверено на 1.7/1.8).
Зависимости: shapely, numpy, tqdm (опц.).

ВЫХОДЫ (3 шт):
  1) --out                 : граф как линии (LineString) между СЛИТЫМИ (и при необходимости
                             сдвинутыми) центроидами зданий (Gabriel-критерий по kNN).
  2) --out-infer-polys     : ПОЛИГОНЫ-СЕГМЕНТЫ ВТОРОГО КОЛЬЦА (а НЕ полигоны модели).
                             Строятся по срединной линии кольца (20 м) с длиной дуги
                             ∝ весу точки; затем бафер 10 м и клиппинг кольцом; после — укорачиваем
                             по дуге на 5 м с каждого конца (если возможно).
  3) --out-infer-centroids : СЛИТЫЕ центроиды (Point) ПОСЛЕ сдвига на срединную линию
                             (для тех, кто был в кольцах). Свойства агрегированы:
       - e            : из якоря кластера (наиболее вероятного)
       - floors       : ceil(среднего по кластеру)  (дополнительно floors_avg — само среднее)
       - is_living    : среднее (0..1)
       - living_area  : сумма
       - merged_from  : список infer_fid, вошедших в кластер
       - merged_count : размер кластера
       - ring_snap    : none|to_ring2_midline (маркер сдвига)
       - ring_zone    : first|second|none (где исходно находился центроид)

Примечания:
- Расстояния считаются в единицах CRS входа — используйте метрическую проекцию.
- Второе кольцо: 10–30 м внутрь от внешней границы квартала; срединная линия — 20 м.
- Если второе кольцо пусто (слишком узкий квартал), сегменты не строятся.
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
    from shapely.geometry import shape, mapping, Point, Polygon, MultiPolygon, LineString, MultiLineString
    from shapely.ops import nearest_points
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

# ---- Гео-хелперы для колец и линий ----

def polygon_or_multi(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return []

def lines_of(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    # boundary может вернуть LinearRing — он является LineString в геометриях Shapely
    return []

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

def line_substring(line: LineString, start_d: float, end_d: float) -> Optional[LineString]:
    """Вырезает подотрезок линии по длинам вдоль геометрии (без shapely.ops.substring)."""
    L = float(line.length)
    if L <= 0: return None
    a = max(0.0, min(L, float(start_d)))
    b = max(0.0, min(L, float(end_d)))
    if b - a <= 1e-9: return None

    coords = list(line.coords)
    acc = 0.0
    out_pts = []

    def interp(p1, p2, t):
        return (p1[0] + (p2[0]-p1[0])*t, p1[1] + (p2[1]-p1[1])*t)

    for i in range(len(coords)-1):
        p1 = coords[i]; p2 = coords[i+1]
        seg_len = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if seg_len <= 0:
            continue
        seg_start = acc
        seg_end = acc + seg_len

        # сегмент полностью перед нужным диапазоном
        if seg_end < a - 1e-12:
            acc = seg_end; continue
        # сегмент полностью после нужного диапазона
        if seg_start > b + 1e-12:
            break

        # пересечение с [a,b]
        s = max(a, seg_start)
        e = min(b, seg_end)
        if e - s <= 1e-9:
            acc = seg_end; continue

        # добавляем точки от s до e
        t0 = (s - seg_start) / seg_len
        t1 = (e - seg_start) / seg_len
        pt0 = interp(p1, p2, t0)
        pt1 = interp(p1, p2, t1)

        if not out_pts:
            out_pts.append(pt0)
        else:
            # стыковка с предыдущим концом
            if out_pts[-1] != pt0:
                out_pts.append(pt0)
        out_pts.append(pt1)
        acc = seg_end

    if len(out_pts) >= 2:
        try:
            return LineString(out_pts)
        except Exception:
            return None
    return None
# ------ вычисление сетки ---------
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
        anchor_i = idx
        anchor_p = points[anchor_i]
        anchor_pr = props_list[anchor_i]
        anchor_e = _to_float(anchor_pr.get(prob_field)) or 0.0

        cluster_ids = [anchor_i]
        taken[anchor_i] = True

        for j in order:
            if taken[j]:
                continue
            pj = points[j]
            ej = _to_float(props_list[j].get(prob_field)) or 0.0
            if ej <= anchor_e + 1e-12 and anchor_p.distance(pj) <= radius_m:
                cluster_ids.append(j)
                taken[j] = True

        # агрегаты
        floors_vals = [get_floors_value(props_list[k]) for k in cluster_ids]
        floors_vals = [v for v in floors_vals if v is not None]
        is_living_vals = [get_is_living_value(props_list[k]) for k in cluster_ids]
        is_living_vals = [v for v in is_living_vals if v is not None]
        living_area_vals = [get_living_area_value(props_list[k]) for k in cluster_ids]
        living_area_vals = [v for v in living_area_vals if v is not None]

        merged = dict(anchor_pr)  # переносим e и др.
        if floors_vals:
            mean_fl = float(sum(floors_vals)/len(floors_vals))
            merged["floors_avg"] = mean_fl
            merged["floors"] = int(math.ceil(mean_fl))
        if is_living_vals:
            merged["is_living"] = float(sum(is_living_vals)/len(is_living_vals))
        if living_area_vals:
            merged["living_area"] = float(sum(living_area_vals))

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

        m_points.append(anchor_p)
        m_props.append(merged)

    return m_points, m_props

# ---- Граф: Gabriel с kNN-кандидатами ----

def build_gabriel_knn_edges(points: List[Point],
                            props_list: List[Dict[str,Any]],
                            knn_k: int,
                            block_index: int,
                            zone_label: str,
                            eps: float = 1e-9) -> List[Dict[str,Any]]:
    edges: List[Dict[str,Any]] = []
    n = len(points)
    if n <= 1:
        return edges

    coords = np.array([[p.x, p.y] for p in points], dtype=np.float64)
    cand = set()
    for i in range(n):
        di = coords - coords[i]
        d2 = (di[:,0]**2 + di[:,1]**2)
        d2[i] = np.inf
        if knn_k >= n:
            nn_idx = np.argsort(d2)
        else:
            nn_idx = np.argpartition(d2, knn_k)[:knn_k]
        for j in nn_idx:
            a, b = (i, int(j)) if i < j else (int(j), i)
            cand.add((a, b))

    for (i, j) in cand:
        mx = 0.5 * (coords[i,0] + coords[j,0])
        my = 0.5 * (coords[i,1] + coords[j,1])
        mid = Point(mx, my)
        dij = math.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
        r = 0.5 * dij
        if r <= eps:
            continue
        violates = False
        for k in range(n):
            if k == i or k == j:
                continue
            if mid.distance(points[k]) <= r - eps:
                violates = True
                break
        if violates:
            continue
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

# ---- Гекс-сетка (pointy-top, сторона side_m) ----

def _hex_polygon(cx: float, cy: float, side_m: float) -> Polygon:
    """Единичный гекс (pointy-top). Углы: 30°, 90°, 150°, 210°, 270°, 330°."""
    pts = []
    for i in range(6):
        ang = math.radians(60 * i + 30.0)
        pts.append((cx + side_m * math.cos(ang), cy + side_m * math.sin(ang)))
    pts.append(pts[0])
    return Polygon(pts)

def hex_grid_covering_polygon(block_geom, side_m: float) -> List[Polygon]:
    """
    Строит pointy-top гекс-сетку с длиной стороны side_m, покрывающую bbox квартала,
    и возвращает ТОЛЬКО те гексы, которые целиком лежат внутри квартала (hex.within(block)).
    """
    if block_geom is None or block_geom.is_empty:
        return []

    # Геометрические параметры pointy-top
    s = float(side_m)
    width = math.sqrt(3.0) * s      # расстояние между центрами по X
    height = 2.0 * s                # полная высота гекса
    dx = width                      # шаг по X
    dy = 1.5 * s                    # шаг по Y
    x_off_odd = width / 2.0         # сдвиг "нечётных" рядов

    minx, miny, maxx, maxy = block_geom.bounds
    # небольшой запас, чтобы «закрыть» край
    pad = max(width, height)
    minx -= pad; miny -= pad; maxx += pad; maxy += pad

    hexes: List[Polygon] = []
    row = 0
    y = miny
    while y <= maxy + 1e-9:
        x0 = minx + (x_off_odd if (row % 2) else 0.0)
        x = x0
        while x <= maxx + 1e-9:
            h = _hex_polygon(x, y, s)
            # берём ТОЛЬКО целиком лежащие в квартале гексы
            if h.within(block_geom):
                hexes.append(h)
            x += dx
        row += 1
        y += dy
    return hexes

def build_hex_grid_features(block_geom,
                            moved_points: List[Point],
                            moved_props:  List[Dict[str,Any]],
                            side_m: float,
                            block_index: int,
                            zone_label: str) -> List[Dict[str,Any]]:
    """
    Строит гекс-сетку по кварталу и агрегирует показатели по попавшим в гекс сдвинутым центроидам:
      - pts_count           : количество точек
      - e_max               : максимальное e
      - floors_avg          : среднее по floors/floors_avg
      - is_living_avg       : среднее is_living
      - living_area_sum     : сумма living_area
    """
    hexes = hex_grid_covering_polygon(block_geom, side_m)
    feats: List[Dict[str,Any]] = []

    # Подготовим числовые векторы свойств
    e_vals = [(_to_float(pr.get("e")) if _to_float(pr.get("e")) is not None else None) for pr in moved_props]
    fl_vals = [get_floors_value(pr) for pr in moved_props]
    il_vals = [get_is_living_value(pr) for pr in moved_props]
    la_vals = [get_living_area_value(pr) for pr in moved_props]

    # Простой O(N*M) проход (для больших N, M можно заменить на STRtree)
    for hid, hex_poly in enumerate(hexes):
        # включаем точки "на границе" — используем covers
        idxs = [i for i, p in enumerate(moved_points) if hex_poly.covers(p)]

        cnt = len(idxs)
        if cnt > 0:
            e_max = max([e_vals[i] for i in idxs if e_vals[i] is not None], default=None)
            fl = [fl_vals[i] for i in idxs if fl_vals[i] is not None]
            il = [il_vals[i] for i in idxs if il_vals[i] is not None]
            la = [la_vals[i] for i in idxs if la_vals[i] is not None]

            floors_avg = (sum(fl)/len(fl)) if fl else None
            is_living_avg = (sum(il)/len(il)) if il else None
            living_area_sum = float(sum(la)) if la else 0.0
        else:
            e_max = None
            floors_avg = None
            is_living_avg = None
            living_area_sum = 0.0

        props = {
            "hex_id": int(hid),
            "block_index": int(block_index),
            "zone": zone_label,
            "side_m": float(side_m),
            "pts_count": int(cnt),
            "e_max": (float(e_max) if e_max is not None else None),
            "floors_avg": (float(floors_avg) if floors_avg is not None else None),
            "is_living_avg": (float(is_living_avg) if is_living_avg is not None else None),
            "living_area_sum": float(living_area_sum),
        }
        feats.append({"type":"Feature", "geometry": mapping(hex_poly), "properties": props})

    return feats

# ---- Основной скрипт ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer + centroid merge + ring segments + graph (Shapely < 2.0)")
    ap.add_argument("--blocks", required=True, help="Входной GeoJSON (FeatureCollection) кварталов")
    ap.add_argument("--out", required=True, help="Выходной GeoJSON линий графа (LineString)")
    ap.add_argument("--out-infer-polys", default=None, help="Выход GeoJSON ПОЛИГОНОВ-СЕГМЕНТОВ второго кольца")
    ap.add_argument("--out-infer-centroids", default=None, help="Выход GeoJSON СЛИТЫХ (и сдвинутых) центроидов (Point)")
    ap.add_argument("--out-hex-grid", default=None,
                help="Выход GeoJSON гекс-сетки по кварталам (Polygon)")
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
    ap.add_argument("--graph-knn-k", type=int, default=6,
                    help="Число ближайших соседей для кандидатных рёбер в Gabriel-графе (по умолчанию 6)")
    # сетка
    ap.add_argument("--hex-side-m", type=float, default=5.0,
                    help="Длина стороны гекса в метрах (по умолчанию 5.0)")
    # crs
    ap.add_argument("--out-epsg", type=int, default=32636,
                    help="EPSG-код для записи всех выходных GeoJSON (по умолчанию 32636)")
    

    args = ap.parse_args()

    stem = os.path.splitext(args.out)[0]
    out_hex_grid_path = args.out_hex_grid or (stem + "_hex_grid.geojson")
    out_infer_polys_path = args.out_infer_polys or (stem + "_ring_segments.geojson")
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
    out_ring_segment_feats: List[Dict[str,Any]] = []
    out_infer_centroid_feats: List[Dict[str,Any]] = []
    out_hex_feats: List[Dict[str,Any]] = []
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

            slots_side = infer_slots_from_block_bbox(block_geom)

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
                        # исходный центроид — идёт только во «внутренний» список для слияния
                        c = g.centroid
                        centroids_raw.append(c)
                        cprops_raw.append(shared_props)
                    else:
                        continue
                except Exception:
                    continue

            # Слияние центроидов внутри квартала
            if not centroids_raw:
                continue

            m_points, m_props = merge_centroids(
                points=centroids_raw,
                props_list=cprops_raw,
                radius_m=float(args.merge_centroids_radius_m),
                prob_field=str(args.prob_field),
            )

            # Построение колец и срединной линии 20 м
            ring1, ring2, midline = build_rings(block_geom)
            mid_lines = lines_of(midline)
            ring2_polys = polygon_or_multi(ring2)

            # Сдвиг центроидов: те, что в 0–10 м и 10–30 м — на срединную линию (20 м)
            moved_points: List[Point] = []
            moved_props:  List[Dict[str,Any]] = []
            ring_assigned_for_segments = []  # индексы точек, попавших в кольцо 2 (после сдвига)
            for p, pr in zip(m_points, m_props):
                status = "none"
                q = p
                if (ring2 and not ring2.is_empty and mid_lines):
                    in_r1 = ring1.contains(p) or ring1.touches(p)
                    in_r2 = ring2.contains(p) or ring2.touches(p)
                    if in_r1 or in_r2:
                        status = "first" if in_r1 and not in_r2 else "second"
                        li, q_on, _m = nearest_on_lines(mid_lines, p)
                        if li >= 0:
                            q = q_on
                            ring_assigned_for_segments.append((len(moved_points), li))  # (idx в moved_points, линия)
                        pr = dict(pr)
                        pr["ring_snap"] = "to_ring2_midline"
                        pr["ring_zone"] = status
                # точки вне колец оставляем
                if "ring_snap" not in pr:
                    pr = dict(pr)
                    pr["ring_snap"] = "none"
                    pr["ring_zone"] = status
                moved_points.append(q)
                moved_props.append(pr)

            # Собираем точки на срединной линии для распределения сегментов
            # Берём только те, что были в 0–10 или 10–30 (status != none) и у которых есть линия
            pts_for_segments_idx = [i for (i, li) in ring_assigned_for_segments]
            lines_map = {}
            for (i, li) in ring_assigned_for_segments:
                lines_map.setdefault(li, []).append(i)

            # Подсчёт LA_sum по всем точкам кольца (для формулы веса)
            la_vals = []
            for idx in pts_for_segments_idx:
                la = get_living_area_value(moved_props[idx])
                la_vals.append(la if (la is not None and la > 0) else 0.0)
            LA_sum = float(sum(la_vals)) if la_vals else 0.0

            # Формируем сегменты на каждой компоненте срединной линии
            for li, idxs in lines_map.items():
                line = mid_lines[li]
                if not line or line.length <= 1e-6:
                    continue
                # Сортируем точки по параметру вдоль линии
                items = []
                for idx in idxs:
                    m = line.project(moved_points[idx])
                    la = get_living_area_value(moved_props[idx])
                    if la is not None and la > 0 and LA_sum > 0:
                        w = 1.0 + (float(la) / LA_sum)
                    else:
                        w = 1.0
                    items.append((m, idx, w))
                items.sort(key=lambda t: t[0])

                W = sum(w for _m, _idx, w in items)
                if W <= 0:
                    continue

                L = float(line.length)
                cursor = 0.0
                order_local = 0
                for m_along, idx, w in items:
                    share = float(w) / W
                    seg_len_raw = L * share

                    start = cursor
                    end = cursor + seg_len_raw
                    cursor = end

                    # Укорачиваем по 5 м с каждого края
                    shrink = 5.0
                    start_sh = start + shrink
                    end_sh = end - shrink
                    if end_sh - start_sh <= 1e-6:
                        # слишком короткий сегмент — пропустим
                        order_local += 1
                        continue

                    seg_line = line_substring(line, start_sh, end_sh)
                    if seg_line is None or seg_line.length <= 1e-6:
                        order_local += 1
                        continue

                    # Бафер 10 м (толщина полосы 20 м), потом клиппим вторым кольцом
                    seg_poly_raw = seg_line.buffer(10.0, cap_style=2, join_style=2)
                    # кольцо может быть мультиполигоном; пересечём со всем
                    seg_poly = None
                    for rp in ring2_polys or []:
                        inter = seg_poly_raw.intersection(rp)
                        if inter and not inter.is_empty:
                            seg_poly = inter if seg_poly is None else seg_poly.union(inter)
                    if (ring2 and not ring2.is_empty) and seg_poly is None:
                        # как fallback: пересечение с ring2 целиком
                        inter = seg_poly_raw.intersection(ring2)
                        if inter and not inter.is_empty:
                            seg_poly = inter

                    if seg_poly is None or seg_poly.is_empty:
                        order_local += 1
                        continue

                    # Свойства сегмента — от соответствующей точки + служебные
                    pr = dict(moved_props[idx])
                    pr.update({
                        "ring": "second",
                        "ring_inner_m": 10.0,
                        "ring_outer_m": 30.0,
                        "seg_line_len_raw_m": float(seg_len_raw),
                        "seg_line_len_shrunk_m": float(end_sh - start_sh),
                        "seg_weight": float(w),
                        "seg_weight_norm_line": float(w / W),
                        "seg_order_on_line": int(order_local),
                        "seg_line_component": int(li),
                        "block_index": int(bi),
                        "zone": z,
                    })

                    out_ring_segment_feats.append({
                        "type":"Feature",
                        "geometry": mapping(seg_poly),
                        "properties": pr
                    })
                    order_local += 1

            # Записываем СЛИТЫЕ и (возможные) СДВИНУТЫЕ центроиды
            for p, pr in zip(moved_points, moved_props):
                out_infer_centroid_feats.append({
                    "type":"Feature", "geometry": mapping(p), "properties": pr
                })

            # строим граф по сдвинутым центроидам
            edges = build_gabriel_knn_edges(
                points=moved_points,
                props_list=moved_props,
                knn_k=int(args.graph_knn_k),
                block_index=bi,
                zone_label=z,
            )
            out_edge_feats.extend(edges)

            hex_feats = build_hex_grid_features(
                block_geom=block_geom,
                moved_points=moved_points,
                moved_props=moved_props,
                side_m=float(args.hex_side_m),
                block_index=bi,
                zone_label=z,
            )
            out_hex_feats.extend(hex_feats)

        # Запись результатов
        write_geojson(args.out, {"type":"FeatureCollection","features": out_edge_feats}, epsg=args.out_epsg)
        write_geojson(out_infer_polys_path, {"type":"FeatureCollection","features": out_ring_segment_feats}, epsg=args.out_epsg)
        write_geojson(out_infer_centroids_path, {"type":"FeatureCollection","features": out_infer_centroid_feats}, epsg=args.out_epsg)
        write_geojson(out_hex_grid_path, {"type":"FeatureCollection","features": out_hex_feats}, epsg=args.out_epsg)

        print(f"[OK] hex grid cells:         {len(out_hex_feats)} → {out_hex_grid_path}")
        print(f"[OK] graph edges:            {len(out_edge_feats)} → {args.out}")
        print(f"[OK] ring2 segments:         {len(out_ring_segment_feats)} → {out_infer_polys_path}")
        print(f"[OK] merged+snapped points:  {len(out_infer_centroid_feats)} → {out_infer_centroids_path}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())
