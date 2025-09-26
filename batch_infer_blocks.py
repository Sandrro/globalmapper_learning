#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_infer_blocks.py — пакетная генерация застройки через train.py --mode infer
с последующим слиянием центроидов внутри кварталов, возможным смещением точек на
срединную линию буфера квартала и построением гекс-сетки зданий. После фильтрации
гексов (≥2 точек) формируется граф соседства, упрощается до одной основной линии с
ветвями, затем по ним строятся прямоугольные буферы и полигоны зданий с агрегацией
атрибутов.

Совместимо с Shapely < 2.0 (проверено на 1.7/1.8).
Зависимости: shapely, tqdm (опц.).

ВЫХОДЫ (4 шт):
  1) --out                 : линии графа зданий по гексам (LineString) — основная линия и ветви.
  2) --out-infer-polys     : полигоны зданий, построенные из буферов линий (Polygon).
  3) --out-infer-centroids : исходные (слитые и, при необходимости, смещённые) точки зданий (Point).
  4) --out-hex-grid        : гексы с агрегированными показателями по точкам (Polygon).

В полигоны зданий записываются:
  - e_max             : максимальная вероятность e среди точек;
  - living_area_sum   : суммарная жилая площадь;
  - is_living_max     : максимальное значение is_living;
  - floors_avg_round  : средняя этажность, округлённая до целого;
  - pts_count         : количество точек в объекте.

Примечания:
- Расстояния считаются в единицах CRS входа — используйте метрическую проекцию.
- Второе кольцо: 10–30 м внутрь от внешней границы квартала; срединная линия — 20 м.
- Если второе кольцо пусто (слишком узкий квартал), смещение происходит только для тех,
  кто попал в доступные линии.
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
    from shapely.ops import nearest_points, unary_union
except Exception:
    print("[FATAL] Требуется shapely и её зависимости (GEOS).", file=sys.stderr)
    raise

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

def hex_characteristic_width(hex_poly: Polygon) -> float:
    minx, miny, maxx, maxy = hex_poly.bounds
    width = min(maxx - minx, maxy - miny)
    area = hex_poly.area
    if width <= 0 and area > 0:
        width = math.sqrt(area)
    return float(max(width, 1e-6))


def build_hex_nodes(block_geom,
                    moved_points: List[Point],
                    moved_props: List[Dict[str, Any]],
                    side_m: float,
                    block_index: int,
                    zone_label: str) -> List[Dict[str, Any]]:
    hexes = hex_grid_covering_polygon(block_geom, side_m)
    nodes: List[Dict[str, Any]] = []

    for hex_poly in hexes:
        idxs = [i for i, p in enumerate(moved_points) if hex_poly.covers(p)]
        if len(idxs) <= 1:
            continue

        e_vals = [_to_float(moved_props[i].get("e")) for i in idxs]
        e_vals = [v for v in e_vals if v is not None]

        floors_vals = [get_floors_value(moved_props[i]) for i in idxs]
        floors_vals = [v for v in floors_vals if v is not None]

        is_living_vals = [get_is_living_value(moved_props[i]) for i in idxs]
        is_living_vals = [v for v in is_living_vals if v is not None]

        living_area_vals = [get_living_area_value(moved_props[i]) for i in idxs]
        living_area_vals = [v for v in living_area_vals if v is not None]

        props = {
            "block_index": int(block_index),
            "zone": zone_label,
            "side_m": float(side_m),
            "pts_count": int(len(idxs)),
            "e_max": (max(e_vals) if e_vals else None),
            "floors_avg": (sum(floors_vals) / len(floors_vals) if floors_vals else None),
            "is_living_avg": (sum(is_living_vals) / len(is_living_vals) if is_living_vals else None),
            "living_area_sum": float(sum(living_area_vals)) if living_area_vals else 0.0,
        }

        nodes.append({
            "polygon": hex_poly,
            "centroid": hex_poly.centroid,
            "width": hex_characteristic_width(hex_poly),
            "point_indices": idxs,
            "props": props,
        })

    return nodes


def build_hex_neighbor_edges(nodes: List[Dict[str, Any]]) -> List[Tuple[int, int, float]]:
    edges: List[Tuple[int, int, float]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a = nodes[i]["polygon"]
            b = nodes[j]["polygon"]
            try:
                touches = a.touches(b) or a.intersection(b).length > 0.0
            except Exception:
                touches = a.touches(b)
            if touches:
                w = nodes[i]["centroid"].distance(nodes[j]["centroid"])
                edges.append((i, j, float(w)))
    return edges


def _components_from_edges(n_nodes: int, edges: List[Tuple[int, int, float]]) -> Tuple[List[List[int]], Dict[int, List[Tuple[int, float]]]]:
    adjacency: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n_nodes)}
    for u, v, w in edges:
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))

    visited = [False] * n_nodes
    components: List[List[int]] = []
    for i in range(n_nodes):
        if visited[i]:
            continue
        stack = [i]
        comp: List[int] = []
        while stack:
            cur = stack.pop()
            if visited[cur]:
                continue
            visited[cur] = True
            comp.append(cur)
            for nb, _w in adjacency[cur]:
                if not visited[nb]:
                    stack.append(nb)
        components.append(comp)
    return components, adjacency


def _prim_mst(component_nodes: List[int], adjacency: Dict[int, List[Tuple[int, float]]]) -> List[Tuple[int, int, float]]:
    if len(component_nodes) <= 1:
        return []

    start = component_nodes[0]
    visited = {start}
    edges_out: List[Tuple[int, int, float]] = []

    import heapq

    heap: List[Tuple[float, int, int]] = []
    for nb, w in adjacency[start]:
        if nb in visited:
            continue
        heapq.heappush(heap, (w, start, nb))

    component_set = set(component_nodes)
    while heap and len(visited) < len(component_nodes):
        w, u, v = heapq.heappop(heap)
        if v in visited:
            continue
        visited.add(v)
        edges_out.append((u, v, w))
        for nb, w2 in adjacency[v]:
            if nb in visited or nb not in component_set:
                continue
            heapq.heappush(heap, (w2, v, nb))
    return edges_out


def _longest_path_in_tree(tree_edges: List[Tuple[int, int, float]]) -> List[int]:
    if not tree_edges:
        return []

    adjacency: Dict[int, List[int]] = {}
    weights: Dict[Tuple[int, int], float] = {}
    nodes_set = set()
    for u, v, w in tree_edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)
        weights[(u, v)] = w
        weights[(v, u)] = w
        nodes_set.add(u)
        nodes_set.add(v)

    def farthest(start: int) -> Tuple[int, Dict[int, Optional[int]], Dict[int, float]]:
        stack: List[Tuple[int, Optional[int]]] = [(start, None)]
        parent: Dict[int, Optional[int]] = {start: None}
        dist: Dict[int, float] = {start: 0.0}
        order: List[int] = []
        while stack:
            node, par = stack.pop()
            order.append(node)
            for nb in adjacency.get(node, []):
                if nb == par:
                    continue
                parent[nb] = node
                dist[nb] = dist[node] + weights.get((node, nb), 0.0)
                stack.append((nb, node))
        far_node = max(order, key=lambda n: dist.get(n, 0.0)) if order else start
        return far_node, parent, dist

    start = next(iter(nodes_set))
    a, _, _ = farthest(start)
    b, parent, _ = farthest(a)
    path: List[int] = []
    cur = b
    while cur is not None:
        path.append(cur)
        if cur == a:
            break
        cur = parent.get(cur)
    path.reverse()
    return path


def _branch_components(branch_edges: List[Tuple[int, int, float]]) -> List[Tuple[set[int], List[Tuple[int, int, float]]]]:
    if not branch_edges:
        return []
    adjacency: Dict[int, List[Tuple[int, float]]] = {}
    for u, v, w in branch_edges:
        adjacency.setdefault(u, []).append((v, w))
        adjacency.setdefault(v, []).append((u, w))

    visited: set[int] = set()
    comps: List[Tuple[set[int], List[Tuple[int, int, float]]]] = []
    for node in list(adjacency.keys()):
        if node in visited:
            continue
        stack = [node]
        comp_nodes: set[int] = set()
        comp_edges: List[Tuple[int, int, float]] = []
        seen_edges: set[Tuple[int, int]] = set()
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp_nodes.add(cur)
            for nb, w in adjacency.get(cur, []):
                ek = (min(cur, nb), max(cur, nb))
                if ek not in seen_edges:
                    seen_edges.add(ek)
                    comp_edges.append((cur, nb, w))
                if nb not in visited:
                    stack.append(nb)
        comps.append((comp_nodes, comp_edges))
    return comps


def _rectangle_for_segment(p0: Point, p1: Point, width: float) -> Optional[Polygon]:
    if width <= 0:
        return None
    dx = p1.x - p0.x
    dy = p1.y - p0.y
    length = math.hypot(dx, dy)
    if length <= 1e-9:
        return None
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    half_w = width / 2.0
    extend = half_w
    p0x = p0.x - ux * extend
    p0y = p0.y - uy * extend
    p1x = p1.x + ux * extend
    p1y = p1.y + uy * extend
    corners = [
        (p0x - px * half_w, p0y - py * half_w),
        (p0x + px * half_w, p0y + py * half_w),
        (p1x + px * half_w, p1y + py * half_w),
        (p1x - px * half_w, p1y - py * half_w),
        (p0x - px * half_w, p0y - py * half_w),
    ]
    poly = Polygon(corners)
    return poly if poly.is_valid else poly.buffer(0)


def _aggregate_points(point_indices: List[int], moved_props: List[Dict[str, Any]]) -> Dict[str, Any]:
    idx_set = sorted(set(point_indices))
    e_vals = []
    floors_vals = []
    is_living_vals = []
    living_area_vals = []
    for idx in idx_set:
        pr = moved_props[idx]
        v = _to_float(pr.get("e"))
        if v is not None:
            e_vals.append(v)
        fl = get_floors_value(pr)
        if fl is not None:
            floors_vals.append(fl)
        il = get_is_living_value(pr)
        if il is not None:
            is_living_vals.append(il)
        la = get_living_area_value(pr)
        if la is not None:
            living_area_vals.append(la)

    floors_avg_round = None
    if floors_vals:
        floors_avg_round = int(round(sum(floors_vals) / len(floors_vals)))

    return {
        "pts_count": len(idx_set),
        "e_max": (max(e_vals) if e_vals else None),
        "is_living_max": (max(is_living_vals) if is_living_vals else None),
        "living_area_sum": float(sum(living_area_vals)) if living_area_vals else 0.0,
        "floors_avg_round": floors_avg_round,
    }

# ---- Основной скрипт ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer + centroid merge + hex building graph (Shapely < 2.0)")
    ap.add_argument("--blocks", required=True, help="Входной GeoJSON (FeatureCollection) кварталов")
    ap.add_argument("--out", required=True, help="Выходной GeoJSON линий графа зданий (LineString)")
    ap.add_argument("--out-infer-polys", default=None, help="Выход GeoJSON полигонов зданий (Polygon)")
    ap.add_argument("--out-infer-centroids", default=None, help="Выход GeoJSON исходных точек зданий (Point)")
    ap.add_argument("--out-hex-grid", default=None,
                help="Выход GeoJSON гекс-сетки зданий (Polygon)")
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
    # сетка
    ap.add_argument("--hex-side-m", type=float, default=5.0,
                    help="Длина стороны гекса в метрах (по умолчанию 5.0)")
    # crs
    ap.add_argument("--out-epsg", type=int, default=32636,
                    help="EPSG-код для записи всех выходных GeoJSON (по умолчанию 32636)")
    

    args = ap.parse_args()

    stem = os.path.splitext(args.out)[0]
    out_hex_grid_path = args.out_hex_grid or (stem + "_hexes.geojson")
    out_buildings_path = args.out_infer_polys or (stem + "_buildings.geojson")
    out_points_path = args.out_infer_centroids or (stem + "_points.geojson")

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
    out_point_feats: List[Dict[str,Any]] = []
    out_hex_feats: List[Dict[str,Any]] = []
    out_graph_feats: List[Dict[str,Any]] = []
    out_building_feats: List[Dict[str,Any]] = []
    global_hex_id = 0
    global_component_id = 0
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

            # Сдвиг центроидов: те, что в 0–10 м и 10–30 м — на срединную линию (20 м)
            moved_points: List[Point] = []
            moved_props:  List[Dict[str,Any]] = []
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
                        pr = dict(pr)
                        pr["ring_snap"] = "to_ring2_midline"
                        pr["ring_zone"] = status
                if "ring_snap" not in pr:
                    pr = dict(pr)
                    pr["ring_snap"] = "none"
                    pr["ring_zone"] = status
                moved_points.append(q)
                moved_props.append(pr)

            # Записываем СЛИТЫЕ и (возможные) СДВИНУТЫЕ центроиды
            for p, pr in zip(moved_points, moved_props):
                out_point_feats.append({
                    "type": "Feature", "geometry": mapping(p), "properties": pr
                })

            hex_nodes = build_hex_nodes(
                block_geom=block_geom,
                moved_points=moved_points,
                moved_props=moved_props,
                side_m=float(args.hex_side_m),
                block_index=bi,
                zone_label=z,
            )

            if not hex_nodes:
                continue

            for node in hex_nodes:
                props = dict(node["props"])
                props["hex_id"] = int(global_hex_id)
                node["hex_id"] = int(global_hex_id)
                out_hex_feats.append({
                    "type": "Feature",
                    "geometry": mapping(node["polygon"]),
                    "properties": props,
                })
                global_hex_id += 1

            hex_edges = build_hex_neighbor_edges(hex_nodes)
            components, adjacency = _components_from_edges(len(hex_nodes), hex_edges)

            for comp in components:
                if not comp:
                    continue

                tree_edges = _prim_mst(comp, adjacency)
                if tree_edges:
                    main_path = _longest_path_in_tree(tree_edges)
                    if not main_path:
                        main_path = [comp[0]]
                else:
                    main_path = [comp[0]]

                main_edge_keys: set[Tuple[int, int]] = set()
                for i in range(len(main_path) - 1):
                    a = main_path[i]
                    b = main_path[i + 1]
                    main_edge_keys.add((min(a, b), max(a, b)))

                comp_widths = [hex_nodes[idx]["width"] for idx in comp if hex_nodes[idx]["width"] > 0]
                default_width = float(sum(comp_widths) / len(comp_widths)) if comp_widths else float(args.hex_side_m)

                main_widths = [hex_nodes[idx]["width"] for idx in main_path if hex_nodes[idx]["width"] > 0]
                main_width = float(sum(main_widths) / len(main_widths)) if main_widths else default_width

                branch_edges = [e for e in tree_edges if (min(e[0], e[1]), max(e[0], e[1])) not in main_edge_keys]
                branch_comps = _branch_components(branch_edges)

                edge_widths: Dict[Tuple[int, int], float] = {}
                for key in main_edge_keys:
                    edge_widths[key] = main_width

                for nodes_set, branch_edge_list in branch_comps:
                    b_widths = [hex_nodes[idx]["width"] for idx in nodes_set if hex_nodes[idx]["width"] > 0]
                    branch_width = float(sum(b_widths) / len(b_widths)) if b_widths else default_width
                    for u, v, _w in branch_edge_list:
                        key = (min(u, v), max(u, v))
                        edge_widths[key] = branch_width

                rectangles: List[Polygon] = []

                for u, v, _w in tree_edges:
                    key = (min(u, v), max(u, v))
                    width = edge_widths.get(key, default_width)
                    rect = _rectangle_for_segment(hex_nodes[u]["centroid"], hex_nodes[v]["centroid"], width)
                    if rect and not rect.is_empty:
                        rectangles.append(rect)
                    line_props = {
                        "component_id": int(global_component_id),
                        "block_index": int(bi),
                        "zone": z,
                        "line_type": "main" if key in main_edge_keys else "branch",
                        "buffer_width_m": float(width),
                        "hex_u": int(hex_nodes[u]["hex_id"]),
                        "hex_v": int(hex_nodes[v]["hex_id"]),
                    }
                    out_graph_feats.append({
                        "type": "Feature",
                        "geometry": mapping(LineString([hex_nodes[u]["centroid"], hex_nodes[v]["centroid"]])),
                        "properties": line_props,
                    })

                if rectangles:
                    building_geom = unary_union(rectangles)
                else:
                    building_geom = unary_union([hex_nodes[idx]["polygon"] for idx in comp])

                if building_geom is None or building_geom.is_empty:
                    global_component_id += 1
                    continue

                point_indices: List[int] = []
                for idx in comp:
                    point_indices.extend(hex_nodes[idx]["point_indices"])

                agg = _aggregate_points(point_indices, moved_props)
                poly_props = {
                    "component_id": int(global_component_id),
                    "block_index": int(bi),
                    "zone": z,
                    "pts_count": int(agg["pts_count"]),
                    "e_max": (float(agg["e_max"]) if agg["e_max"] is not None else None),
                    "living_area_sum": float(agg["living_area_sum"]),
                    "is_living_max": (float(agg["is_living_max"]) if agg["is_living_max"] is not None else None),
                    "floors_avg_round": (int(agg["floors_avg_round"]) if agg["floors_avg_round"] is not None else None),
                }

                out_building_feats.append({
                    "type": "Feature",
                    "geometry": mapping(building_geom),
                    "properties": poly_props,
                })

                global_component_id += 1

        # Запись результатов
        write_geojson(args.out, {"type": "FeatureCollection", "features": out_graph_feats}, epsg=args.out_epsg)
        write_geojson(out_buildings_path, {"type": "FeatureCollection", "features": out_building_feats}, epsg=args.out_epsg)
        write_geojson(out_points_path, {"type": "FeatureCollection", "features": out_point_feats}, epsg=args.out_epsg)
        write_geojson(out_hex_grid_path, {"type": "FeatureCollection", "features": out_hex_feats}, epsg=args.out_epsg)

        print(f"[OK] hex cells:              {len(out_hex_feats)} → {out_hex_grid_path}")
        print(f"[OK] building graph lines:   {len(out_graph_feats)} → {args.out}")
        print(f"[OK] building polygons:      {len(out_building_feats)} → {out_buildings_path}")
        print(f"[OK] points:                 {len(out_point_feats)} → {out_points_path}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    sys.exit(main())
