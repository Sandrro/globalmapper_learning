#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transform.py
Иерархическая канонизация → HF parquet.
Быстрый assign зданий к кварталам:
  - Shapely ≥2: STRtree.query_bulk + векторный contains
  - Shapely 1.x: STRtree.query (батчи) + prepared.contains, затем fallback по poly∩block (max area)
Подробный logger и параллельная обработка.

Выход:
  - blocks.parquet, branches.parquet, nodes_fixed.parquet, edges.parquet
  - masks/{block_id}.png
"""

import os, sys, math, time, logging, argparse, multiprocessing as mp
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.strtree import STRtree
from PIL import Image, ImageDraw
import networkx as nx
import json

# --- tqdm (optional) ---
try:
    from tqdm.auto import tqdm
    TQDM = True
except Exception:
    TQDM = False
    def tqdm(x, **k): return x

# ----------------- LOGGER -----------------
def setup_logger(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )

log = logging.getLogger("transform")

# ----------------- I/O -----------------
def load_vector(path: str) -> gpd.GeoDataFrame:
    if path.lower().endswith(".parquet"):
        log.info(f"Читаю Parquet: {path}")
        return gpd.read_parquet(path)
    log.info(f"Читаю векторный файл: {path}")
    return gpd.read_file(path)

# --- Shapely 1.x/2.x совместимость для WKB ---
def geom_to_wkb(geom) -> bytes:
    try:
        return shapely.to_wkb(geom)  # Shapely 2
    except Exception:
        from shapely import wkb
        return wkb.dumps(geom)

def geom_from_wkb(buf: bytes):
    try:
        return shapely.from_wkb(buf)  # Shapely 2
    except Exception:
        from shapely import wkb
        return wkb.loads(buf)

# ----------------- GEO UTILS -----------------
def ensure_valid_series(geos: gpd.GeoSeries) -> gpd.GeoSeries:
    try:
        return shapely.make_valid(geos)  # Shapely 2
    except Exception:
        return geos.buffer(0)  # fallback

def normalize_crs_str(s: str) -> str:
    s = str(s)
    return s if s.upper().startswith("EPSG:") else f"EPSG:{s}"

def to_metric_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("У слоя отсутствует CRS.")
    if str(gdf.crs) != target_crs:
        return gdf.to_crs(target_crs)
    return gdf

def _dist(a,b): return math.hypot(b[0]-a[0], b[1]-a[1])

def get_size_with_vector(poly: Polygon):
    mrr = poly.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])
    if axis1 <= axis2:
        return axis2, axis1, np.array(bbox[0])-np.array(bbox[1]), np.array(bbox[0])-np.array(bbox[3])
    else:
        return axis1, axis2, np.array(bbox[0])-np.array(bbox[3]), np.array(bbox[0])-np.array(bbox[1])

def get_extend_line(a: Point, b: Point, block: Polygon, isfront: bool, is_extend_from_end=False):
    minx, miny, maxx, maxy = block.bounds
    if a.x == b.x: tgt = Point(a.x, miny if a.y <= b.y else maxy)
    elif a.y == b.y: tgt = Point(minx if a.x <= b.x else maxx, a.y)
    else:
        k = (b.y - a.y) / (b.x - a.x); m = a.y - k*a.x
        cand = [Point(x, k*x+m) for x in (minx, maxx)]
        cand += [Point((y-m)/k, y) for y in (miny, maxy)]
        cand.sort(key=lambda p: a.distance(p)); tgt = cand[0]
    extended = LineString([a, tgt])
    inter = block.boundary.intersection(extended)
    if inter.is_empty:
        near = b if is_extend_from_end else a
    elif inter.geom_type == "Point":
        near = inter
    else:
        pts = [g for g in getattr(inter, "geoms", []) if g.geom_type=="Point"]
        near = min(pts, key=lambda p: p.distance(a)) if pts else (b if is_extend_from_end else a)
    return LineString([near, a] if (isfront and not is_extend_from_end) else
                      [a, near] if (not isfront and not is_extend_from_end) else
                      [near, b] if (isfront and is_extend_from_end) else
                      [b, near])

def get_block_width_from_pt_on_axis(block: Polygon, vec_axis: np.ndarray, pt_on_axis: Point) -> float:
    unit_v = vec_axis / (np.linalg.norm(vec_axis)+1e-9)
    left_v = np.array([-unit_v[1], unit_v[0]])
    right_v= np.array([ unit_v[1],-unit_v[0]])
    dl = Point(pt_on_axis.x + left_v[0],  pt_on_axis.y + left_v[1])
    dr = Point(pt_on_axis.x + right_v[0], pt_on_axis.y + right_v[1])
    L = get_extend_line(dl, pt_on_axis, block, False, is_extend_from_end=True)
    R = get_extend_line(dr, pt_on_axis, block, False, is_extend_from_end=True)
    return L.length + R.length

def _largest_polygon(geom):
    """Вернуть крупнейший Polygon из геометрии (или None)."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        geoms = [g for g in geom.geoms if (g is not None and not g.is_empty)]
        return max(geoms, key=lambda g: g.area) if geoms else None
    # на всякий случай
    try:
        hull = geom.convex_hull
        return hull if isinstance(hull, Polygon) else None
    except Exception:
        return None

# ----------------- SKELETON / BRANCHES -----------------
def polygon_to_skgeom(poly: Polygon):
    import skgeom as sg
    ext = list(poly.exterior.coords)[:-1]; ext.reverse()
    return sg.Polygon(ext)

def build_straight_skeleton(poly: Polygon):
    import skgeom as sg
    skpoly = polygon_to_skgeom(poly)
    return sg.skeleton.create_interior_straight_skeleton(skpoly)

def skeleton_to_graph(skel):
    G = nx.Graph()
    for v in skel.vertices:
        G.add_node(v.id, x=float(v.point.x()), y=float(v.point.y()), time=float(v.time))
    for h in skel.halfedges:
        if not h.is_bisector: continue
        u, v = h.vertex.id, h.opposite.vertex.id
        if u==v: continue
        p1, p2 = h.vertex.point, h.opposite.vertex.point
        w = math.hypot(float(p1.x())-float(p2.x()), float(p1.y())-float(p2.y()))
        G.add_edge(u,v,weight=w)
    return G

def extract_branches(G: nx.Graph) -> List[LineString]:
    branches, used = [], set()
    degs = dict(G.degree())
    junctions = {n for n,d in degs.items() if d != 2}
    for s in junctions:
        for n in G.neighbors(s):
            ekey = tuple(sorted((s,n)))
            if ekey in used: continue
            path = [s,n]; used.add(ekey)
            prev, cur = s, n
            while G.degree[cur]==2:
                nb = [x for x in G.neighbors(cur) if x!=prev][0]
                used.add(tuple(sorted((cur,nb))))
                path.append(nb); prev,cur = cur,nb
            coords = [(G.nodes[i]["x"], G.nodes[i]["y"]) for i in path]
            if len(coords)>=2: branches.append(LineString(coords))
    if not branches and len(G.nodes):
        coords = [(G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes]
        try: branches=[LineString(coords)]
        except Exception: pass
    return branches

# ----------------- CANON per BRANCH + Δ-ANGLE -----------------
def branch_vectors(branch: LineString):
    coords = np.asarray(branch.coords, dtype=float).T
    if coords.shape[1] < 2: return np.array([0.0,1.0]), np.array([[1.0,0.0]])
    rel=[0.0]; vecs=[]
    for i in range(1, coords.shape[1]):
        rel.append(branch.project(Point(coords[0,i], coords[1,i]), normalized=True))
        vecs.append(coords[:,i]-coords[:,i-1])
    return np.asarray(rel), np.asarray(vecs)

def insert_pos(rel_cuts: np.ndarray, s: float) -> int:
    return int(np.searchsorted(rel_cuts, s, side="left"))

def canon_by_branch(bldg_polys: List[Polygon], block: Polygon, branch: LineString):
    N = len(bldg_polys)
    if N==0: return (np.zeros((0,2)), np.zeros((0,2)), np.zeros((0,)), np.zeros((0,)))
    rel_cuts, vecs = branch_vectors(branch)
    if len(vecs)==0: vecs=np.array([[1.0,0.0]]); rel_cuts=np.array([0.0,1.0])

    cuts_width=[]
    coords = np.asarray(branch.coords, dtype=float).T
    for i in range(1, coords.shape[1]-1):
        w = get_block_width_from_pt_on_axis(block, coords[:,i]-coords[:,i-1], Point(coords[0,i], coords[1,i]))
        cuts_width.append(w)
    mean_width = float(np.mean(cuts_width)) if cuts_width else (block.bounds[3]-block.bounds[1])
    L = max(branch.length, 1e-6)
    aspect = float(mean_width)/float(L)

    pos = np.zeros((N,2), float); size=np.zeros((N,2), float); phi=np.zeros((N,), float)
    for i, poly in enumerate(bldg_polys):
        c = poly.centroid
        s = branch.project(c, normalized=True)
        pos[i,0] = 2.0*s - 1.0
        pt_on = branch.interpolate(s, normalized=True)
        vec_to = np.array([c.x-pt_on.x, c.y-pt_on.y], float)
        ins = max(0, min(len(vecs)-1, insert_pos(rel_cuts, s)-1))
        axis_vec = vecs[ins]
        cross = np.cross(axis_vec, vec_to)
        dist = np.linalg.norm(vec_to)
        rel_y = 2.0*dist/mean_width
        pos[i,1] = -rel_y if cross<=0 else rel_y

        longside, shortside, long_vec, short_vec = get_size_with_vector(poly)
        long_vec  = long_vec /(np.linalg.norm(long_vec)+1e-9)
        short_vec = short_vec/(np.linalg.norm(short_vec)+1e-9)
        axis_unit = axis_vec /(np.linalg.norm(axis_vec)+1e-9)
        ang_long  = math.acos(np.clip(np.dot(long_vec , axis_unit), -1, 1))
        ang_short = math.acos(np.clip(np.dot(short_vec, axis_unit), -1, 1))
        if ang_short < ang_long:
            longside, shortside = shortside, longside
            long_vec = short_vec
        size[i,0] = 2.0*longside/L
        size[i,1] = 2.0*shortside/mean_width

        dot = np.clip(np.dot(long_vec, axis_unit), -1, 1)
        raw = min(math.pi - math.acos(dot), math.acos(dot))
        sign = 1.0 if np.cross(axis_unit, long_vec)>=0 else -1.0
        phi[i] = sign*raw
    return pos, size, phi, np.full((N,), aspect, float)

# ----------------- SHAPE PREFIT: s_i + a_i -----------------
def occupancy_ratio(poly: Polygon) -> float:
    mrr = poly.minimum_rotated_rectangle
    return float(poly.area) / max(float(mrr.area), 1e-9)

def reflex_count(poly: Polygon) -> int:
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords); cnt=0
    for i in range(n):
        p_prev = np.array(coords[(i-1)%n]); p = np.array(coords[i]); p_next = np.array(coords[(i+1)%n])
        v1 = p - p_prev; v2 = p_next - p
        z = np.cross(v1, v2)
        if z < 0: cnt += 1
    return cnt

def classify_shape(poly: Polygon) -> int:
    a = occupancy_ratio(poly)
    is_convex = poly.area >= 0.98 * poly.convex_hull.area
    if is_convex and a >= 0.8:
        return 0  # Rect
    r = reflex_count(poly)
    if len(poly.interiors) >= 1 or (a < 0.6 and r <= 3):
        return 2  # U
    if r >= 4:
        return 3  # X
    return 1      # L

# ----------------- MASK 64x64 + SCALE -----------------
def save_block_mask_64(poly: Polygon, out_path: str, size: int=64) -> bool:
    poly = _largest_polygon(poly)
    if poly is None:
        log.warning(f"mask: пропускаю пустой/некорректный блок → {out_path}")
        return False
    minx, miny, maxx, maxy = poly.bounds
    def w2p(x,y):
        u = (x - minx) / max(maxx-minx, 1e-9) * (size-1)
        v = (y - miny) / max(maxy-miny, 1e-9) * (size-1)
        return (u, (size-1)-v)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    ext = [w2p(x,y) for (x,y) in list(poly.exterior.coords)]
    draw.polygon(ext, fill=255)
    for ring in poly.interiors:
        ints = [w2p(x,y) for (x,y) in list(ring.coords)]
        draw.polygon(ints, fill=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return True


def block_scale_l(poly: Polygon, branches: List[LineString]) -> float:
    if branches:
        L = max([br.length for br in branches])
        if L > 0: return float(L)
    mrr = poly.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3]); axis2 = _dist(bbox[0], bbox[1])
    return float(max(axis1, axis2))

# ----------------- PADDING N -----------------
def allocate_slots_per_branch(counts: List[int], N: int) -> List[int]:
    total = sum(counts)
    if total == 0: return [0]*len(counts)
    base = [max(1, int(round(c/total * N))) for c in counts]
    diff = N - sum(base)
    rema = [c/total*N - b for c,b in zip(counts, base)]
    order = np.argsort(rema)[::-1]
    i=0
    while diff != 0 and len(order)>0:
        idx = int(order[i%len(order)])
        if diff>0: base[idx] += 1; diff -= 1
        else:
            if base[idx]>1: base[idx] -= 1; diff += 1
        i += 1
    return base

def evenly_subsample(idx: np.ndarray, k: int) -> np.ndarray:
    if len(idx) <= k: return idx
    pos = np.linspace(0, len(idx)-1, k).round().astype(int)
    return idx[pos]

# --- helper: корректная проверка пустоты для list/ndarray ---
def _cand_len(cands) -> int:
    try:
        # numpy array
        return int(cands.size)
    except Exception:
        # list/tuple
        try:
            return len(cands)
        except Exception:
            return 0

def preflight_overlap_check(bldgs: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, sample: int = 5000):
    # bbox
    bx = blocks.total_bounds  # minx, miny, maxx, maxy
    gx = bldgs.total_bounds
    log.info(f"blocks bbox: minx={bx[0]:.1f}, miny={bx[1]:.1f}, maxx={bx[2]:.1f}, maxy={bx[3]:.1f}")
    log.info(f"bldgs  bbox: minx={gx[0]:.1f}, miny={gx[1]:.1f}, maxx={gx[2]:.1f}, maxy={gx[3]:.1f}")

    if blocks.sindex is None:
        log.warning("blocks.sindex is None — GeoPandas не смог создать пространственный индекс")
        return

    # грубая оценка перекрытий по сэмплу зданий
    idx = np.arange(len(bldgs))
    if len(idx) > sample:
        idx = np.random.RandomState(42).choice(idx, size=sample, replace=False)
    try:
        inp, blk = blocks.sindex.query_bulk(bldgs.geometry.iloc[idx], predicate="intersects")
        hit_ratio = (len(np.unique(inp)) / len(idx)) if len(idx) else 0.0
        log.info(f"preflight: пересекается {hit_ratio*100:.2f}% выборки зданий со слоями blocks")
        if hit_ratio < 0.5:
            log.warning("Очень низкая доля пересечений. Похоже, слои мало (или вовсе не) перекрываются "
                        "(возможно, поданы не те 'blocks', например функциональные зоны).")
    except Exception as e:
        log.warning(f"preflight: не удалось оценить пересечения через sindex.query_bulk: {e}")

def _to_wkb_bytes(geom) -> bytes:
    try:
        import shapely
        return shapely.to_wkb(geom)     # Shapely 2.x
    except Exception:
        from shapely import wkb
        return wkb.dumps(geom)          # Shapely 1.x

def _preflight_overlap_check_compat(bldgs: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, sample: int = 4000):
    bx = blocks.total_bounds; gx = bldgs.total_bounds
    log.info(f"blocks bbox: minx={bx[0]:.1f}, miny={bx[1]:.1f}, maxx={bx[2]:.1f}, maxy={bx[3]:.1f}")
    log.info(f"bldgs  bbox: minx={gx[0]:.1f}, miny={gx[1]:.1f}, maxx={gx[2]:.1f}, maxy={gx[3]:.1f}")
    try:
        # Современный путь (если доступен)
        if hasattr(blocks.sindex, "query_bulk"):
            idx_inp, _ = blocks.sindex.query_bulk(
                bldgs.geometry.sample(min(sample, len(bldgs)), random_state=42),
                predicate="intersects"
            )
            hit = (len(np.unique(idx_inp)) / max(1, min(sample, len(bldgs))))
            log.info(f"preflight(query_bulk): ≈{hit*100:.2f}% зданий пересекают blocks")
        else:
            # Старый API: rtree.Index.intersection(bbox)
            rs = np.random.RandomState(42)
            ids = rs.choice(len(bldgs), size=min(sample, len(bldgs)), replace=False)
            hits = 0
            for i in ids:
                b = bldgs.geometry.iloc[i]
                cand = list(blocks.sindex.intersection(b.bounds))
                if any(blocks.geometry.iloc[j].intersects(b) for j in cand):
                    hits += 1
            hit = hits / max(1, len(ids))
            log.info(f"preflight(intersection): ≈{hit*100:.2f}% зданий пересекают blocks")
        if hit < 0.5:
            log.warning("Низкая доля пересечений — возможно, подан слой, который почти не перекрывается со зданиями.")
    except Exception as e:
        log.warning(f"preflight: не удалось оценить пересечения: {e}")


# ---------- helpers for parallel assign ----------
from shapely import wkb as _shp_wkb
import multiprocessing as mp

_ASSIGN_STATE = {
    "blk_ids": None,
    "blk_geoms": None,
    "blk_sindex": None,
    "blk_prepared": None,
}

def _init_assign_worker(blk_wkbs, blk_ids):
    """Инициализатор пула: один раз на процесс.
    Собираем геометрии блоков, пространственный индекс и prepared-предикаты.
    """
    geoms = [ _shp_wkb.loads(b) for b in blk_wkbs ]
    gs = gpd.GeoSeries(geoms, crs=None)   # CRS тут не нужен, всё уже в метрах
    from shapely.prepared import prep
    _ASSIGN_STATE["blk_ids"] = np.asarray(blk_ids)
    _ASSIGN_STATE["blk_geoms"] = geoms
    _ASSIGN_STATE["blk_sindex"] = gs.sindex
    _ASSIGN_STATE["blk_prepared"] = [prep(g) for g in geoms]

def _assign_chunk(args):
    """Обрабатывает чанк зданий в одном воркере.
    args: (global_idx_np[int], bldg_wkbs[List[bytes]])
    return: (global_idx_np[int], assigned_block_ix_np[int])
    """
    idx_global, wkbs = args
    geoms_b = [ _shp_wkb.loads(b) for b in wkbs ]
    centroids = [ g.centroid for g in geoms_b ]

    sindex = _ASSIGN_STATE["blk_sindex"]
    blk_geoms = _ASSIGN_STATE["blk_geoms"]
    blk_prepared = _ASSIGN_STATE["blk_prepared"]

    assigned = np.full(len(idx_global), -1, dtype=np.int64)

    # STEP1: centroid ∈ block (prepared.contains)
    for i, pt in enumerate(centroids):
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

def fast_assign_blocks(bldgs: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, log_prefix="", num_workers: int = 1, chunk_size: int = 20000) -> pd.Series:
    """Параллельное присвоение block_id зданиям.
    Совместимо с любыми версиями GeoPandas/Rtree (использует sindex.intersection).
    """
    t0 = time.perf_counter()

    # фильтрация валидных геометрий блоков
    valid = blocks.geometry.notna() & (~blocks.geometry.is_empty)
    blocks_v = blocks.loc[valid].copy()
    if blocks_v.empty:
        log.error(f"{log_prefix}нет валидных геометрий blocks"); 
        return pd.Series(pd.NA, index=bldgs.index, dtype="object")

    blk_ids_v = blocks_v["block_id"].to_numpy()
    blk_wkbs  = [ geom_to_wkb(g) for g in blocks_v.geometry.values ]  # bytes

    # подготовим WKB чанк зданий
    n = len(bldgs)
    all_idx = np.arange(n, dtype=np.int64)
    b_wkbs  = [ geom_to_wkb(g) for g in bldgs.geometry.values ]

    # последовательный путь (на всякий случай)
    if num_workers <= 1:
        _init_assign_worker(blk_wkbs, blk_ids_v)
        assigned_ix = np.full(n, -1, dtype=np.int64)
        for start in (tqdm(range(0, n, chunk_size), desc=f"{log_prefix}assign chanks") if TQDM else range(0, n, chunk_size)):
            end = min(start + chunk_size, n)
            ig, ax = _assign_chunk((all_idx[start:end], b_wkbs[start:end]))
            assigned_ix[ig] = ax
    else:
        # параллельный путь
        log.info(f"{log_prefix}parallel assign: workers={num_workers}, chunk_size={chunk_size}")
        assigned_ix = np.full(n, -1, dtype=np.int64)

        tasks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            tasks.append((all_idx[start:end], b_wkbs[start:end]))

        with mp.get_context("spawn").Pool(
            processes=num_workers,
            initializer=_init_assign_worker,
            initargs=(blk_wkbs, blk_ids_v),
        ) as pool:
            it = pool.imap_unordered(_assign_chunk, tasks, chunksize=1)
            if TQDM:
                it = tqdm(it, total=len(tasks), desc=f"{log_prefix}assign parallel")
            for ig, ax in it:
                assigned_ix[ig] = ax

    n_assigned = int((assigned_ix >= 0).sum())
    t1 = time.perf_counter()
    log.info(f"{log_prefix}assigned total: {n_assigned}/{n} in {t1 - t0:.2f}s")

    # собираем Series block_id
    out = pd.Series(pd.NA, index=bldgs.index, dtype="object")
    ok = assigned_ix >= 0
    out.iloc[np.where(ok)[0]] = blk_ids_v[assigned_ix[ok]]
    return out

def _empty_block_payload(blk_id: str, zone, mask_path: str, N: int) -> Dict[str, Any]:
    block_row = {
        "block_id": blk_id, "zone": zone,
        "n_buildings": 0, "n_branches": 0, "scale_l": 0.0, "mask_path": mask_path
    }
    nodes = [{
        "block_id": blk_id, "slot_id": i, "e_i": 0, "branch_local_id": -1,
        "posx": 0.0, "posy": 0.0, "size_x": 0.0, "size_y": 0.0, "phi_resid": 0.0,
        "s_i": 0, "a_i": 0.0, "floors_num": pd.NA, "living_area": 0.0,
        "is_living": False, "has_floors": False,
        "services_present": [], "services_capacity": [], "aspect_ratio": 0.0
    } for i in range(N)]
    return {"block": block_row, "branches": [], "nodes": nodes, "edges": []}

# ----------------- WORKER -----------------
def worker_process_block(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Обрабатывает один квартал:
      - poly_wkb -> Polygon (крупнейшая часть из MultiPolygon)
      - маска 64x64
      - скелет -> ветви; fallback к диагонали MRR
      - канон. координаты/размеры/дельта-угол по ветвям
      - паддинг до N; рёбра решётки
    Возвращает dict с ключами: block, branches, nodes, edges (всегда).
    """
    blk_id      = task["block_id"]
    zone        = task["zone"]
    poly_wkb    = task["poly_wkb"]
    rows_bldgs  = task["bldgs_rows"]  # список dict с geometry=WKB и атрибутами
    N           = int(task["N"])
    mask_size   = int(task["mask_size"])
    mask_dir    = task["mask_dir"]

    mask_path = os.path.join(mask_dir, f"{blk_id}.png")

    # --- локальный хелпер: пустой payload с паддингом ---
    def _empty_payload(scale_l: float = 0.0) -> Dict[str, Any]:
        block_row = {
            "block_id": blk_id, "zone": zone,
            "n_buildings": 0, "n_branches": 0,
            "scale_l": float(scale_l), "mask_path": mask_path
        }
        nodes = [{
            "block_id": blk_id, "slot_id": i, "e_i": 0, "branch_local_id": -1,
            "posx": 0.0, "posy": 0.0, "size_x": 0.0, "size_y": 0.0, "phi_resid": 0.0,
            "s_i": 0, "a_i": 0.0, "floors_num": pd.NA, "living_area": 0.0,
            "is_living": False, "has_floors": False,
            "services_present": [], "services_capacity": [], "aspect_ratio": 0.0
        } for i in range(N)]
        return {"block": block_row, "branches": [], "nodes": nodes, "edges": []}

    # --- геометрия квартала ---
    try:
        poly = geom_from_wkb(poly_wkb)
    except Exception:
        poly = None
    poly = _largest_polygon(poly)

    # --- маска 64x64 (чёрная, если геометрии нет) ---
    ok_mask = False
    if poly is not None:
        try:
            ok_mask = bool(save_block_mask_64(poly, mask_path, size=mask_size))
        except Exception:
            ok_mask = False
    if not ok_mask:
        # создаём пустую маску вручную
        try:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            Image.new("L", (mask_size, mask_size), 0).save(mask_path)
        except Exception:
            pass
        # если квартал пустой — возвращаем пустой payload
        if poly is None:
            return _empty_payload(0.0)

    # --- если в квартале нет зданий — вернуть паддинг с корректным scale_l ---
    if not rows_bldgs:
        scale_l = float(block_scale_l(poly, [])) if poly is not None else 0.0
        return _empty_payload(scale_l)

    # --- скелет и ветви; надёжный fallback ---
    try:
        skel = build_straight_skeleton(poly)
        G = skeleton_to_graph(skel)
        branches = extract_branches(G)
    except Exception:
        branches = []
    if not branches:
        # fallback: главная диагональ minimum_rotated_rectangle
        mrr = poly.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        branches = [LineString([coords[0], coords[2]])]
    valid_branches = [(i, br) for i, br in enumerate(branches)
                      if isinstance(br, LineString) and br.length > 0]
    scale_l = float(block_scale_l(poly, [br for _, br in valid_branches]))

    # --- здания: из WKB -> GeoDataFrame ---
    g_geoms = []
    attrs = []
    for r in rows_bldgs:
        try:
            g = geom_from_wkb(r["geometry"])
            if g is None or g.is_empty:
                continue
            g_geoms.append(g)
            # копия без geometry
            a = dict(r)
            a.pop("geometry", None)
            attrs.append(a)
        except Exception:
            continue

    if len(g_geoms) == 0:
        # все здания оказались пустыми/битими
        return _empty_payload(scale_l)

    df = pd.DataFrame(attrs)
    gdf = gpd.GeoDataFrame(df, geometry=g_geoms, crs=None)

    # --- привязка зданий к ближайшей ветви по центроиду ---
    cent = gdf.geometry.centroid
    nearest_idx = []
    for p in cent:
        dists = [br.distance(p) for _, br in valid_branches]
        nearest_idx.append(int(np.argmin(dists)) if dists else 0)
    gdf["branch_local_id"] = nearest_idx

    # --- канонические признаки по ветвям ---
    per_branch: Dict[int, dict] = {}
    branches_rows = [{"block_id": blk_id, "branch_local_id": i, "length": float(br.length)}
                     for i, br in valid_branches]
    counts = []
    branch_ids_sorted = []
    for j, br in valid_branches:
        rows = gdf[gdf["branch_local_id"] == j].reset_index(drop=True)
        if rows.empty:
            per_branch[j] = {
                "rows": rows, "pos": np.zeros((0, 2)), "size": np.zeros((0, 2)),
                "phi": np.zeros((0,)), "aspect": np.zeros((0,)), "s": np.zeros((0,), int),
                "a": np.zeros((0,), float)
            }
            cnt = 0
        else:
            pos, size, phi, aspect = canon_by_branch(list(rows.geometry.values), poly, br)
            s_labels, a_vals = [], []
            for geom in rows.geometry.values:
                try:
                    s_labels.append(classify_shape(geom))
                except Exception:
                    s_labels.append(0)
                try:
                    a_vals.append(occupancy_ratio(geom))
                except Exception:
                    a_vals.append(0.0)
            per_branch[j] = {
                "rows": rows, "pos": pos, "size": size, "phi": phi, "aspect": aspect,
                "s": np.array(s_labels, int), "a": np.array(a_vals, float)
            }
            cnt = len(rows)
        counts.append(cnt)
        branch_ids_sorted.append(j)

    # --- распределение N слотов между ветвями и выбор узлов ---
    alloc = allocate_slots_per_branch(counts, N)
    slot_meta = []
    slot = 0
    for j, cnt, kslots in zip(branch_ids_sorted, counts, alloc):
        feat = per_branch[j]
        if cnt == 0 or kslots == 0:
            continue
        order = np.argsort(feat["pos"][:, 0])  # слева-направо
        chosen = evenly_subsample(order, kslots)
        posx = feat["pos"][chosen, 0]
        ord2 = np.argsort(posx)
        for k in chosen[ord2]:
            slot_meta.append((slot, j, int(k)))
            slot += 1
            if slot >= N:
                break
        if slot >= N:
            break

    # --- узлы ---
    nodes = []
    for slot_id, j, k in slot_meta:
        feat = per_branch[j]
        r = feat["rows"].iloc[k]
        nodes.append({
            "block_id": blk_id, "slot_id": slot_id, "e_i": 1,
            "branch_local_id": int(j),
            "posx": float(feat["pos"][k, 0]), "posy": float(feat["pos"][k, 1]),
            "size_x": float(feat["size"][k, 0]), "size_y": float(feat["size"][k, 1]),
            "phi_resid": float(feat["phi"][k]),
            "s_i": int(feat["s"][k]) if "s" in feat else 0,
            "a_i": float(feat["a"][k]) if "a" in feat else 0.0,
            "floors_num": (pd.NA if pd.isna(r.get("floors_num")) else int(r.get("floors_num"))),
            "living_area": float(r.get("living_area", 0.0)),
            "is_living": bool(r.get("is_living", False)),
            "has_floors": bool(r.get("has_floors", False)),
            "services_present": list(r.get("services_present", [])),
            "services_capacity": list(r.get("services_capacity", [])),
            "aspect_ratio": float(feat["aspect"][k]),
        })

    # паддинг до N
    for slot_id in range(len(nodes), N):
        nodes.append({
            "block_id": blk_id, "slot_id": slot_id, "e_i": 0,
            "branch_local_id": -1,
            "posx": 0.0, "posy": 0.0, "size_x": 0.0, "size_y": 0.0, "phi_resid": 0.0,
            "s_i": 0, "a_i": 0.0, "floors_num": pd.NA, "living_area": 0.0,
            "is_living": False, "has_floors": False,
            "services_present": [], "services_capacity": [], "aspect_ratio": 0.0
        })

    # --- рёбра: вдоль веток + межветочные (решётка) ---
    edges = []
    slots_by_branch: Dict[int, list] = {}
    for slot_id, j, k in slot_meta:
        slots_by_branch.setdefault(j, []).append((slot_id, k))

    # по ветке
    for j, lst in slots_by_branch.items():
        feat = per_branch[j]
        lst_sorted = sorted(lst, key=lambda t: float(feat["pos"][t[1], 0]))
        for a in range(len(lst_sorted) - 1):
            s1 = lst_sorted[a][0]; s2 = lst_sorted[a + 1][0]
            edges.append({"block_id": blk_id, "src_slot": s1, "dst_slot": s2})
            edges.append({"block_id": blk_id, "src_slot": s2, "dst_slot": s1})

    # меж веток (грубая «решётка»)
    for idx in range(len(branch_ids_sorted) - 1):
        j1 = branch_ids_sorted[idx]
        j2 = branch_ids_sorted[idx + 1]
        if j1 not in slots_by_branch or j2 not in slots_by_branch:
            continue
        f1 = per_branch[j1]; f2 = per_branch[j2]
        A = sorted(slots_by_branch[j1], key=lambda t: float(f1["pos"][t[1], 0]))
        B = sorted(slots_by_branch[j2], key=lambda t: float(f2["pos"][t[1], 0]))
        m = min(len(A), len(B))
        for rnk in range(m):
            a = int(round((len(A) - 1) * rnk / max(m - 1, 1)))
            b = int(round((len(B) - 1) * rnk / max(m - 1, 1)))
            s1 = A[a][0]; s2 = B[b][0]
            edges.append({"block_id": blk_id, "src_slot": s1, "dst_slot": s2})
            edges.append({"block_id": blk_id, "src_slot": s2, "dst_slot": s1})

    block_row = {
        "block_id": blk_id, "zone": zone,
        "n_buildings": int(len(gdf)), "n_branches": int(len(valid_branches)),
        "scale_l": scale_l, "mask_path": mask_path
    }
    return {"block": block_row, "branches": branches_rows, "nodes": nodes, "edges": edges}

def _dump_orphans(bldgs: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # --- здания без привязки ---
    orphan_b = bldgs[bldgs["block_id"].isna()].copy()
    if not orphan_b.empty:
        # списки → строки (GeoJSON не любит массивы в полях)
        for col in ("services_present", "services_capacity"):
            if col in orphan_b.columns:
                orphan_b[col] = orphan_b[col].apply(
                    lambda v: json.dumps(v) if isinstance(v, list) else json.dumps([])
                )
        outb = os.path.join(out_dir, "orphaned_buildings.geojson")
        try:
            orphan_b.to_file(outb, driver="GeoJSON")
            log.info(f"orphaned_buildings: сохранено {len(orphan_b)} → {outb}")
        except Exception as e:
            log.warning(f"orphaned_buildings: не удалось сохранить GeoJSON: {e}")

    # --- кварталы без зданий ---
    used = set(bldgs["block_id"].dropna().unique().tolist())
    orphan_blk = blocks[~blocks["block_id"].isin(used)].copy()
    if not orphan_blk.empty:
        outk = os.path.join(out_dir, "orphaned_blocks.geojson")
        try:
            orphan_blk.to_file(outk, driver="GeoJSON")
            log.info(f"orphaned_blocks: сохранено {len(orphan_blk)} → {outk}")
        except Exception as e:
            log.warning(f"orphaned_blocks: не удалось сохранить GeoJSON: {e}")

# ----------------- MAIN -----------------
def main():
    ap = argparse.ArgumentParser("Иерархическая канонизация → HF parquet (адаптивный fast assign + logger + multiprocessing)")
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--target-crs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--mask-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    setup_logger(args.log_level)
    target_crs = normalize_crs_str(args.target_crs)
    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    log.info("Загрузка данных…")
    blocks = load_vector(args.blocks)
    bldgs  = load_vector(args.buildings)

    log.info(f"Приведение CRS к {target_crs}…")
    blocks = to_metric_crs(blocks, target_crs)
    bldgs  = to_metric_crs(bldgs, target_crs)

    log.info("Починка геометрий…")
    blocks["geometry"] = ensure_valid_series(blocks.geometry)
    bldgs["geometry"]  = ensure_valid_series(bldgs.geometry)

    blocks = blocks.reset_index(drop=True).copy()
    bldgs  = bldgs.reset_index(drop=True).copy()
    blocks["block_id"]   = blocks.index.astype(str)
    bldgs["building_id"] = bldgs.index.astype(str)
    if "zone" not in blocks.columns:
        blocks["zone"] = pd.NA

    # ---------- FAST ASSIGN вместо sjoin within ----------
    log.info("Быстрое присвоение зданий кварталам (STRtree)…")
    t0 = time.perf_counter()
    bldgs["block_id"] = fast_assign_blocks(
    bldgs, blocks, log_prefix="assign: ",
    num_workers=max(1, int(args.num_workers)),   # берём уже существующий CLI-параметр
    chunk_size=20000                              # можно подвинтить под ОЗУ/CPU
    )
    n_unassigned = int(bldgs["block_id"].isna().sum())
    if n_unassigned:
        log.warning(f"Неприсвоенных зданий: {n_unassigned} (будут пропущены)")
        _dump_orphans(bldgs, blocks, args.out_dir)
        bldgs = bldgs[~bldgs["block_id"].isna()].copy()
    t1 = time.perf_counter()
    log.info(f"assign total time: {t1 - t0:.2f}s")

    # гарантируем атрибуты зданий
    if "floors_num" not in bldgs.columns and "storeys_count" in bldgs.columns:
        bldgs["floors_num"] = bldgs["storeys_count"]
    for k, default in {
        "floors_num": pd.NA, "living_area": 0.0, "is_living": False, "has_floors": False,
        "services_present": None, "services_capacity": None
    }.items():
        if k not in bldgs.columns: bldgs[k] = default
    bldgs["floors_num"] = bldgs["floors_num"].astype("Int64")
    bldgs["living_area"] = pd.to_numeric(bldgs["living_area"], errors="coerce").fillna(0.0).astype(float)
    bldgs["is_living"] = bldgs["is_living"].astype(bool)
    bldgs["has_floors"] = bldgs["has_floors"].astype(bool)
    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)): return list(x)
        return [] if pd.isna(x) else []
    bldgs["services_present"]  = bldgs["services_present"].apply(_as_list)
    bldgs["services_capacity"] = bldgs["services_capacity"].apply(_as_list)

    # ---------- ЗАДАЧИ ДЛЯ ВОРКЕРОВ ----------
    log.info("Формирование заданий по кварталам…")
    block_records = blocks[["block_id","zone","geometry"]].copy()
    block_records["poly_wkb"] = block_records.geometry.apply(geom_to_wkb)
    block_records = block_records.drop(columns=["geometry"])

    groups = bldgs.groupby("block_id")
    tasks = []
    for _, row in block_records.iterrows():
        blk_id = row["block_id"]
        sub = groups.get_group(blk_id) if blk_id in groups.groups else pd.DataFrame(columns=bldgs.columns)
        rows_bldgs = []
        if not sub.empty:
            for r in sub.itertuples(index=False):
                rows_bldgs.append({
                    "building_id": r.building_id,
                    "geometry": geom_to_wkb(getattr(r, "geometry")),
                    "floors_num": getattr(r, "floors_num", pd.NA),
                    "living_area": getattr(r, "living_area", 0.0),
                    "is_living": bool(getattr(r, "is_living", False)),
                    "has_floors": bool(getattr(r, "has_floors", False)),
                    "services_present": list(getattr(r, "services_present", [])),
                    "services_capacity": list(getattr(r, "services_capacity", [])),
                })
        tasks.append({
            "block_id": blk_id, "zone": row["zone"], "poly_wkb": row["poly_wkb"],
            "bldgs_rows": rows_bldgs, "N": args.N, "mask_size": args.mask_size, "mask_dir": mask_dir
        })

    num_workers = max(1, int(args.num_workers))
    log.info(f"Параллельная обработка блоков: workers={num_workers}, blocks={len(tasks)}")
    if num_workers == 1:
        results = [worker_process_block(t) for t in tqdm(tasks, disable=not TQDM)]
    else:
        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(worker_process_block, tasks), total=len(tasks), disable=not TQDM))

    # ---------- СБОРКА И ЗАПИСЬ ----------
    log.info("Сборка результатов и запись Parquet…")
    blocks_rows=[]; branches_rows=[]; nodes_rows=[]; edges_rows=[]
    for res in results:
        blocks_rows.append(res["block"])
        branches_rows.extend(res["branches"])
        nodes_rows.extend(res["nodes"])
        edges_rows.extend(res["edges"])

    blocks_df   = pd.DataFrame(blocks_rows)
    branches_df = pd.DataFrame(branches_rows)
    nodes_df    = pd.DataFrame(nodes_rows)
    edges_df    = pd.DataFrame(edges_rows)
    if "floors_num" in nodes_df.columns:
        nodes_df["floors_num"] = nodes_df["floors_num"].astype("Int64")

    out = args.out_dir
    blocks_df.to_parquet(os.path.join(out, "blocks.parquet"), index=False)
    branches_df.to_parquet(os.path.join(out, "branches.parquet"), index=False)
    nodes_df.to_parquet(os.path.join(out, "nodes_fixed.parquet"), index=False)
    edges_df.to_parquet(os.path.join(out, "edges.parquet"), index=False)

    log.info(f"[ok] saved {os.path.join(out,'blocks.parquet')}     rows={len(blocks_df)}")
    log.info(f"[ok] saved {os.path.join(out,'branches.parquet')}   rows={len(branches_df)}")
    log.info(f"[ok] saved {os.path.join(out,'nodes_fixed.parquet')} rows={len(nodes_df)}")
    log.info(f"[ok] saved {os.path.join(out,'edges.parquet')}      rows={len(edges_df)}")
    log.info(f"[note] masks in {mask_dir}")

if __name__ == "__main__":
    sys.exit(main())
