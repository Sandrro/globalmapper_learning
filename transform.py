#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
transform.py
Иерархическая канонизация → HF parquet.

Обновления:
- Быстрый assign зданий к кварталам (STRtree + prepared.contains + fallback по площади пересечения).
- Параллельная обработка кварталов с мягким UNIX-таймаутом на каждый квартал (по умолчанию 300 сек).
- КАЖДЫЙ квартал обрабатывается и сохраняется в отдельные parquet-файлы:
    out/by_block/<block_id>/{blocks,branches,nodes_fixed,edges}.parquet
  В конце все per-block-файлы собираются в общие 4 файла в корне out/.
- Сохранение orphaned_buildings.geojson и orphaned_blocks.geojson.
- Сохранение timeout.geojson (кварталы, не уложившиеся в таймаут, + их здания).
- Исправлен NameError: _cand_len в воркере, аккуратные проверки без этой функции.
- Исправлена сериализация списков с np.int64/np.float64 в timeout.geojson.
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
import signal
from contextlib import contextmanager

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

# ----------------- TIMEOUT helper -----------------
@contextmanager
def time_limit(seconds: int):
    """Unix-only soft таймаут. В воркере (отдельный процесс) безопасно."""
    def _handler(signum, frame):
        raise TimeoutError("worker timeout")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# ---------- helpers for parallel assign ----------
from shapely import wkb as _shp_wkb

_ASSIGN_STATE = {
    "blk_ids": None,
    "blk_geoms": None,
    "blk_sindex": None,
    "blk_prepared": None,
}

def _init_assign_worker(blk_wkbs, blk_ids):
    geoms = [ _shp_wkb.loads(b) for b in blk_wkbs ]
    gs = gpd.GeoSeries(geoms, crs=None)
    from shapely.prepared import prep
    _ASSIGN_STATE["blk_ids"] = np.asarray(blk_ids)
    _ASSIGN_STATE["blk_geoms"] = geoms
    _ASSIGN_STATE["blk_sindex"] = gs.sindex
    _ASSIGN_STATE["blk_prepared"] = [prep(g) for g in geoms]

def _assign_chunk(args):
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
    t0 = time.perf_counter()

    valid = blocks.geometry.notna() & (~blocks.geometry.is_empty)
    blocks_v = blocks.loc[valid].copy()
    if blocks_v.empty:
        log.error(f"{log_prefix}нет валидных геометрий blocks"); 
        return pd.Series(pd.NA, index=bldgs.index, dtype="object")

    blk_ids_v = blocks_v["block_id"].to_numpy()
    blk_wkbs  = [ geom_to_wkb(g) for g in blocks_v.geometry.values ]

    n = len(bldgs)
    all_idx = np.arange(n, dtype=np.int64)
    b_wkbs  = [ geom_to_wkb(g) for g in bldgs.geometry.values ]

    if num_workers <= 1:
        _init_assign_worker(blk_wkbs, blk_ids_v)
        assigned_ix = np.full(n, -1, dtype=np.int64)
        rng = tqdm(range(0, n, chunk_size), desc=f"{log_prefix}assign chunks") if TQDM else range(0, n, chunk_size)
        for start in rng:
            end = min(start + chunk_size, n)
            ig, ax = _assign_chunk((all_idx[start:end], b_wkbs[start:end]))
            assigned_ix[ig] = ax
    else:
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

    out = pd.Series(pd.NA, index=bldgs.index, dtype="object")
    ok = assigned_ix >= 0
    out.iloc[np.where(ok)[0]] = blk_ids_v[assigned_ix[ok]]
    return out

# ----------------- EMPTY PAYLOAD HELPER -----------------
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

# ----------------- worker: per-block save + timeout -----------------
def _safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        try:
            return int(x.size)
        except Exception:
            return 0

def _block_dir(out_dir: str, blk_id: str) -> str:
    return os.path.join(out_dir, "by_block", str(blk_id))

def _write_block_parquets(out_dir: str, blk_id: str, block_row: dict, branches_rows: list, nodes_rows: list, edges_rows: list):
    bdir = _block_dir(out_dir, blk_id)
    os.makedirs(bdir, exist_ok=True)
    pd.DataFrame([block_row]).to_parquet(os.path.join(bdir, "blocks.parquet"), index=False)
    pd.DataFrame(branches_rows).to_parquet(os.path.join(bdir, "branches.parquet"), index=False)
    nodes_df = pd.DataFrame(nodes_rows)
    if "floors_num" in nodes_df.columns:
        nodes_df["floors_num"] = nodes_df["floors_num"].astype("Int64")
    nodes_df.to_parquet(os.path.join(bdir, "nodes_fixed.parquet"), index=False)
    pd.DataFrame(edges_rows).to_parquet(os.path.join(bdir, "edges.parquet"), index=False)


def worker_process_block(task: Dict[str, Any]) -> Dict[str, Any]:
    blk_id      = task["block_id"]
    zone        = task["zone"]
    poly_wkb    = task["poly_wkb"]
    rows_bldgs  = task["bldgs_rows"]
    N           = int(task["N"])
    mask_size   = int(task["mask_size"])
    mask_dir    = task["mask_dir"]
    timeout_sec = int(task.get("timeout_sec", 300))
    out_dir     = task["out_dir"]

    mask_path = os.path.join(mask_dir, f"{blk_id}.png")

    # квартальная геометрия
    try:
        poly = geom_from_wkb(poly_wkb)
    except Exception:
        poly = None
    poly = _largest_polygon(poly)

    # mask
    ok_mask = False
    if poly is not None:
        try:
            ok_mask = bool(save_block_mask_64(poly, mask_path, size=mask_size))
        except Exception:
            ok_mask = False
    if not ok_mask:
        try:
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            Image.new("L", (mask_size, mask_size), 0).save(mask_path)
        except Exception:
            pass
        if poly is None:
            # пустой квартал
            payload = _empty_block_payload(blk_id, zone, mask_path, N)
            _write_block_parquets(out_dir, blk_id, payload["block"], [], payload["nodes"], [])
            return {"block_id": blk_id, "status": "ok_empty"}

    if not rows_bldgs:
        scale_l = float(block_scale_l(poly, [])) if poly is not None else 0.0
        payload = _empty_block_payload(blk_id, zone, mask_path, N)
        payload["block"]["scale_l"] = scale_l
        _write_block_parquets(out_dir, blk_id, payload["block"], [], payload["nodes"], [])
        return {"block_id": blk_id, "status": "ok_empty"}

    try:
        with time_limit(timeout_sec):
            # 1) skeleton & branches
            try:
                skel = build_straight_skeleton(poly)
                G = skeleton_to_graph(skel)
                branches = extract_branches(G)
            except Exception:
                branches = []
            if not branches:
                mrr = poly.minimum_rotated_rectangle
                coords = list(mrr.exterior.coords)
                branches = [LineString([coords[0], coords[2]])]
            valid_branches = [(i, br) for i, br in enumerate(branches)
                              if isinstance(br, LineString) and br.length > 0]
            scale_l = float(block_scale_l(poly, [br for _, br in valid_branches]))

            # 2) buildings from WKB
            g_geoms, attrs = [], []
            for r in rows_bldgs:
                try:
                    g = geom_from_wkb(r["geometry"])
                except Exception:
                    g = None
                if g is None or g.is_empty:
                    continue
                g_geoms.append(g)
                a = dict(r); a.pop("geometry", None)
                attrs.append(a)
            if len(g_geoms) == 0:
                payload = _empty_block_payload(blk_id, zone, mask_path, N)
                payload["block"]["scale_l"] = scale_l
                _write_block_parquets(out_dir, blk_id, payload["block"], [], payload["nodes"], [])
                return {"block_id": blk_id, "status": "ok_empty"}

            df = pd.DataFrame(attrs)
            gdf = gpd.GeoDataFrame(df, geometry=g_geoms, crs=None)

            # 3) nearest branch by centroid
            cent = gdf.geometry.centroid
            nearest_idx = []
            for p in cent:
                dists = [br.distance(p) for _, br in valid_branches]
                nearest_idx.append(int(np.argmin(dists)) if dists else 0)
            gdf["branch_local_id"] = nearest_idx

            # 4) per-branch features
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
                        try: s_labels.append(int(classify_shape(geom)))
                        except Exception: s_labels.append(0)
                        try: a_vals.append(float(occupancy_ratio(geom)))
                        except Exception: a_vals.append(0.0)
                    per_branch[j] = {
                        "rows": rows, "pos": pos, "size": size, "phi": phi, "aspect": aspect,
                        "s": np.array(s_labels, int), "a": np.array(a_vals, float)
                    }
                    cnt = len(rows)
                counts.append(cnt)
                branch_ids_sorted.append(j)

            # 5) allocate N slots & choose
            alloc = allocate_slots_per_branch(counts, N)
            slot_meta = []
            slot = 0
            for j, cnt, kslots in zip(branch_ids_sorted, counts, alloc):
                feat = per_branch[j]
                if cnt == 0 or kslots == 0:
                    continue
                order = np.argsort(feat["pos"][:, 0])
                chosen = evenly_subsample(order, kslots)
                posx = feat["pos"][chosen, 0]
                ord2 = np.argsort(posx)
                for k in chosen[ord2]:
                    slot_meta.append((slot, j, int(k)))
                    slot += 1
                    if slot >= N: break
                if slot >= N: break

            # 6) nodes
            nodes = []
            for slot_id, j, k in slot_meta:
                feat = per_branch[j]
                r = feat["rows"].iloc[k]
                s_vec = feat.get("s", [])
                a_vec = feat.get("a", [])
                s_i = int(s_vec[k]) if isinstance(s_vec, (list, tuple, np.ndarray)) and _safe_len(s_vec) > k else 0
                a_i = float(a_vec[k]) if isinstance(a_vec, (list, tuple, np.ndarray)) and _safe_len(a_vec) > k else 0.0
                nodes.append({
                    "block_id": blk_id, "slot_id": slot_id, "e_i": 1,
                    "branch_local_id": int(j),
                    "posx": float(feat["pos"][k, 0]), "posy": float(feat["pos"][k, 1]),
                    "size_x": float(feat["size"][k, 0]), "size_y": float(feat["size"][k, 1]),
                    "phi_resid": float(feat["phi"][k]),
                    "s_i": s_i,
                    "a_i": a_i,
                    "floors_num": (pd.NA if pd.isna(r.get("floors_num")) else int(r.get("floors_num"))),
                    "living_area": float(r.get("living_area", 0.0)),
                    "is_living": bool(r.get("is_living", False)),
                    "has_floors": bool(r.get("has_floors", False)),
                    "services_present": list(r.get("services_present", [])),
                    "services_capacity": list(r.get("services_capacity", [])),
                    "aspect_ratio": float(feat["aspect"][k]) if _safe_len(feat.get("aspect", []))>k else 0.0,
                })

            for slot_id in range(len(nodes), N):
                nodes.append({
                    "block_id": blk_id, "slot_id": slot_id, "e_i": 0,
                    "branch_local_id": -1,
                    "posx": 0.0, "posy": 0.0, "size_x": 0.0, "size_y": 0.0, "phi_resid": 0.0,
                    "s_i": 0, "a_i": 0.0, "floors_num": pd.NA, "living_area": 0.0,
                    "is_living": False, "has_floors": False,
                    "services_present": [], "services_capacity": [], "aspect_ratio": 0.0
                })

            # 7) edges
            edges = []
            slots_by_branch: Dict[int, list] = {}
            for slot_id, j, k in slot_meta:
                slots_by_branch.setdefault(j, []).append((slot_id, k))

            for j, lst in slots_by_branch.items():
                feat = per_branch[j]
                lst_sorted = sorted(lst, key=lambda t: float(feat["pos"][t[1], 0]))
                for a in range(len(lst_sorted) - 1):
                    s1 = lst_sorted[a][0]; s2 = lst_sorted[a + 1][0]
                    edges.append({"block_id": blk_id, "src_slot": s1, "dst_slot": s2})
                    edges.append({"block_id": blk_id, "src_slot": s2, "dst_slot": s1})

            for idx in range(len(branch_ids_sorted) - 1):
                j1 = branch_ids_sorted[idx]; j2 = branch_ids_sorted[idx + 1]
                if j1 not in slots_by_branch or j2 not in slots_by_branch: continue
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

            # записать per-block parquet
            _write_block_parquets(out_dir, blk_id, block_row, branches_rows, nodes, edges)
            return {"block_id": blk_id, "status": "ok"}

    except TimeoutError:
        return {
            "status": "timeout",
            "block_id": blk_id,
            "zone": zone,
            "poly_wkb": poly_wkb,
            "bldgs_rows": rows_bldgs,
        }

# ----------------- ORPHANS / TIMEOUT DUMPS -----------------
def _json_safe_list(v) -> str:
    try:
        if v is None:
            return "[]"
        lst = list(v)
        out = []
        for x in lst:
            if isinstance(x, (np.integer,)):
                out.append(int(x))
            elif isinstance(x, (np.floating,)):
                out.append(float(x))
            else:
                out.append(x)
        return json.dumps(out)
    except Exception:
        return "[]"

def _dump_orphans(bldgs: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    orphan_b = bldgs[bldgs["block_id"].isna()].copy()
    if not orphan_b.empty:
        for col in ("services_present", "services_capacity"):
            if col in orphan_b.columns:
                orphan_b[col] = orphan_b[col].apply(_json_safe_list)
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

def _dump_timeouts(timeout_packets: List[Dict[str, Any]], out_dir: str, crs) -> None:
    if not timeout_packets: return
    rows = []
    for item in timeout_packets:
        blk_id = item.get("block_id")
        zone   = item.get("zone")
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
            rows.append({
                "kind": "building",
                "block_id": blk_id,
                "building_id": r.get("building_id"),
                "floors_num": (None if pd.isna(r.get("floors_num")) else int(r.get("floors_num"))),
                "living_area": float(r.get("living_area", 0.0)),
                "is_living": bool(r.get("is_living", False)),
                "has_floors": bool(r.get("has_floors", False)),
                "services_present": _json_safe_list(r.get("services_present", [])),
                "services_capacity": _json_safe_list(r.get("services_capacity", [])),
                "geometry": g,
            })
    if not rows: return
    gdf = gpd.GeoDataFrame(pd.DataFrame(rows), geometry="geometry", crs=crs)
    outp = os.path.join(out_dir, "timeout.geojson")
    try:
        gdf.to_file(outp, driver="GeoJSON")
        log.info(f"timeout.geojson: сохранено {len(gdf)} объектов (блоки+здания) → {outp}")
    except Exception as e:
        log.warning(f"timeout.geojson: не удалось сохранить GeoJSON: {e}")

# ----------------- MERGE per-block parquet -----------------
from glob import glob

def _merge_parquet_tree(by_block_root: str, rel_name: str, out_path: str):
    paths = sorted(glob(os.path.join(by_block_root, "*", rel_name)))
    if not paths:
        pd.DataFrame([]).to_parquet(out_path, index=False)
        return 0
    dfs = []
    for p in (tqdm(paths, desc=f"merge {rel_name}") if TQDM else paths):
        try:
            dfs.append(pd.read_parquet(p))
        except Exception:
            pass
    if not dfs:
        pd.DataFrame([]).to_parquet(out_path, index=False)
        return 0
    df = pd.concat(dfs, ignore_index=True)
    if rel_name == "nodes_fixed.parquet" and "floors_num" in df.columns:
        try:
            df["floors_num"] = df["floors_num"].astype("Int64")
        except Exception:
            pass
    df.to_parquet(out_path, index=False)
    return len(df)

# ----------------- MAIN -----------------
def main():
    ap = argparse.ArgumentParser("Иерархическая канонизация → HF parquet (assign + per-block save + timeout)")
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--buildings", required=True)
    ap.add_argument("--target-crs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--N", type=int, default=120)
    ap.add_argument("--mask-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--block-timeout-sec", type=int, default=300, help="Таймаут на обработку одного квартала (сек)")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    setup_logger(args.log_level)
    target_crs = normalize_crs_str(args.target_crs)
    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    by_block_root = os.path.join(args.out_dir, "by_block")
    os.makedirs(by_block_root, exist_ok=True)

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

    # ---------- FAST ASSIGN ----------
    log.info("Быстрое присвоение зданий кварталам (STRtree)…")
    t0 = time.perf_counter()
    bldgs["block_id"] = fast_assign_blocks(
        bldgs, blocks, log_prefix="assign: ",
        num_workers=max(1, int(args.num_workers)),
        chunk_size=20000
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

    # ---------- задачи для воркеров ----------
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
            "bldgs_rows": rows_bldgs, "N": args.N, "mask_size": args.mask_size, "mask_dir": mask_dir,
            "timeout_sec": int(args.block_timeout_sec), "out_dir": args.out_dir
        })

    num_workers = max(1, int(args.num_workers))
    log.info(f"Параллельная обработка блоков: workers={num_workers}, blocks={len(tasks)}")
    if num_workers == 1:
        results = [worker_process_block(t) for t in tqdm(tasks, disable=not TQDM)]
    else:
        with mp.get_context("spawn").Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(worker_process_block, tasks), total=len(tasks), disable=not TQDM))

    # собрать таймауты
    timeouts = [r for r in results if r and r.get("status") == "timeout"]
    _dump_timeouts(timeouts, args.out_dir, crs=blocks.crs)

    # ---------- Мердж per-block parquet в общие 4 файла ----------
    log.info("Сборка результатов и запись Parquet…")
    out_root = args.out_dir

    # Merge per-block parquet trees → global 4 files
    blocks_total   = _merge_parquet_tree(by_block_root, "blocks.parquet",   os.path.join(out_root, "blocks.parquet"))
    branches_total = _merge_parquet_tree(by_block_root, "branches.parquet", os.path.join(out_root, "branches.parquet"))
    nodes_total    = _merge_parquet_tree(by_block_root, "nodes_fixed.parquet", os.path.join(out_root, "nodes_fixed.parquet"))
    edges_total    = _merge_parquet_tree(by_block_root, "edges.parquet",    os.path.join(out_root, "edges.parquet"))

    log.info(f"[ok] saved {os.path.join(out_root,'blocks.parquet')}     rows={blocks_total}")
    log.info(f"[ok] saved {os.path.join(out_root,'branches.parquet')}   rows={branches_total}")
    log.info(f"[ok] saved {os.path.join(out_root,'nodes_fixed.parquet')} rows={nodes_total}")
    log.info(f"[ok] saved {os.path.join(out_root,'edges.parquet')}      rows={edges_total}")
    log.info(f"[note] per-block parquet in {by_block_root}")
    log.info(f"[note] masks in {mask_dir}")

    # Ensure orphaned_blocks is written even if there were no orphaned buildings
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

    # Summary on timeouts
    if timeouts:
        unique_timeout_blocks = sorted({t.get("block_id") for t in timeouts if t})
        log.warning(f"Timeout on {len(unique_timeout_blocks)} blocks → see timeout.geojson")

    log.info("Готово.")

if __name__ == "__main__":
    sys.exit(main())