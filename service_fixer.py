#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_nodes_add_services_geojson.py

Добавляет в nodes_fixed.parquet колонки:
  - services_present: JSON-список имён сервисов
  - services_capacity: JSON-список {name, value}

Источник:
  - Зоны: GeoJSON (например, CRS84/4326). Если нет properties.block_id — берётся индекс фичи (строкой),
    чтобы совпасть с поведением transform_pipeline_cli.py.
  - Сервисы: GeoJSON в EPSG:4326 (или другом CRS), имя берётся из properties[--name-field].

Фильтрация:
  --exclude-names "Парк,Детская площадка,Спортивная площадка" (регистронезависимо).

Привязка:
  1) сервис -> блок (contains, иначе ближайший полигон);
  2) centre_point или centroid -> перевод в целевой CRS;
  3) world->canonical по блоку, далее soft top-K ближайших узлов (веса ~ 1/d).

Примечание: если твои nodes_fixed.parquet получены тем же набором зон, индексация block_id совпадёт автоматически.
"""
import os, json, math, argparse
from typing import List, Tuple, Dict, Callable
import numpy as np
import pandas as pd
from shapely.geometry import shape as shp_shape, Point as ShpPoint, Polygon as ShpPolygon
from shapely.strtree import STRtree
from shapely.affinity import rotate as shp_rotate, scale as shp_scale, translate as shp_translate

try:
    from pyproj import Transformer
except Exception as e:
    raise SystemExit("pyproj не установлен. Установи: pip install pyproj") from e


# ---------- утилиты ----------
def _norm_name(s: str) -> str:
    return str(s).strip().casefold()

def canonicalize_polygon(poly: ShpPolygon):
    xs, ys = poly.exterior.xy
    X = np.vstack([np.asarray(xs) - np.mean(xs), np.asarray(ys) - np.mean(ys)])
    C = X @ X.T / max(1, X.shape[1]-1)
    eigvals, eigvecs = np.linalg.eig(C)
    v = eigvecs[:, int(np.argmax(eigvals))]
    angle = math.degrees(math.atan2(float(v[1]), float(v[0])))
    c = poly.centroid
    p0 = shp_translate(poly, xoff=-c.x, yoff=-c.y)
    p1 = shp_rotate(p0, -angle, origin=(0,0), use_radians=False)
    minx, miny, maxx, maxy = p1.bounds
    w, h = maxx - minx, maxy - miny
    s = 1.0 / max(w, h) if max(w, h) > 1e-12 else 1.0
    p2 = shp_scale(p1, xfact=s, yfact=s, origin=(0,0))
    minx2, miny2, _, _ = p2.bounds
    p_can = shp_translate(p2, xoff=-minx2, yoff=-miny2)

    def _mat_translate(dx, dy):
        return np.array([[1.0, 0.0, float(dx)],
                         [0.0, 1.0, float(dy)],
                         [0.0, 0.0, 1.0]], dtype=float)
    def _mat_rotate(deg):
        th = math.radians(float(deg)); c_, s_ = math.cos(th), math.sin(th)
        return np.array([[ c_, -s_, 0.0],
                         [ s_,  c_, 0.0],
                         [ 0.0, 0.0, 1.0]], dtype=float)
    def _mat_scale(sx, sy):
        return np.array([[float(sx), 0.0, 0.0],
                         [0.0, float(sy), 0.0],
                         [0.0, 0.0, 1.0]], dtype=float)

    M    = _mat_translate(-minx2, -miny2) @ _mat_scale(s, s) @ _mat_rotate(-angle) @ _mat_translate(-c.x, -c.y)
    Minv = _mat_translate( c.x,  c.y )   @ _mat_rotate( angle) @ _mat_scale(1/s, 1/s) @ _mat_translate(minx2, miny2)

    def _apply(M_, xy):
        v = np.array([float(xy[0]), float(xy[1]), 1.0], dtype=float)
        r = M_ @ v
        return np.array([r[0], r[1]], dtype=float)

    return p_can, (lambda p: _apply(M, p)), (lambda p: _apply(Minv, p))


def soft_weights(dists: np.ndarray, k: int):
    d = np.asarray(dists, dtype=float)
    if d.size == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    idx = np.argsort(d)[:max(1, k)]
    dd = d[idx]
    dd = np.maximum(dd, 1e-6)
    w = 1.0 / dd
    w = w / w.sum()
    return idx, w


# ---------- загрузка данных ----------
def load_blocks_geojson(path: str) -> Tuple[List[ShpPolygon], List[str]]:
    """
    Читает FeatureCollection. Если нет properties.block_id — присваивает block_id по индексу (строкой):
      '0','1','2',...
    MultiPolygon -> берётся самый большой полигон.
    """
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else []
    polys, ids = [], []
    for i, f in enumerate(feats):
        geom = shp_shape(f["geometry"])
        if geom.is_empty:
            continue
        if geom.geom_type == "MultiPolygon":
            geom = max(list(geom.geoms), key=lambda g: g.area)
        props = (f.get("properties") or {})
        bid = props.get("block_id")
        if bid is None:
            # реплицируем transform_pipeline_cli: index как строка
            bid = str(i)
        else:
            bid = str(bid)
        polys.append(geom)
        ids.append(bid)
    return polys, ids


def load_services_and_project(
    path: str,
    transformer: Transformer,
    name_field: str = "service_type_name",
    cap_fields: List[str] = None,
    exclude_names_norm: set = None,
):
    """
    Грузим сервисы из GeoJSON (обычно EPSG:4326/CRS84). Берём properties.centre_point или centroid.
    Проецируем координаты через transformer в целевой CRS блоков.
    """
    if cap_fields is None:
        cap_fields = ["capacity", "capacity_modeled", "weight_value"]
    exclude_names_norm = exclude_names_norm or set()

    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else []

    items = []
    for f in feats:
        geom = shp_shape(f["geometry"])
        if geom.is_empty:
            continue
        props = f.get("properties", {}) or {}

        name = props.get(name_field) or props.get("name")
        if not name:
            continue
        if _norm_name(name) in exclude_names_norm:
            continue

        cp = props.get("centre_point")
        if isinstance(cp, (list, tuple)) and len(cp) == 2:
            lon, lat = float(cp[0]), float(cp[1])
        else:
            c = geom.centroid
            lon, lat = float(c.x), float(c.y)

        val = None
        for cf in (cap_fields or []):
            if cf in props and props[cf] is not None:
                try:
                    val = float(props[cf]); break
                except Exception:
                    pass
        if val is None:
            val = 1.0

        x, y = transformer.transform(lon, lat)  # always_xy=True
        items.append({"pt": ShpPoint(x, y), "name": str(name), "value": float(val)})
    return items

import numpy as np  # убедись, что импорт есть сверху

def _as_geom_list(query_result, polys):
    """
    Приводит результат STRtree.query(...) к списку геометрий.
    Поддерживает случаи, когда возвращаются индексы (np.int64).
    """
    if query_result is None:
        return []
    # numpy.ndarray
    if isinstance(query_result, np.ndarray):
        if query_result.dtype == object:
            # shapely 1.x/2.x: массив геометрий
            return [g for g in query_result.tolist() if hasattr(g, "geom_type")]
        if np.issubdtype(query_result.dtype, np.integer):
            # массив индексов
            return [polys[int(i)] for i in query_result.tolist()]
        # на всякий случай
        return list(query_result)
    # list/tuple
    if isinstance(query_result, (list, tuple)):
        out = []
        for x in query_result:
            if isinstance(x, (int, np.integer)):
                out.append(polys[int(x)])
            else:
                out.append(x)
        return out
    # одиночный объект
    return [query_result]

# ---------- основная логика ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="nodes_fixed.parquet")
    ap.add_argument("--blocks", required=True, help="zones.geojson (CRS84/4326). Если нет block_id — берётся индекс.")
    ap.add_argument("--services", required=True, help="services.geojson (обычно EPSG:4326)")
    ap.add_argument("--services-epsg", type=int, default=4326, help="EPSG сервисов (вход)")
    ap.add_argument("--target-epsg", type=int, default=4326, help="EPSG зон/блоков (если zones в CRS84/4326 — ставь 4326)")
    ap.add_argument("--name-field", default="service_type_name", help="Имя поля сервиса (в properties)")
    ap.add_argument("--cap-fields", default="capacity,capacity_modeled,weight_value", help="Поля ёмкости через запятую")
    ap.add_argument("--exclude-names", default="Парк,Детская площадка,Спортивная площадка",
                    help="ИСКЛЮЧИТЬ сервисы (через запятую), регистронезависимо")
    ap.add_argument("--topk", type=int, default=3, help="Сколько ближайших узлов получает долю capacity")
    ap.add_argument("--out", default=None, help="Куда сохранить (по умолчанию перезапишет --nodes)")
    args = ap.parse_args()

    # исключения по именам
    exclude_norm = {_norm_name(s) for s in (args.exclude_names.split(",") if args.exclude_names else []) if s.strip()}

    # узлы
    nodes = pd.read_parquet(args.nodes)
    required = {"block_id", "slot_id", "posx", "posy"}
    if not required.issubset(nodes.columns):
        missing = ", ".join(sorted(required - set(nodes.columns)))
        raise SystemExit(f"В nodes_fixed.parquet нет колонок: {missing}")

    # зоны/блоки
    polys, ids = load_blocks_geojson(args.blocks)
    if not polys:
        raise SystemExit("В --blocks не найдено ни одного полигона")
    tree = STRtree(polys)
    id_by_geom = {g: i for i, g in enumerate(polys)}
    bid_by_geom = {g: ids[i] for i, g in enumerate(polys)}

    # CRS
    tf = Transformer.from_crs(f"EPSG:{args.services_epsg}", f"EPSG:{args.target_epsg}", always_xy=True)

    # сервисы
    cap_fields = [s.strip() for s in args.cap_fields.split(",") if s.strip()]
    services = load_services_and_project(
        args.services, transformer=tf, name_field=args.name_field,
        cap_fields=cap_fields, exclude_names_norm=exclude_norm
    )

    # world->canonical per block
    can_map: Dict[str, Tuple[ShpPolygon, Callable]] = {}
    for g, i in id_by_geom.items():
        try:
            _, fwd, _ = canonicalize_polygon(g)
            can_map[ids[i]] = (g, fwd)
        except Exception:
            pass

    # индексы узлов по блоку
    nodes_by_blk: Dict[str, pd.DataFrame] = {
        str(b): df.sort_values("slot_id").reset_index(drop=True)
        for b, df in nodes.groupby(nodes["block_id"].astype(str))
    }

    acc_present: Dict[Tuple[str, int], set] = {}
    acc_caps: Dict[Tuple[str, int, str], float] = {}

    matched, total = 0, 0
    for sv in services:
        total += 1
        pt = sv["pt"]
        # Shapely 2.x умеет предикаты прямо в query — воспользуемся, а при 1.x сделаем fallback
        host = None
        try:
            cands = tree.query(pt, predicate="contains")  # shapely 2.x
            cand_geoms = _as_geom_list(cands, polys)
            if cand_geoms:
                host = cand_geoms[0]
        except TypeError:
            # shapely 1.x — нет predicate
            cands = tree.query(pt)
            cand_geoms = _as_geom_list(cands, polys)
            for cand in cand_geoms:
                if cand.contains(pt):
                    host = cand
                    break

        # 2) Если никто не содержит — берём ближайший (и тут ТОЖЕ разворачиваем в геометрии)
        if host is None:
            cands = tree.query(pt)
            cand_geoms = _as_geom_list(cands, polys)
            polys_iter = cand_geoms if cand_geoms else polys  # fallback — все полигоны
            host = min(polys_iter, key=lambda P: P.distance(pt))
        if host is None:
            host = min(polys, key=lambda P: P.distance(pt))
        blk_id = bid_by_geom.get(host)
        if blk_id not in can_map or blk_id not in nodes_by_blk:
            continue

        matched += 1
        _, fwd = can_map[blk_id]
        x_can, y_can = fwd((pt.x, pt.y))

        df = nodes_by_blk[blk_id]
        if df.empty:
            continue
        P = df[["posx", "posy"]].to_numpy(dtype=float)
        d = np.sqrt(np.maximum(0.0, (P[:, 0] - x_can) ** 2 + (P[:, 1] - y_can) ** 2))
        idx, w = soft_weights(d, min(args.topk, len(P)))
        for j, wj in zip(idx, w):
            slot = int(df.iloc[int(j)]["slot_id"])
            key = (blk_id, slot)
            acc_present.setdefault(key, set()).add(sv["name"])
            cap_key = (blk_id, slot, sv["name"])
            acc_caps[cap_key] = acc_caps.get(cap_key, 0.0) + float(sv["value"]) * float(wj)

    # сбор новых колонок
    sp_col, sc_col = [], []
    idx_tuples = list(zip(nodes["block_id"].astype(str), nodes["slot_id"].astype(int)))
    for key in idx_tuples:
        names = sorted(list(acc_present.get(key, set())))
        caps_by_name: Dict[str, float] = {}
        for (b, s, n), v in acc_caps.items():
            if (b, s) == key and v > 0:
                caps_by_name[n] = caps_by_name.get(n, 0.0) + float(v)
        caps_list = [{"name": n, "value": float(v)} for n, v in caps_by_name.items()]
        sp_col.append(json.dumps(names, ensure_ascii=False))
        sc_col.append(json.dumps(caps_list, ensure_ascii=False))

    nodes_out = nodes.copy()
    nodes_out["services_present"] = sp_col
    nodes_out["services_capacity"] = sc_col

    out_path = args.out or args.nodes
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nodes_out.to_parquet(out_path, index=False)

    uniq_names = {n for (_, _, n) in acc_caps.keys()}
    print(f"[OK] Saved → {out_path}")
    print(f"Services total (after exclude): {total}, matched to blocks: {matched}, unique attached types: {len(uniq_names)}")
    if exclude_norm:
        print("[info] Excluded names:", ", ".join(sorted(exclude_norm)))

if __name__ == "__main__":
    main()
