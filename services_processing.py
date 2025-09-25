from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

DEFAULT_CAPACITY_FIELDS: Tuple[str, ...] = ("capacity", "capacity_modeled", "weight_value")
DEFAULT_EXCLUDED_SERVICES: Tuple[str, ...] = (
    "Парк",
    "Спортивная площадка",
    "Детская площадка",
)
HAS_PREFIX = "has_service__"
CAPACITY_PREFIX = "service_capacity__"
SCHEMA_VERSION = 1

SchemaItem = Dict[str, str]

__all__ = [
    "DEFAULT_CAPACITY_FIELDS",
    "DEFAULT_EXCLUDED_SERVICES",
    "HAS_PREFIX",
    "CAPACITY_PREFIX",
    "attach_services_to_buildings",
    "write_service_schema",
    "load_service_schema",
    "infer_service_schema_from_nodes",
]


def _normalize_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name).strip()).lower()
    s = s.replace("\xa0", " ")
    s = re.sub(r"[\s/\\]+", "_", s)
    s = re.sub(r"[^0-9a-zA-Zа-яё_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "service"


def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None or getattr(geom, "is_empty", True):
        return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        geoms = list(geom.geoms)
        if not geoms:
            return None
        return max(geoms, key=lambda g: g.area if g is not None else 0.0)
    return None


def _extract_capacity(row: Mapping[str, object], fields: Sequence[str]) -> float:
    for field in fields:
        if field in row and row[field] is not None:
            try:
                return float(row[field])
            except Exception:
                continue
    return 1.0


def _service_excluded(name: str, exclude: Iterable[str]) -> bool:
    norm = str(name).strip().casefold()
    return norm in {str(v).strip().casefold() for v in exclude}


def attach_services_to_buildings(
    bldgs: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
    capacity_fields: Optional[Sequence[str]] = None,
    exclude_names: Optional[Sequence[str]] = None,
) -> Tuple[gpd.GeoDataFrame, List[SchemaItem]]:
    """Augment building GeoDataFrame with per-service columns and synthetic rows.

    Returns the updated buildings GeoDataFrame **copy** and the ordered schema
    describing generated columns.  The schema keeps the original service names.
    """

    if services is None or services.empty:
        return bldgs, []

    cap_fields = [c for c in (capacity_fields or DEFAULT_CAPACITY_FIELDS) if str(c).strip()]
    if not cap_fields:
        cap_fields = list(DEFAULT_CAPACITY_FIELDS)

    excluded = tuple(exclude_names or DEFAULT_EXCLUDED_SERVICES)

    svc = services.copy()
    if "geometry" not in svc:
        return bldgs, []
    svc = svc[svc.geometry.notna()]
    svc = svc[~svc.geometry.is_empty]
    if svc.empty:
        return bldgs, []

    if "service_type_name" not in svc.columns:
        return bldgs, []

    bdf = bldgs.copy()
    valid_mask = bdf.geometry.notna() & (~bdf.geometry.is_empty)
    bdf_valid = bdf.loc[valid_mask]
    sindex = bdf_valid.sindex if not bdf_valid.empty else None

    # service aggregations
    aggregated: Dict[int, Dict[str, float]] = {}
    new_rows: List[Tuple[Polygon, Dict[str, float]]] = []
    service_names: List[str] = []

    index_labels = list(bdf_valid.index)

    for row in svc.itertuples(index=False):
        name = str(getattr(row, "service_type_name", "")).strip()
        if name == "" or _service_excluded(name, excluded):
            continue
        geom = getattr(row, "geometry", None)
        if geom is None or getattr(geom, "is_empty", True):
            continue
        cap_val = _extract_capacity(row._asdict(), cap_fields)
        hits: List[int] = []
        if sindex is not None:
            try:
                cand = sindex.query(geom, predicate="intersects")  # type: ignore[assignment]
            except Exception:
                cand = sindex.intersection(geom.bounds)  # type: ignore[assignment]
            cand = list(np.asarray(cand))
            for idx in cand:
                try:
                    idx_label = index_labels[int(idx)]
                except Exception:
                    idx_label = idx
                try:
                    b_geom = bdf_valid.loc[idx_label, "geometry"]
                except KeyError:
                    continue
                if b_geom is None or getattr(b_geom, "is_empty", True):
                    continue
                try:
                    intersects = bool(geom.intersects(b_geom))
                except Exception:
                    intersects = False
                if intersects:
                    hits.append(int(idx_label))
        if hits:
            for hid in hits:
                aggregated.setdefault(hid, {})[name] = aggregated.setdefault(hid, {}).get(name, 0.0) + cap_val
        else:
            poly = _largest_polygon(geom)
            if poly is None:
                continue
            new_rows.append((poly, {name: cap_val}))
        if name not in service_names:
            service_names.append(name)

    if not service_names and not new_rows:
        return bdf, []

    # stable order: sort by original name
    service_names = sorted(service_names)
    schema: List[SchemaItem] = []
    lookup: Dict[str, SchemaItem] = {}
    for name in service_names:
        slug = _normalize_name(name)
        has_col = f"{HAS_PREFIX}{slug}"
        cap_col = f"{CAPACITY_PREFIX}{slug}"
        item: SchemaItem = {
            "name": name,
            "slug": slug,
            "has_column": has_col,
            "capacity_column": cap_col,
        }
        schema.append(item)
        lookup[name] = item
        bdf[has_col] = False
        bdf[cap_col] = 0.0

    for idx, values in aggregated.items():
        for name, cap in values.items():
            item = lookup.get(name)
            if item is None:
                continue
            has_col = item["has_column"]
            cap_col = item["capacity_column"]
            bdf.at[idx, has_col] = True
            prev = bdf.at[idx, cap_col] if cap_col in bdf.columns else 0.0
            try:
                prev_f = float(prev)
            except Exception:
                prev_f = 0.0
            bdf.at[idx, cap_col] = prev_f + float(cap)

    if new_rows:
        base_cols = [c for c in bdf.columns if c != "geometry"]
        new_data: List[Dict[str, object]] = []
        new_geoms: List[Polygon] = []
        for geom, values in new_rows:
            row_dict: Dict[str, object] = {col: pd.NA for col in base_cols}
            if "floors_num" in row_dict:
                row_dict["floors_num"] = 0
            if "living_area" in row_dict:
                row_dict["living_area"] = 0.0
            if "has_floors" in row_dict:
                row_dict["has_floors"] = False
            if "is_living" in row_dict:
                row_dict["is_living"] = False
            for item in schema:
                row_dict[item["has_column"]] = False
                row_dict[item["capacity_column"]] = 0.0
            for name, cap in values.items():
                item = lookup.get(name)
                if item is None:
                    continue
                row_dict[item["has_column"]] = True
                row_dict[item["capacity_column"]] = float(cap)
            new_data.append(row_dict)
            new_geoms.append(geom)
        new_df = gpd.GeoDataFrame(new_data, geometry=new_geoms, crs=bdf.crs)
        bdf = pd.concat([bdf, new_df], ignore_index=True)

    return bdf, schema


def write_service_schema(schema: Sequence[SchemaItem], out_path: str) -> None:
    if not out_path or not schema:
        return
    data = {
        "version": SCHEMA_VERSION,
        "services": [dict(item) for item in schema],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_service_schema(path: str) -> List[SchemaItem]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    items = data.get("services") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    schema: List[SchemaItem] = []
    for obj in items:
        if not isinstance(obj, Mapping):
            continue
        name = obj.get("name")
        has_col = obj.get("has_column")
        cap_col = obj.get("capacity_column")
        if not name or not has_col or not cap_col:
            continue
        schema.append(
            {
                "name": str(name),
                "slug": _normalize_name(obj.get("slug", name)),
                "has_column": str(has_col),
                "capacity_column": str(cap_col),
            }
        )
    return schema


def infer_service_schema_from_nodes(nodes: Mapping | Sequence[str]) -> List[SchemaItem]:
    if isinstance(nodes, Mapping):
        columns = list(nodes.keys())
    elif isinstance(nodes, pd.DataFrame):
        columns = list(nodes.columns)
    else:
        columns = list(nodes)
    schema: List[SchemaItem] = []
    seen = set()
    for col in columns:
        if not isinstance(col, str) or not col.startswith(HAS_PREFIX):
            continue
        slug = col[len(HAS_PREFIX) :]
        cap_col = f"{CAPACITY_PREFIX}{slug}"
        if cap_col not in columns or slug in seen:
            continue
        seen.add(slug)
        schema.append(
            {
                "name": slug,
                "slug": slug,
                "has_column": col,
                "capacity_column": cap_col,
            }
        )
    schema.sort(key=lambda item: item["name"])
    return schema
