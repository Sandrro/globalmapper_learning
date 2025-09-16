"""
geo_common.py — общие гео-утилиты для пайплайна иерархической канонизации

Содержимое:
- load_vector: чтение GeoDataFrame из Parquet/векторных форматов.
- WKB совместимость Shapely 1.x/2.x: geom_to_wkb / geom_from_wkb.
- ensure_valid_series: починка геометрий (make_valid | buffer(0)).
- normalize_crs_str, to_metric_crs: нормализация и преобразование CRS.
- largest_polygon: выбор наибольшего полигона из сложной геометрии.
- _dist, get_size_with_vector: геометрические утилиты (минимальный охватывающий прямоугольник).
- save_block_mask_64: рендер бинарной маски квартала в PNG 64×64 (с отверстиями).
- block_scale_l: оценка масштаба L для квартала (макс. длина ветви, иначе MRR).

Модуль не зависит от конкретных данных и может переиспользоваться в скриптах
assign / воркере / мердже.
"""
from __future__ import annotations

import math
import os
from typing import Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import shapely
from PIL import Image, ImageDraw
from shapely.geometry import LineString, MultiPolygon, Polygon

__all__ = [
    "load_vector",
    "geom_to_wkb",
    "geom_from_wkb",
    "ensure_valid_series",
    "normalize_crs_str",
    "to_metric_crs",
    "largest_polygon",
    "get_size_with_vector",
    "save_block_mask_64",
    "block_scale_l",
]

# ----------------- I/O -----------------

def load_vector(path: str) -> gpd.GeoDataFrame:
    """Загрузить геоданные из файла. Если .parquet — используем read_parquet, иначе read_file.

    Parameters
    ----------
    path : str
        Путь к векторному файлу (Parquet/GeoPackage/GeoJSON/Shapefile и т.п.).

    Returns
    -------
    GeoDataFrame
    """
    if path.lower().endswith(".parquet"):
        return gpd.read_parquet(path)
    return gpd.read_file(path)


# --- Shapely 1.x/2.x совместимость для WKB ---

def geom_to_wkb(geom) -> bytes:
    """Сериализовать геометрию в WKB, поддержка Shapely 1/2."""
    try:
        return shapely.to_wkb(geom)  # Shapely 2
    except Exception:
        from shapely import wkb
        return wkb.dumps(geom)


def geom_from_wkb(buf: bytes):
    """Десериализовать WKB в геометрию, поддержка Shapely 1/2."""
    try:
        return shapely.from_wkb(buf)  # Shapely 2
    except Exception:
        from shapely import wkb
        return wkb.loads(buf)


# ----------------- GEO UTILS -----------------

def ensure_valid_series(geos: gpd.GeoSeries) -> gpd.GeoSeries:
    """Попытаться сделать геометрии валидными: make_valid (Shapely 2) или buffer(0) (fallback)."""
    try:
        return shapely.make_valid(geos)  # Shapely 2
    except Exception:
        # Shapely 1.x fallback (may be slow, but robust)
        return geos.buffer(0)


def normalize_crs_str(s: str | int) -> str:
    """Привести значение CRS к строке формата 'EPSG:XXXX'."""
    s = str(s)
    return s if s.upper().startswith("EPSG:") else f"EPSG:{s}"


def to_metric_crs(gdf: gpd.GeoDataFrame, target_crs: str) -> gpd.GeoDataFrame:
    """Преобразовать слой в указанный метрический CRS (если отличается)."""
    if gdf.crs is None:
        raise ValueError("У слоя отсутствует CRS.")
    if str(gdf.crs) != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


# ---------- базовая геометрия ----------

def _dist(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    return math.hypot(bx - ax, by - ay)


def largest_polygon(geom) -> Optional[Polygon]:
    """Вернуть крупнейший Polygon из произвольной геометрии, либо None.

    * Polygon → сам.
    * MultiPolygon → компонент с макс. площадью.
    * Иное → попытка взять convex_hull, если он Polygon.
    """
    if geom is None or getattr(geom, "is_empty", True):
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


def get_size_with_vector(poly: Polygon) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Вернуть (longside, shortside, long_vec, short_vec) из minimum_rotated_rectangle.

    Векторы ориентированы по сторонам MRR (в мировых координатах), нормализация не выполняется.
    """
    mrr = poly.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])
    if axis1 <= axis2:
        return (
            float(axis2),
            float(axis1),
            np.asarray(bbox[0]) - np.asarray(bbox[1]),
            np.asarray(bbox[0]) - np.asarray(bbox[3]),
        )
    else:
        return (
            float(axis1),
            float(axis2),
            np.asarray(bbox[0]) - np.asarray(bbox[3]),
            np.asarray(bbox[0]) - np.asarray(bbox[1]),
        )


# ---------- Маска квартала и масштаб ----------

def save_block_mask_64(poly: Polygon, out_path: str, size: int = 64) -> bool:
    """Сохранить бинарную маску квартала в PNG size×size. Возвращает True, если удалось.

    Отрисовывает внешнюю границу (белым) и отверстия (чёрным). Если poly некорректен — False.
    """
    poly = largest_polygon(poly)
    if poly is None:
        return False

    minx, miny, maxx, maxy = poly.bounds

    def w2p(x: float, y: float) -> Tuple[float, float]:
        # world → pixel (y инвертируем, чтобы верх был 0)
        u = (x - minx) / max(maxx - minx, 1e-9) * (size - 1)
        v = (y - miny) / max(maxy - miny, 1e-9) * (size - 1)
        return (u, (size - 1) - v)

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    # exterior
    ext = [w2p(x, y) for (x, y) in list(poly.exterior.coords)]
    draw.polygon(ext, fill=255)
    # holes
    for ring in poly.interiors:
        ints = [w2p(x, y) for (x, y) in list(ring.coords)]
        draw.polygon(ints, fill=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return True


def block_scale_l(poly: Polygon, branches: Optional[List[LineString]] = None) -> float:
    """Вернуть характерный масштаб L квартала.

    Предпочитаем максимальную длину осмысленной ветви (если задана), иначе —
    большую сторону minimum_rotated_rectangle.
    """
    branches = branches or []
    if branches:
        L = max((float(br.length) for br in branches if isinstance(br, LineString)), default=0.0)
        if L > 0:
            return L
    mrr = poly.minimum_rotated_rectangle
    bbox = list(mrr.exterior.coords)
    axis1 = _dist(bbox[0], bbox[3])
    axis2 = _dist(bbox[0], bbox[1])
    return float(max(axis1, axis2))
