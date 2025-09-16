"""
canon_features.py — канонические фичи по ветви

Назначение:
- Векторизация ветви квартала в параметр s∈[0,1] и набор локальных направлений сегментов.
- Проекция зданий в каноническую систему координат ветви: pos(x,y), size_x/size_y, угол phi.
- Оценка аспект-отношения блока относительно ветви.
- Простая морфологическая классификация формы здания (Rect/L/U/X).

Совместимо с ранее данным train.py (ожидаемые признаки: pos/size/phi/s_i/a_i/aspect_ratio).
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from geo_common import get_size_with_vector

__all__ = [
    "branch_vectors",
    "insert_pos",
    "canon_by_branch",
    "occupancy_ratio",
    "reflex_count",
    "classify_shape",
]

# ----------------- ВЕТВЬ → ПАРАМЕТРИЗАЦИЯ -----------------

def _as_np_xy(seq: Sequence[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(seq, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr

def branch_vectors(branch: LineString) -> Tuple[np.ndarray, np.ndarray]:
    """Вернуть (rel_cuts, seg_vecs) по ломаной ``branch``.

    rel_cuts : np.ndarray, shape (M+1,)
        Накопленные доли длины вдоль линии в диапазоне [0,1], включая 0 и 1.
    seg_vecs : np.ndarray, shape (M,2)
        Вектор направления каждого сегмента (в мировых координатах).
    """
    coords = _as_np_xy(list(branch.coords))  # (K,2)
    if coords.shape[0] < 2:
        return np.array([0.0, 1.0], dtype=float), np.array([[1.0, 0.0]], dtype=float)

    # длины сегментов
    deltas = coords[1:] - coords[:-1]  # (M,2)
    seg_len = np.linalg.norm(deltas, axis=1)  # (M,)
    total = float(seg_len.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.array([0.0, 1.0], dtype=float), np.array([[1.0, 0.0]], dtype=float)

    cuts = np.concatenate([[0.0], np.cumsum(seg_len) / total])  # (M+1,)
    return cuts, deltas


def insert_pos(rel_cuts: np.ndarray, s: float) -> int:
    """Индекс позиции вставки для параметра ``s`` (аналог bisect_left).

    Возвращает i, такое что rel_cuts[i-1] <= s < rel_cuts[i].
    """
    s = float(np.clip(s, 0.0, 1.0))
    return int(np.searchsorted(rel_cuts, s, side="left"))


# ----------------- ПРОЕКЦИЯ ЗДАНИЙ В КАНОН -----------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def _angle_signed(a: np.ndarray, b: np.ndarray) -> float:
    """Малая ориентированная разница углов между векторами a и b ([-pi/2, +pi/2])."""
    a = _unit(a); b = _unit(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    raw = min(math.pi - math.acos(dot), math.acos(dot))  # ∈ [0, π/2]
    z = a[0]*b[1] - a[1]*b[0]
    return raw if z >= 0 else -raw


def _axis_at_s(seg_vecs: np.ndarray, rel_cuts: np.ndarray, s: float) -> np.ndarray:
    idx = max(0, min(len(seg_vecs) - 1, insert_pos(rel_cuts, s) - 1))
    return seg_vecs[idx]


def _point_on(branch: LineString, s: float) -> Point:
    return branch.interpolate(s, normalized=True)


def canon_by_branch(
    bldg_polys: List[Polygon],
    block_poly: Polygon,
    branch: LineString,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Канонизация зданий относительно ветви.

    Параметры
    ---------
    bldg_polys : list[Polygon]
        Геометрии зданий внутри квартала.
    block_poly : Polygon
        Полигон квартала (необходим для оценки масштаба/аспекта).
    branch : LineString
        Выбранная «ось» квартала.

    Возврат
    ------
    pos : (N,2)  — x∈[-1,1] — позиция вдоль ветви; y — поперёк (≈двойная нормализация шириной).
    size: (N,2)  — (длина вдоль ветви, ширина поперёк), нормированные.
    phi : (N,)   — ориентированная разница углов (радианы) длинной оси здания к оси ветви.
    aspect: (N,) — одинаковое для всех зданий значение mean_width / L ветви (как в transform.py).
    """
    N = len(bldg_polys)
    if N == 0:
        return (
            np.zeros((0, 2), dtype=float),
            np.zeros((0, 2), dtype=float),
            np.zeros((0,), dtype=float),
            np.zeros((0,), dtype=float),
        )

    rel_cuts, seg_vecs = branch_vectors(branch)

    # Оценка масштаба: длина ветви и средняя ширина квартала.
    L = max(float(branch.length), 1e-6)
    blk_long, blk_short, _, _ = get_size_with_vector(block_poly)
    mean_width = float(blk_short)  # поперечная характерная ширина
    aspect = float(mean_width) / float(L)

    pos = np.zeros((N, 2), dtype=float)
    size = np.zeros((N, 2), dtype=float)
    phi = np.zeros((N,), dtype=float)
    aspect_arr = np.full((N,), aspect, dtype=float)

    for i, poly in enumerate(bldg_polys):
        # 1) x — доля вдоль ветви
        c = poly.centroid
        s = float(branch.project(c, normalized=True))  # ∈[0,1]
        pos[i, 0] = 2.0 * s - 1.0

        # 2) локальная ось и поперечная компонента
        axis_vec = _axis_at_s(seg_vecs, rel_cuts, s)
        axis_u = _unit(axis_vec)
        pt_on = _point_on(branch, s)
        v_to = np.array([c.x - pt_on.x, c.y - pt_on.y], dtype=float)
        sign = 1.0 if (axis_u[0]*v_to[1] - axis_u[1]*v_to[0]) >= 0 else -1.0
        dist = float(np.linalg.norm(v_to))
        pos[i, 1] = sign * (2.0 * dist / max(mean_width, 1e-9))

        # 3) размер здания (по MRR здания)
        longside, shortside, long_vec, short_vec = get_size_with_vector(poly)
        # выбираем настоящую «длинную» ось как ту, что ближе к axis_u
        a1 = abs(float(np.dot(_unit(long_vec), axis_u)))
        a2 = abs(float(np.dot(_unit(short_vec), axis_u)))
        if a2 > a1:  # менять местами
            longside, shortside = shortside, longside
            long_vec = short_vec
        size[i, 0] = 2.0 * float(longside) / L
        size[i, 1] = 2.0 * float(shortside) / max(mean_width, 1e-9)

        # 4) phi — ориентированная малая разница углов между длинной осью и осью ветви
        phi[i] = _angle_signed(long_vec, axis_u)

    return pos, size, phi, aspect_arr


# ----------------- ПРОСТАЯ МОРФОКЛАССИФИКАЦИЯ -----------------

def occupancy_ratio(poly: Polygon) -> float:
    """Отношение площади к площади MRR (≈ заполняемость)."""
    mrr = poly.minimum_rotated_rectangle
    return float(poly.area) / max(float(mrr.area), 1e-9)


def reflex_count(poly: Polygon) -> int:
    """Число рефлекс-углов по внешнему контуру (оценка «изломанности» формы)."""
    coords = list(poly.exterior.coords)[:-1]
    n = len(coords)
    cnt = 0
    for i in range(n):
        p_prev = np.asarray(coords[(i - 1) % n], dtype=float)
        p = np.asarray(coords[i], dtype=float)
        p_next = np.asarray(coords[(i + 1) % n], dtype=float)
        v1 = p - p_prev
        v2 = p_next - p
        z = v1[0]*v2[1] - v1[1]*v2[0]
        if z < 0:
            cnt += 1
    return int(cnt)


def classify_shape(poly: Polygon) -> int:
    """Грубая классификация формы здания: 0=Rect, 1=L, 2=U, 3=X.

    Критерии эмпирические и быстрые:
    - если фигура почти выпуклая и occupancy >= 0.8 → Rect
    - если есть выемки/отверстия (interiors) ИЛИ occupancy < 0.6 при небольшом числе рефлекс-углов → U
    - если рефлекс-углов много (>=4) → X
    - иначе → L
    """
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
