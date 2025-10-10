#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
skeleton_branches.py — построение ветвей квартала

Содержимое:
- polygon_to_skgeom / build_straight_skeleton: обёртки над skgeom (fallback в вызывающем коде при ошибке).
- skeleton_to_graph: перевод straight skeleton → networkx.Graph.
- extract_branches: извлечение ветвей как LineString по путям между узлами степени != 2.
- simplify_branches_if_needed: опциональное упрощение ветвей (Douglas–Peucker).
- filter_short_branches: фильтрация ветвей по минимальной длине.
"""
from __future__ import annotations

from typing import List

import math
import networkx as nx
from shapely.geometry import LineString, Polygon


# ----------------- skgeom wrappers -----------------

def polygon_to_skgeom(poly: Polygon):
    """Shapely Polygon → skgeom.Polygon (внешнее кольцо по часовой стрелке).
    skgeom ожидает CW-ориентацию внешнего контура.
    """
    import skgeom as sg  # локальный импорт: модуль может отсутствовать

    ext = list(poly.exterior.coords)[:-1]
    ext.reverse()  # CW
    return sg.Polygon(ext)


def build_straight_skeleton(poly: Polygon):
    """Построить interior straight skeleton через skgeom."""
    import skgeom as sg  # локальный импорт

    skpoly = polygon_to_skgeom(poly)
    return sg.skeleton.create_interior_straight_skeleton(skpoly)


# ----------------- skeleton → graph -----------------

def skeleton_to_graph(skel) -> nx.Graph:
    """skgeom skeleton → networkx.Graph (веса рёбер = длины сегментов)."""
    G = nx.Graph()
    for v in skel.vertices:
        G.add_node(v.id, x=float(v.point.x()), y=float(v.point.y()), time=float(v.time))
    for h in skel.halfedges:
        if not h.is_bisector:
            continue
        u, v = h.vertex.id, h.opposite.vertex.id
        if u == v:
            continue
        p1, p2 = h.vertex.point, h.opposite.vertex.point
        w = math.hypot(float(p1.x()) - float(p2.x()), float(p1.y()) - float(p2.y()))
        G.add_edge(u, v, weight=w)
    return G


# ----------------- extract branches -----------------

def extract_branches(G: nx.Graph) -> List[LineString]:
    """Собрать ветви как линейные цепочки между узлами степени != 2.
    Если получилась пустота — вернуть одну LineString по порядку узлов графа (fallback).
    """
    branches: List[LineString] = []
    used = set()
    degs = dict(G.degree())
    junctions = {n for n, d in degs.items() if d != 2}

    for s in junctions:
        for n in G.neighbors(s):
            ekey = tuple(sorted((s, n)))
            if ekey in used:
                continue
            path = [s, n]
            used.add(ekey)
            prev, cur = s, n
            while G.degree[cur] == 2:
                nb = [x for x in G.neighbors(cur) if x != prev][0]
                used.add(tuple(sorted((cur, nb))))
                path.append(nb)
                prev, cur = cur, nb
            coords = [(G.nodes[i]["x"], G.nodes[i]["y"]) for i in path]
            if len(coords) >= 2:
                branches.append(LineString(coords))

    if not branches and len(G.nodes):
        coords = [(G.nodes[i]["x"], G.nodes[i]["y"]) for i in G.nodes]
        try:
            branches = [LineString(coords)]
        except Exception:
            pass
    return branches


# ----------------- utils: simplify / filter -----------------

def simplify_branches_if_needed(branches: List[LineString], tol: float | None) -> List[LineString]:
    """Вернуть упрощённые ветви при tol>0; иначе вернуть исходные.
    Используется Shapely LineString.simplify(preserve_topology=False).
    """
    if not branches:
        return branches
    if tol is None or tol <= 0:
        return branches
    out: List[LineString] = []
    for br in branches:
        try:
            s = br.simplify(tol, preserve_topology=False)
            if isinstance(s, LineString) and s.length > 0:
                out.append(s)
        except Exception:
            out.append(br)
    return out


def filter_short_branches(branches: List[LineString], min_length: float | None) -> List[LineString]:
    """Отфильтровать ветви короче порога; при min_length<=0 вернуть исходный список."""
    if not branches:
        return branches
    if min_length is None or min_length <= 0:
        return branches
    return [br for br in branches if isinstance(br, LineString) and br.length >= float(min_length)]


__all__ = [
    "polygon_to_skgeom",
    "build_straight_skeleton",
    "skeleton_to_graph",
    "extract_branches",
    "simplify_branches_if_needed",
    "filter_short_branches",
]
