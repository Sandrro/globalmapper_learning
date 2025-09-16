#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_graphs_cli.py — визуализация получившихся графов по зонам

Задача:
  - Для каждой функциональной зоны выбрать квартал (block), где максимум активных узлов (e_i==1).
  - Нарисовать граф этого квартала:
      * координаты узлов — канон-пространство (posx,posy) из nodes_fixed.parquet;
      * размер узла — по living_area (робастный скейлинг);
      * цвет узла — по сервисам; если сервисов несколько, рисуем круговую диаграмму (pie) в узле.
  - Добавить легенду по сервисам и примерную шкалу размера (жилая площадь).

Вход:
  - nodes_fixed.parquet, edges.parquet, blocks.parquet из папки результатов (out/),
  - services.json (опц.): карта service_name -> vector_index, чтобы зафиксировать порядок и палитру.

Пример запуска:
  python visualize_graphs_cli.py \
    --dataset-dir out \
    --services-json out/services.json \
    --out-dir out/viz \
    --dpi 200

Выход:
  - PNG по одной картинке на зону: out/viz/zone_<zone>__block_<block_id>.png
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd


# ----------------- utils -----------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _load_services(services_json: Optional[str], nodes: pd.DataFrame) -> Tuple[List[str], Dict[str, int]]:
    """Вернём (ordered_services, name→index). Если services.json не задан, инферим по данным.
    Поддерживаем два формата services_present в nodes:
      - список имён сервисов (['school','shop',...]) — предпочтительный;
      - бинарный вектор 0/1 той же длины, что словарь (как в сырье) — тогда даём имена 'svc_<idx>'.
    """
    if services_json and os.path.exists(services_json):
        with open(services_json, "r", encoding="utf-8") as f:
            mp = json.load(f)  # name -> index
        # Отсортируем по index
        items = sorted(((name, idx) for name, idx in mp.items()), key=lambda t: int(t[1]))
        names = [k for k, _ in items]
        name2idx = {k: i for i, k in enumerate(names)}
        return names, name2idx

    # Фолбэк: собрать имена из данных
    names_set: Dict[str, int] = {}
    for x in nodes.get("services_present", []):
        if isinstance(x, str):
            try:
                x = json.loads(x)
            except Exception:
                x = []
        if isinstance(x, (list, tuple)):
            if x and isinstance(x[0], (int, float)):
                # бинарный вектор 0/1: сопоставим индексы с псевдонимами
                for i, v in enumerate(x):
                    try:
                        if float(v) > 0:
                            names_set.setdefault(f"svc_{i}", i)
                    except Exception:
                        pass
            else:
                for name in x:
                    names_set.setdefault(str(name), len(names_set))
    if not names_set:
        # нет сервисов — вернём пустой
        return [], {}
    # Отсортировать по индексу вставки
    ordered = sorted(names_set.items(), key=lambda t: t[1])
    names = [k for k, _ in ordered]
    return names, {k: i for i, k in enumerate(names)}


def _extract_services_present(row_services: Any, name2idx: Dict[str, int]) -> List[int]:
    """Вернуть список индексов сервисов для узла.
    Поддержка форматов: список имён; список 0/1; JSON-строка.
    """
    x = row_services
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return []
    if isinstance(x, (list, tuple)):
        if not x:
            return []
        if isinstance(x[0], (int, float)):
            # 0/1 вектор
            idxs = [i for i, v in enumerate(x) if (isinstance(v, (int, float)) and float(v) > 0)]
            # Отфильтруем по известной карте, если задана
            if name2idx:
                kmax = max(name2idx.values())
                return [i for i in idxs if 0 <= i <= kmax]
            return idxs
        else:
            # список имён
            out = []
            for name in x:
                if str(name) in name2idx:
                    out.append(name2idx[str(name)])
            return out
    return []


def _node_sizes(living_area: pd.Series, min_size: float = 50.0, max_size: float = 800.0) -> np.ndarray:
    """Робастный мэппинг площади в пиксели через квантили + sqrt.
    Возвращает массив размеров для scatter (area in points^2).
    """
    la = pd.to_numeric(living_area, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if la.size == 0:
        return np.array([])
    q1, q9 = np.quantile(la, [0.1, 0.9]) if np.any(la > 0) else (0.0, 1.0)
    a = np.clip(la, q1, q9)
    a = np.sqrt(a - q1 + 1e-9)
    if a.max() <= 0:
        return np.full_like(a, min_size)
    a = (a - a.min()) / (a.max() - a.min())
    return min_size + a * (max_size - min_size)


def _make_palette(n: int) -> List[Tuple[float, float, float, float]]:
    """Палитра цветов для сервисов из табличных colormap (tab20)."""
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(max(1, n))]
    return colors


def _draw_pie_node(ax, x: float, y: float, parts: Sequence[float], colors: Sequence[Any], radius: float) -> None:
    total = float(sum(parts)) if parts else 0.0
    if total <= 0:
        circ = mpatches.Circle((x, y), radius=radius, fill=False, ec="black", lw=0.6)
        ax.add_patch(circ)
        return
    start = 0.0
    for frac, col in zip(parts, colors):
        if frac <= 0:
            continue
        theta1 = start * 360.0
        theta2 = (start + frac / total) * 360.0
        wedge = mpatches.Wedge((x, y), radius, theta1, theta2, facecolor=col, edgecolor="white", linewidth=0.4)
        ax.add_patch(wedge)
        start += frac / total
    # тонкая чёрная окантовка
    circ = mpatches.Circle((x, y), radius=radius, fill=False, ec="black", lw=0.6)
    ax.add_patch(circ)


# ----------------- main -----------------

def main() -> int:
    ap = argparse.ArgumentParser("Визуализация графов по зонам")
    ap.add_argument("--dataset-dir", required=True, help="Папка с parquet-файлами (nodes_fixed.parquet, edges.parquet, blocks.parquet)")
    ap.add_argument("--services-json", default=None, help="Файл services.json (name→index), опционально")
    ap.add_argument("--out-dir", required=True, help="Куда класть PNG")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--node-radius", type=float, default=0.02, help="Базовый радиус круга для pie-узлов (в долях оси)")
    ap.add_argument("--max-zones", type=int, default=None, help="Ограничить число зон для вывода")
    args = ap.parse_args()

    nodes_path = os.path.join(args.dataset_dir, "nodes_fixed.parquet")
    edges_path = os.path.join(args.dataset_dir, "edges.parquet")
    blocks_path = os.path.join(args.dataset_dir, "blocks.parquet")

    if not (os.path.exists(nodes_path) and os.path.exists(edges_path) and os.path.exists(blocks_path)):
        raise FileNotFoundError("Ожидались nodes_fixed.parquet, edges.parquet и blocks.parquet в --dataset-dir")

    _ensure_dir(args.out_dir)

    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    blocks = pd.read_parquet(blocks_path)

    # Присвоим зоне узлы через blocks
    blk_zone = blocks.set_index("block_id")["zone"].to_dict()
    nodes["zone"] = nodes["block_id"].map(blk_zone).fillna("nan")

    # Только активные узлы для подсчёта "плотности"
    nodes["e_i"] = pd.to_numeric(nodes["e_i"], errors="coerce").fillna(0).astype(int)

    # Выбор представителя для каждой зоны
    # Считаем по количеству активных узлов
    agg = nodes[nodes["e_i"] == 1].groupby(["zone", "block_id"], dropna=False).size().rename("n_active").reset_index()
    if agg.empty:
        # дополнительная диагностика
        total_nodes = len(nodes)
        active_est = int((nodes["e_i"] == 1).sum())
        print(f"Нет активных узлов (e_i==1) в dataset — нечего визуализировать. Всего узлов: {total_nodes}, активных: {active_est}. Возможно, все zone=NaN — попробуй передать --services-json и убедиться, что blocks.parquet содержит колонку zone.")
        return 0

    best_blocks = agg.sort_values(["zone", "n_active"], ascending=[True, False]).groupby("zone").head(1)

    if args.max_zones is not None:
        best_blocks = best_blocks.head(int(args.max_zones))

    # Палитра сервисов
    svc_names, name2idx = _load_services(args.services_json, nodes)
    colors = _make_palette(len(svc_names) if svc_names else 1)

    # Легенда по сервисам (proxy artists)
    legend_handles = []
    if svc_names:
        for i, name in enumerate(svc_names):
            legend_handles.append(mpatches.Patch(color=colors[i % len(colors)], label=str(name)))
    else:
        legend_handles.append(mpatches.Patch(color="lightgray", label="без сервисов"))

    # Пробежим по зонам
    for _, row in best_blocks.iterrows():
        zone = row["zone"]
        blk = row["block_id"]

        sub_nodes = nodes[nodes["block_id"] == blk].copy()
        sub_edges = edges[edges["block_id"] == blk].copy()

        # Позиции: posx,posy (уже канон-пространство)
        px = pd.to_numeric(sub_nodes["posx"], errors="coerce").fillna(0.0).to_numpy()
        py = pd.to_numeric(sub_nodes["posy"], errors="coerce").fillna(0.0).to_numpy()
        pos = {int(i): (float(x), float(y)) for i, x, y in zip(sub_nodes["slot_id"], px, py)}

        # Размеры по living_area
        sizes = _node_sizes(sub_nodes.get("living_area", pd.Series([], dtype=float)))
        # Для pie-радиуса переведём относительный radius из аргумента в координаты axes
        # Будем рисовать в осях с равными масштабами; radius берём из args.node_radius

        # Подготовка графа (для рёбер)
        G = nx.DiGraph()
        for sid in sub_nodes["slot_id"].astype(int).tolist():
            G.add_node(int(sid))
        for e in sub_edges.itertuples(index=False):
            try:
                G.add_edge(int(getattr(e, "src_slot")), int(getattr(e, "dst_slot")))
            except Exception:
                pass

        fig, ax = plt.subplots(figsize=(10, 8), dpi=args.dpi)
        ax.set_aspect("equal")
        ax.set_title(f"Zone: {zone}  |  Block: {blk}  |  Active nodes: {int(row['n_active'])}")
        ax.set_xlabel("posx")
        ax.set_ylabel("posy")

        # Рисуем рёбра как линии
        for u, v in G.edges():
            if u in pos and v in pos:
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                ax.plot([x1, x2], [y1, y2], lw=0.5, alpha=0.6, zorder=1, color="#999999")

        # Диапазон осей под данные
        if len(pos) > 0:
            X = np.array([p[0] for p in pos.values()])
            Y = np.array([p[1] for p in pos.values()])
            pad = 0.1
            ax.set_xlim(X.min() - pad, X.max() + pad)
            ax.set_ylim(Y.min() - pad, Y.max() + pad)

        # Рисуем узлы: если нет/1 сервис — круг; если >1 — pie
        # sizes сейчас в единицах points^2 у scatter. Для радиуса переведём в долю оси грубо:
        # возьмём корень из area, нормализуем относительно диагонали окна (эвристика)
        diag = math.hypot(*(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])) or 1.0
        base_r = float(args.node_radius) * diag

        # Для легенды по размерам подготовим три примера
        la_series = pd.to_numeric(sub_nodes.get("living_area", pd.Series([], dtype=float)), errors="coerce").fillna(0.0)
        if not la_series.empty:
            q_small, q_mid, q_big = np.quantile(la_series, [0.2, 0.5, 0.9])
            size_samples = [q_small, q_mid, q_big]
        else:
            size_samples = [50.0, 200.0, 800.0]

        # Нарисуем сами узлы
        for i, r in sub_nodes.iterrows():
            sid = int(r["slot_id"]) if not pd.isna(r["slot_id"]) else None
            if sid is None or sid not in pos:
                continue
            x, y = pos[sid]
            la = float(r.get("living_area", 0.0) or 0.0)
            # радиус как базовый + добавка по площади (корень)
            r_add = 0.0
            try:
                r_add = 0.5 * base_r * (math.sqrt(max(la, 0.0)) / (math.sqrt(max(size_samples[-1], 1e-6))))
            except Exception:
                pass
            radius = max(0.2 * base_r, base_r + r_add)

            svc_idxs = _extract_services_present(r.get("services_present"), name2idx)
            if not svc_idxs:
                circ = mpatches.Circle((x, y), radius=radius, fc="lightgray", ec="black", lw=0.6, zorder=3)
                ax.add_patch(circ)
            elif len(svc_idxs) == 1:
                c = colors[svc_idxs[0] % len(colors)] if colors else "lightgray"
                circ = mpatches.Circle((x, y), radius=radius, fc=c, ec="black", lw=0.6, zorder=3)
                ax.add_patch(circ)
            else:
                parts = [1.0 for _ in svc_idxs]
                cols = [colors[j % len(colors)] for j in svc_idxs]
                _draw_pie_node(ax, x, y, parts, cols, radius)

        # Легенды: сервисы
        leg1 = ax.legend(handles=legend_handles, title="Сервисы", loc="upper right")
        ax.add_artist(leg1)

        # Легенда по размерам
        size_handles = []
        for s_val in size_samples:
            rr = max(0.2 * base_r, base_r + 0.5 * base_r * (math.sqrt(max(s_val, 0.0)) / (math.sqrt(max(size_samples[-1], 1e-6)))))
            size_handles.append(mpatches.Circle((0, 0), radius=rr, fc="#DDDDDD", ec="black", lw=0.6))
        leg2 = ax.legend(size_handles, [f"LA≈{int(s)}" for s in size_samples], title="Размер узла: жилая площадь", loc="lower right", framealpha=0.8)
        ax.add_artist(leg2)

        ax.grid(True, alpha=0.2, linestyle=":")

        # Сохранение
        safe_zone = str(zone).replace("/", "-").replace(" ", "_")
        out_path = os.path.join(args.out_dir, f"zone_{safe_zone}__block_{blk}.png")
        plt.savefig(out_path, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)
        print(f"[ok] saved → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
