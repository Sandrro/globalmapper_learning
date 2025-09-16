#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_graph_cli.py — экспорт квартальных графов в другие форматы (GEXF, JSON node-link/GraphSON, GPickle)

Исходные parquet (после пайплайна):
  - nodes_fixed.parquet
  - edges.parquet
  - blocks.parquet

Поддерживаемые форматы (`--format`):
  - gexf       → *.gexf[.gz] (подходит для Gephi)
  - nodelink   → *.json[.gz] (NetworkX json_graph.node_link_data; по сути GraphSON-подобный)
  - gpickle    → *.gpickle[.gz] (быстро и надёжно для Python/NetworkX)

Особенности:
  - Есть флаг `--force-string`: принудительная строковая сериализация всех атрибутов
    (помогает, если библиотека записи чувствительна к типам NumPy 2.0).
  - `--only-active` — писать только узлы с e_i==1 и рёбра между ними.
  - `--undirected` — сохранять рёбра как неориентированные (слияние дубликатов).

Примеры:
  GEXF (для Gephi):
    python export_graph_cli.py \
      --dataset-dir out \
      --out-dir out/gexf \
      --format gexf --undirected --force-string

  JSON (node-link / GraphSON-подобный):
    python export_graph_cli.py \
      --dataset-dir out \
      --out-dir out/json \
      --format nodelink --gzip

  GPickle (для дальнейшей работы в Python):
    python export_graph_cli.py \
      --dataset-dir out \
      --out-dir out/gpickle \
      --format gpickle
"""
from __future__ import annotations

import argparse
import json
import os
import gzip
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph


# ----------------- helpers -----------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _as_json_str(x: Any) -> str:
    try:
        if x is None:
            return "[]"
        if isinstance(x, str):
            try:
                json.loads(x)
                return x
            except Exception:
                return json.dumps([x], ensure_ascii=False)
        if isinstance(x, (list, tuple)):
            return json.dumps(list(x), ensure_ascii=False)
        if hasattr(x, "tolist"):
            return json.dumps(x.tolist(), ensure_ascii=False)
        return json.dumps([x], ensure_ascii=False)
    except Exception:
        return "[]"


def _to_py_scalar(v: Any) -> Any:
    if v is None:
        return None
    try:
        if v is pd.NA:
            return None
    except Exception:
        pass
    try:
        if hasattr(v, "item"):
            return v.item()
    except Exception:
        pass
    if isinstance(v, float) and (v != v):  # NaN
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    return v


def _clean_attrs(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        vv = _to_py_scalar(v)
        if isinstance(vv, float) and (vv != vv):
            continue
        if vv is None:
            continue
        out[k] = vv
    return out


def _stringify_attrs(d: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in d.items():
        if v is None or (isinstance(v, float) and v != v):
            continue
        if isinstance(v, (list, tuple)) or hasattr(v, "tolist"):
            out[k] = _as_json_str(v)
            continue
        try:
            if v is pd.NA:
                continue
        except Exception:
            pass
        try:
            if hasattr(v, "item"):
                v = v.item()
        except Exception:
            pass
        out[k] = str(v)
    return out


def _copy_graph(G: nx.Graph, stringify: bool) -> nx.Graph:
    H = nx.Graph() if isinstance(G, nx.Graph) and not isinstance(G, nx.DiGraph) else nx.DiGraph()
    if stringify:
        H.graph.update(_stringify_attrs(dict(G.graph)))
    else:
        H.graph.update(_clean_attrs(dict(G.graph)))
    for n, attrs in G.nodes(data=True):
        H.add_node(n, **(_stringify_attrs(attrs) if stringify else _clean_attrs(attrs)))
    for u, v, attrs in G.edges(data=True):
        H.add_edge(u, v, **(_stringify_attrs(attrs) if stringify else _clean_attrs(attrs)))
    return H


# ----------------- build graph per block -----------------

def build_graph_for_block(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    block_id: str,
    zone_val: Any,
    only_active: bool = False,
    undirected: bool = False,
) -> nx.Graph:
    sub_nodes = nodes_df[nodes_df["block_id"].astype(str) == str(block_id)].copy()
    sub_edges = edges_df[edges_df["block_id"].astype(str) == str(block_id)].copy()

    if only_active and "e_i" in sub_nodes.columns:
        sub_nodes = sub_nodes[pd.to_numeric(sub_nodes["e_i"], errors="coerce").fillna(0) > 0]

    G: nx.Graph
    G = nx.Graph() if undirected else nx.DiGraph()

    for r in sub_nodes.itertuples(index=False):
        sid = getattr(r, "slot_id", None)
        if pd.isna(sid):
            continue
        sid = int(sid)
        attr = {
            "slot_id": sid,
            "block_id": str(block_id),
            "zone": zone_val if zone_val is not None else "nan",
            "posx": getattr(r, "posx", None),
            "posy": getattr(r, "posy", None),
            "size_x": getattr(r, "size_x", None),
            "size_y": getattr(r, "size_y", None),
            "phi_resid": getattr(r, "phi_resid", None),
            "s_i": getattr(r, "s_i", None),
            "a_i": getattr(r, "a_i", None),
            "floors_num": getattr(r, "floors_num", None),
            "living_area": getattr(r, "living_area", None),
            "is_living": getattr(r, "is_living", None),
            "has_floors": getattr(r, "has_floors", None),
            "aspect_ratio": getattr(r, "aspect_ratio", None),
            "branch_local_id": getattr(r, "branch_local_id", None),
        }
        if "services_present" in sub_nodes.columns:
            attr["services_present"] = _as_json_str(getattr(r, "services_present"))
        if "services_capacity" in sub_nodes.columns:
            attr["services_capacity"] = _as_json_str(getattr(r, "services_capacity"))
        G.add_node(sid, **_clean_attrs(attr))

    seen_undirected: set[Tuple[int, int]] = set()
    for e in sub_edges.itertuples(index=False):
        try:
            u = int(getattr(e, "src_slot")); v = int(getattr(e, "dst_slot"))
        except Exception:
            continue
        if u not in G or v not in G:
            if only_active:
                continue
            G.add_node(u); G.add_node(v)
        if undirected:
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) in seen_undirected:
                continue
            seen_undirected.add((a, b))
            G.add_edge(a, b)
        else:
            if G.has_edge(u, v):
                continue
            G.add_edge(u, v)

    G.graph["block_id"] = str(block_id)
    G.graph["zone"] = zone_val if zone_val is not None else "nan"
    G.graph["n_nodes"] = int(G.number_of_nodes())
    G.graph["n_edges"] = int(G.number_of_edges())
    return G


# ----------------- writers -----------------

def _write_gexf(G: nx.Graph, out_path: str, force_string: bool) -> bool:
    H = _copy_graph(G, stringify=force_string)
    try:
        nx.write_gexf(H, out_path)
        return True
    except Exception as e:
        # Последняя попытка — ещё сильнее «очистить» типы
        try:
            H2 = _copy_graph(H, stringify=True)
            nx.write_gexf(H2, out_path)
            return True
        except Exception:
            print(f"[gexf skip] {out_path}: {e}")
            return False


def _write_nodelink(G: nx.Graph, out_path: str, gzip_out: bool) -> bool:
    # Преобразуем атрибуты к python-примитивам
    H = _copy_graph(G, stringify=False)
    try:
        # В разных версиях NetworkX сигнатура без аргумента directed:
        data = json_graph.node_link_data(H)  # сам возьмет H.is_directed()
    except TypeError:
        # На всякий: повторяем без параметров (старые ветки тоже сюда попадают)
        data = json_graph.node_link_data(H)
    try:
        if gzip_out:
            with gzip.open(out_path, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        else:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[json skip] {out_path}: {e}")
        return False


def _write_gpickle(G: nx.Graph, out_path: str) -> bool:
    try:
        nx.write_gpickle(G, out_path)
        return True
    except Exception as e:
        print(f"[gpickle skip] {out_path}: {e}")
        return False


# ----------------- export all blocks -----------------

def export_all_blocks(
    dataset_dir: str,
    out_dir: str,
    fmt: str,
    only_active: bool = False,
    undirected: bool = False,
    gzip_out: bool = False,
    zone_fill: str = "nan",
    force_string: bool = False,
) -> int:
    nodes_path = os.path.join(dataset_dir, "nodes_fixed.parquet")
    edges_path = os.path.join(dataset_dir, "edges.parquet")
    blocks_path = os.path.join(dataset_dir, "blocks.parquet")
    if not (os.path.exists(nodes_path) and os.path.exists(edges_path) and os.path.exists(blocks_path)):
        raise FileNotFoundError("Ожидались nodes_fixed.parquet, edges.parquet и blocks.parquet в --dataset-dir")

    _ensure_dir(out_dir)
    nodes = pd.read_parquet(nodes_path)
    edges = pd.read_parquet(edges_path)
    blocks = pd.read_parquet(blocks_path)
    blk_zone = blocks.set_index("block_id")["zone"].to_dict()

    block_ids = sorted(set(str(b) for b in nodes["block_id"].astype(str).unique().tolist()))
    n_ok = 0

    for bid in block_ids:
        zone_val = blk_zone.get(bid, zone_fill)
        G = build_graph_for_block(nodes, edges, bid, zone_val, only_active=only_active, undirected=undirected)
        if G.number_of_nodes() == 0:
            continue

        safe_zone = str(zone_val).replace("/", "-").replace(" ", "_")
        if fmt == "gexf":
            fname = f"block_{bid}__zone_{safe_zone}.gexf"
            if gzip_out:
                fname += ".gz"
            out_path = os.path.join(out_dir, fname)
            ok = _write_gexf(G, out_path, force_string)
        elif fmt == "nodelink":
            fname = f"block_{bid}__zone_{safe_zone}.json"
            if gzip_out:
                fname += ".gz"
            out_path = os.path.join(out_dir, fname)
            ok = _write_nodelink(G, out_path, gzip_out)
        elif fmt == "gpickle":
            fname = f"block_{bid}__zone_{safe_zone}.gpickle"
            if gzip_out:
                fname += ".gz"  # networkx поддерживает *.gpickle.gz
            out_path = os.path.join(out_dir, fname)
            ok = _write_gpickle(G, out_path)
        else:
            raise ValueError(f"Неизвестный формат: {fmt}")

        if ok:
            n_ok += 1

    print(f"[done] exported {n_ok} {fmt} files → {out_dir}")
    return n_ok


# ----------------- CLI -----------------

def main() -> int:
    ap = argparse.ArgumentParser("Экспорт графов в GEXF / JSON node-link / GPickle")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--format", choices=["gexf", "nodelink", "gpickle"], default="gexf")
    ap.add_argument("--only-active", action="store_true")
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--gzip", action="store_true", help="Сохранять с .gz (для gexf/json — внешнее сжатие; для gpickle поддерживается natively)")
    ap.add_argument("--zone-fill", type=str, default="nan")
    ap.add_argument("--force-string", action="store_true", help="Только для gexf: сериализовать все атрибуты строками")
    args = ap.parse_args()

    _ensure_dir(args.out_dir)
    export_all_blocks(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        fmt=args.format,
        only_active=bool(args.only_active),
        undirected=bool(args.undirected),
        gzip_out=bool(args.gzip),
        zone_fill=str(args.zone_fill),
        force_string=bool(args.force_string),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
