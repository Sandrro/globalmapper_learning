#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_parquets_cli.py — собрать per-block parquet-файлы в четыре общих файла

Ищет дерево out/by_block/<block_id>/{blocks,branches,nodes_fixed,edges}.parquet
и сливает их в корневые файлы каталога out/:
  out/blocks.parquet
  out/branches.parquet
  out/nodes_fixed.parquet
  out/edges.parquet

Особенности:
- Пропускает отсутствующие/битые parquet без падения.
- Для nodes_fixed принудительно приводит floors_num к Int64, если колонка есть.
- Логи с количеством собранных строк и временем выполнения.

Пример:
  python merge_parquets_cli.py \
    --by-block-root out/by_block \
    --out-dir out \
    --log-level INFO
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from glob import glob
from typing import List

import pandas as pd

log = logging.getLogger("merge_parquets")


def setup_logger(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _merge_parquet_tree(by_block_root: str, rel_name: str, out_path: str) -> int:
    """Собрать все by_block/*/<rel_name> в один parquet по пути out_path.

    Возвращает количество записей в результирующем файле.
    """
    t0 = time.perf_counter()
    paths = sorted(glob(os.path.join(by_block_root, "*", rel_name)))
    if not paths:
        pd.DataFrame([]).to_parquet(out_path, index=False)
        log.warning(f"нет файлов для {rel_name} — создан пустой parquet → {out_path}")
        return 0

    dfs: List[pd.DataFrame] = []
    bad = 0
    for p in paths:
        try:
            dfs.append(pd.read_parquet(p))
        except Exception as e:  # noqa: BLE001
            bad += 1
            log.warning(f"пропускаю '{p}': {e}")

    if not dfs:
        pd.DataFrame([]).to_parquet(out_path, index=False)
        log.warning(f"все входные файлы для {rel_name} повреждены/недоступны → пустой parquet")
        return 0

    df = pd.concat(dfs, ignore_index=True)

    # Специальный случай для узлов
    if rel_name == "nodes_fixed.parquet" and "floors_num" in df.columns:
        try:
            df["floors_num"] = df["floors_num"].astype("Int64")
        except Exception:  # noqa: BLE001
            pass

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    n = len(df)
    t1 = time.perf_counter()
    msg = f"[{rel_name}] merged {n} rows → {out_path} in {t1 - t0:.2f}s"
    if bad:
        msg += f" (skipped {bad} bad files)"
    log.info(msg)
    return n


def main() -> int:
    ap = argparse.ArgumentParser("Сборка per-block parquet в 4 общих файла")
    ap.add_argument("--by-block-root", required=True, help="Каталог by_block с подкаталогами по block_id")
    ap.add_argument("--out-dir", required=True, help="Куда писать blocks/branches/nodes_fixed/edges.parquet")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logger(args.log_level)
    os.makedirs(args.out_dir, exist_ok=True)

    total_blocks = _merge_parquet_tree(args.by_block_root, "blocks.parquet",   os.path.join(args.out_dir, "blocks.parquet"))
    total_branches = _merge_parquet_tree(args.by_block_root, "branches.parquet", os.path.join(args.out_dir, "branches.parquet"))
    total_nodes = _merge_parquet_tree(args.by_block_root, "nodes_fixed.parquet", os.path.join(args.out_dir, "nodes_fixed.parquet"))
    total_edges = _merge_parquet_tree(args.by_block_root, "edges.parquet",    os.path.join(args.out_dir, "edges.parquet"))

    log.info(f"[ok] saved blocks.parquet     rows={total_blocks}")
    log.info(f"[ok] saved branches.parquet   rows={total_branches}")
    log.info(f"[ok] saved nodes_fixed.parquet rows={total_nodes}")
    log.info(f"[ok] saved edges.parquet      rows={total_edges}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
