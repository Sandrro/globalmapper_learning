#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patch_zones_from_source_cli.py — дописывает колонку 'zone' в преобразованный датасет
из исходного слоя кварталов без полного пересчёта.

Предпосылки:
- В пайплайне block_id присваивался как индекс исходного слоя кварталов (0..N-1).
- Мы используем ТОТ ЖЕ файл кварталов (или его точную копию с тем же порядком),
  читаем колонку с названием зоны (по умолчанию 'functional_zone_type_name')
  и переносим её в out/blocks.parquet (и, по желанию, в by_block/*/blocks.parquet).

Пример:
  python patch_zones_from_source_cli.py \
    --zones /path/zones.geojson \
    --blocks-parquet out/blocks.parquet \
    --zone-column functional_zone_type_name \
    --update-per-block --by-block-root out/by_block \
    --backup
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
import pandas as pd

try:
    import geopandas as gpd
except Exception:
    gpd = None


def _fail(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    ap = argparse.ArgumentParser("Патч зоны в blocks.parquet из исходного слоя кварталов")
    ap.add_argument("--zones", required=True, help="Путь к исходному слою кварталов (GeoJSON/GeoPackage/Parquet и т.д.)")
    ap.add_argument("--blocks-parquet", required=True, help="out/blocks.parquet, созданный пайплайном")
    ap.add_argument("--zone-column", default="functional_zone_type_name",
                    help="Имя колонки с лейблом зоны в исходном слое кварталов")
    ap.add_argument("--backup", action="store_true", help="Сделать .bak копию blocks.parquet перед изменением")
    ap.add_argument("--update-per-block", action="store_true",
                    help="Также обновить by_block/*/blocks.parquet колонкой 'zone'")
    ap.add_argument("--by-block-root", default=None, help="Корень per-block директорий (обычно out/by_block)")
    args = ap.parse_args()

    if not os.path.exists(args.blocks_parquet):
        _fail(f"Не найден {args.blocks_parquet}")

    # 1) читаем преобразованные блоки
    df_blocks = pd.read_parquet(args.blocks_parquet)
    if "block_id" not in df_blocks.columns:
        _fail("В blocks.parquet отсутствует колонка 'block_id'")

    # 2) читаем исходные кварталы
    zones_path = args.zones
    if zones_path.lower().endswith(".parquet"):
        df_z = pd.read_parquet(zones_path)
    else:
        if gpd is None:
            _fail("Для чтения GeoJSON/GPKG нужен geopandas: pip install geopandas")
        df_z = gpd.read_file(zones_path)

    if args.zone_column not in df_z.columns:
        _fail(f"В исходном слое нет колонки зоны '{args.zone_column}' "
              f"(доступно: {', '.join(df_z.columns.astype(str).tolist()[:20])}...)")

    # 3) формируем block_id как индекс (как в пайплайне)
    df_z = df_z.reset_index(drop=True).copy()
    df_z["block_id"] = df_z.index.astype(str)
    df_z["zone"] = df_z[args.zone_column].astype(str)

    # sanity-check: размеры
    n_src = len(df_z)
    n_dst = df_blocks["block_id"].nunique()

    if n_src != n_dst:
        print(f"[WARN] Число кварталов в исходном слое ({n_src}) != blocks.parquet ({n_dst}). "
              f"Сопоставляем по совпадающим block_id, остальное будет заполнено 'nan'.")

    # 4) merge по block_id
    keep_cols = ["block_id", "zone"]
    df_merge = df_blocks.merge(df_z[keep_cols], on="block_id", how="left", suffixes=("", "_from_src"))

    # если в blocks.parquet уже есть своя 'zone', заменим её новой
    if "zone_from_src" in df_merge.columns:
        df_merge["zone"] = df_merge["zone_from_src"]
        df_merge = df_merge.drop(columns=["zone_from_src"])

    # 5) бэкап и запись
    if args.backup:
        shutil.copy2(args.blocks_parquet, args.blocks_parquet + ".bak")

    df_merge.to_parquet(args.blocks_parquet, index=False)
    n_filled = int(df_merge["zone"].notna().sum())
    print(f"[ok] Обновлён {args.blocks_parquet}: заполнено zone для {n_filled}/{len(df_merge)} кварталов.")

    # 6) (опц.) обновляем per-block деревья
    if args.update_per_block:
        root = args.by_block_root
        if not root:
            _fail("--update-per-block требует указать --by-block-root (обычно out/by_block)")

        # словарь block_id -> zone
        id2zone = dict(zip(df_merge["block_id"].astype(str), df_merge["zone"].astype(str)))

        import glob
        paths = sorted(glob.glob(os.path.join(root, "*", "blocks.parquet")))
        touched = 0
        for p in paths:
            try:
                df_one = pd.read_parquet(p)
                if df_one.empty:
                    continue
                bid = str(df_one.iloc[0]["block_id"])
                z = id2zone.get(bid, "nan")
                df_one["zone"] = z
                df_one.to_parquet(p, index=False)
                touched += 1
            except Exception as e:
                print(f"[WARN] skip {p}: {e}")
        print(f"[ok] Перезаписано per-block blocks.parquet: {touched} шт. в {root}")

    print("Готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
