#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_sample.py — быстрый просмотр сэмпа из Parquet
Флаг --non-empty-services оставляет только строки с сервисами (по колонкам has_service__).
Добавлен --debug-services для сводки по сервисам и их capacity.
"""
import argparse, sys
import pandas as pd
import numpy as np

from services_processing import infer_service_schema_from_nodes


def _any_service_mask(df: pd.DataFrame, schema) -> pd.Series:
    if not schema:
        return pd.Series(False, index=df.index)
    mask = pd.Series(False, index=df.index)
    for item in schema:
        col = item.get("has_column")
        if col and col in df.columns:
            mask = mask | df[col].fillna(False).astype(bool)
    return mask


def _debug_services_summary(df: pd.DataFrame, schema) -> None:
    if not schema:
        print("== DEBUG services ==\nСхема сервисов не обнаружена.")
        return
    print("== DEBUG services ==")
    print(f"Строк всего: {len(df):,}")
    for item in schema:
        name = item.get("name")
        has_col = item.get("has_column")
        cap_col = item.get("capacity_column")
        if not has_col or has_col not in df.columns:
            continue
        has_vals = df[has_col].fillna(False).astype(bool)
        cnt = int(has_vals.sum())
        cap_sum = 0.0
        if cap_col and cap_col in df.columns:
            caps = pd.to_numeric(df[cap_col], errors="coerce").fillna(0.0)
            cap_sum = float(caps[has_vals].sum())
        print(f"  {name}: count={cnt:,}, capacity_sum={cap_sum:.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Путь к .parquet")
    ap.add_argument("--n", type=int, default=10, help="Размер сэмпа (по умолчанию 10)")
    ap.add_argument("--random", action="store_true", help="Случайный сэмп вместо .head()")
    ap.add_argument("--seed", type=int, default=42, help="Зерно для случайного сэмпа")
    ap.add_argument("--cols", type=str, default="", help="Список колонок через запятую")
    ap.add_argument("--non-empty-services", action="store_true",
                    help="Показывать только строки, где присутствует хотя бы один сервис")
    ap.add_argument("--debug-services", action="store_true",
                    help="Вывести сводку по сервисам перед выборкой")
    args = ap.parse_args()

    user_cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_colwidth", 200)

    try:
        df = pd.read_parquet(args.path)
    except Exception as e:
        print(f"[ERR] Не удалось прочитать {args.path}: {e}", file=sys.stderr)
        sys.exit(1)

    n_total = len(df)
    if n_total == 0:
        print(f"Файл: {args.path}\nФайл пустой.")
        return

    service_schema = infer_service_schema_from_nodes(df)

    if args.debug_services:
        _debug_services_summary(df, service_schema)

    if args.non_empty_services:
        mask = _any_service_mask(df, service_schema)
        df = df[mask]

    n_after = len(df)
    if n_after == 0:
        print(f"Файл: {args.path}")
        if args.non_empty_services:
            print("После фильтрации по сервисам строк не осталось.")
        else:
            print("Файл пустой.")
        return

    k = min(args.n, n_after)
    sample = df.sample(k, random_state=args.seed) if args.random else df.head(k)

    if user_cols:
        out_cols = [c for c in user_cols if c in sample.columns]
        if out_cols:
            sample = sample[out_cols]

    print(f"Файл: {args.path}")
    if args.non_empty_services:
        print(f"Строк всего: {n_total:,}; с сервисами: {n_after:,}; показываю {k}")
    else:
        print(f"Строк всего: {n_total:,}; показываю {k}")
    print("\n--- Сэмп ---")
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()
