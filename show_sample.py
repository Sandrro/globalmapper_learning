#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_sample.py — быстрый просмотр сэмпа из Parquet
Флаг --non-empty-services оставляет только строки с реально НЕпустым services_capacity.
Добавлен --debug-services для сводки по типам значений.
"""
import argparse, sys, json
import pandas as pd
import numpy as np

def _strict_has_services(x) -> bool:
    # None
    if x is None:
        return False

    # Явные коллекции Python
    if isinstance(x, (list, tuple, dict)):
        try:
            return len(x) > 0
        except Exception:
            return False

    # NumPy массивы (избегаем pd.isna на массивах)
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return False
        # если объектный массив — проверим элементы рекурсивно
        if x.dtype == object:
            for el in x.ravel():
                if _strict_has_services(el):
                    return True
            return False
        # числовые/строковые массивы тут считаем пустыми (нет структур сервиса)
        return False

    # Скаляры NaN (float/np.floating)
    if isinstance(x, (float, np.floating)):
        return not np.isnan(x) and False  # скаляр float сам по себе сервиса не описывает

    # Байты/строки — пробуем распарсить JSON
    if isinstance(x, (str, bytes)):
        s = x.decode("utf-8") if isinstance(x, bytes) else x
        s = s.strip()
        if s in ("", "[]", "{}"):
            return False
        try:
            j = json.loads(s)
            if isinstance(j, (list, tuple, dict)):
                return len(j) > 0
            return False
        except Exception:
            # Нестандартизованные строки считаем пустыми в строгом режиме
            return False

    # Всё остальное — пусто
    return False

def _debug_services_summary(series: pd.Series):
    total = len(series)
    cnt_na = int(series.isna().sum()) if hasattr(series, "isna") else 0
    as_type = series.apply(lambda v: type(v).__name__ if v is not None else "NA")
    top_types = as_type.value_counts().head(8)
    try:
        as_str = series[as_type.eq("str")].astype(str).str.strip()
        cnt_str_empty = int((as_str.eq("") | as_str.eq("[]") | as_str.eq("{}")).sum())
    except Exception:
        cnt_str_empty = 0
    mask_strict = series.apply(_strict_has_services)
    cnt_nonempty = int(mask_strict.sum())
    print("== DEBUG services_capacity ==")
    print(f"Всего значений: {total:,}")
    print(f"NA/пропусков : {cnt_na:,}")
    print("Топ типов:")
    for k, v in top_types.items():
        print(f"  {k}: {v:,}")
    if cnt_str_empty:
        print(f'Пустые строковые "[]"/"{{}}"/"" : {cnt_str_empty:,}')
    print(f"Непустых по строгому правилу: {cnt_nonempty:,}")
    print("Примеры непустых (до 5):")
    for val in series[mask_strict].head(5).tolist():
        print("  ", val)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Путь к .parquet")
    ap.add_argument("--n", type=int, default=10, help="Размер сэмпа (по умолчанию 10)")
    ap.add_argument("--random", action="store_true", help="Случайный сэмп вместо .head()")
    ap.add_argument("--seed", type=int, default=42, help="Зерно для случайного сэмпа")
    ap.add_argument("--cols", type=str, default="", help="Список колонок через запятую")
    ap.add_argument("--non-empty-services", action="store_true",
                    help="Показывать только строки, где services_capacity непустой (строго)")
    ap.add_argument("--debug-services", action="store_true",
                    help="Вывести сводку по services_capacity перед выборкой")
    args = ap.parse_args()

    user_cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    read_cols = None
    if user_cols:
        if args.non_empty_services and "services_capacity" not in user_cols:
            read_cols = list(dict.fromkeys(user_cols + ["services_capacity"]))
        else:
            read_cols = user_cols

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_colwidth", 200)

    try:
        df = pd.read_parquet(args.path, columns=read_cols)
    except Exception as e:
        print(f"[ERR] Не удалось прочитать {args.path}: {e}", file=sys.stderr)
        sys.exit(1)

    n_total = len(df)
    if n_total == 0:
        print(f"Файл: {args.path}\nФайл пустой.")
        return

    if args.debug_services:
        if "services_capacity" in df.columns:
            _debug_services_summary(df["services_capacity"])
        else:
            print("== DEBUG services_capacity ==\nКолонка 'services_capacity' отсутствует.")

    if args.non_empty_services:
        if "services_capacity" not in df.columns:
            print("[!] Колонка 'services_capacity' не найдена — нечем фильтровать.", file=sys.stderr)
            sys.exit(2)
        mask = df["services_capacity"].apply(_strict_has_services)
        df = df[mask]

    n_after = len(df)
    if n_after == 0:
        print(f"Файл: {args.path}")
        if args.non_empty_services:
            print("После фильтрации по непустому services_capacity строк не осталось.")
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
        print(f"Строк всего: {n_total:,}; с непустым services_capacity: {n_after:,}; показываю {k}")
    else:
        print(f"Строк всего: {n_total:,}; показываю {k}")
    print("\n--- Сэмп ---")
    print(sample.to_string(index=False))

if __name__ == "__main__":
    main()
