"""
services_utils.py — утилиты для работы с сервисами (present/capacity + карта индексов)

Назначение:
- Парсинг services_present:
  * списки имён;
  * бинарные векторы 0/1 (если передана карта service_type_name→vector_index);
  * JSON-строки, строки с разделителями, dict-форматы.
- Парсинг services_capacity:
  * списки чисел;
  * списки dict[{name,value}] или dict {name:value};
  * числовые векторы (если есть карта → перевод в [{name,value}] для ненулевых позиций).
- Нормализация колонок в DataFrame зданий (для transform-пайплайна).
- Загрузка CSV-карты и сохранение её в services.json для train.py.

Совместимо с HCanonGraphDataset._services_vecs в train.py.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "load_service_map_csv",
    "write_services_json",
    "parse_services_present",
    "parse_services_capacity",
    "normalize_services_columns",
    "services_present_list",
    "services_capacity_list",
]

# -------------------------
# I/O карты сервисов
# -------------------------

def load_service_map_csv(path: str) -> Dict[str, int]:
    """Загрузить CSV-карту сервисов → индекс.

    Ожидаемые колонки: ``service_type_name``, ``vector_index``.
    Имя нормализуется strip(); допускаются дубликаты имён — берётся последнее.
    """
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    if not {"service_type_name", "vector_index"}.issubset(df.columns):
        raise ValueError("CSV must have columns: service_type_name, vector_index")
    mp: Dict[str, int] = {}
    for r in df.itertuples(index=False):
        name = str(getattr(r, "service_type_name")).strip()
        try:
            idx = int(getattr(r, "vector_index"))
        except Exception:
            continue
        if name != "":
            mp[name] = idx
    return mp


def write_services_json(service_map: Mapping[str, int], out_path: str) -> None:
    """Сохранить карту ``name→index`` в JSON (для ``train.py --services-json``)."""
    if not service_map:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dict(service_map), f, ensure_ascii=False, indent=2)


# -------------------------
# Парсинг present/capacity
# -------------------------

def _looks_like_binary_vector(arr: Sequence[Any]) -> bool:
    if len(arr) == 0:
        return False
    for v in arr:
        if v is None:
            continue
        try:
            iv = int(float(v) != 0.0)
        except Exception:
            return False
        if iv not in (0, 1):
            return False
    return True


def parse_services_present(x: Any, service_map: Optional[Mapping[str, int]] = None) -> List[str]:
    """Нормализовать ``services_present`` в список **имён** сервисов.

    Поддерживаемые форматы ввода:
      - list/tuple/ndarray имён (строки → как есть);
      - бинарный вектор 0/1 (при наличии ``service_map``);
      - dict: ``{"names":[...]}`` или ``{"school":1, "clinic":0}``;
      - JSON-строка перечисленных выше вариантов;
      - строка с разделителями (",;|\t").
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # list-like
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = list(x)
        if service_map and _looks_like_binary_vector(arr):
            idx2name = {int(i): name for name, i in service_map.items()}
            out: List[str] = []
            for i, v in enumerate(arr):
                try:
                    if float(v) != 0.0 and i in idx2name:
                        out.append(idx2name[i])
                except Exception:
                    continue
            return out
        # иначе трактуем элементы как имена
        out = []
        for v in arr:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            out.append(str(v))
        return out

    # dict-like
    if isinstance(x, dict):
        if "names" in x and isinstance(x["names"], (list, tuple)):
            return [str(v) for v in x["names"] if not (isinstance(v, float) and pd.isna(v))]
        return [str(k) for k, v in x.items() if bool(v)]

    # string
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        # попытка JSON
        try:
            v = json.loads(s)
            return parse_services_present(v, service_map)
        except Exception:
            pass
        # сплит по разделителям
        parts = re.split(r"[,;|\t]+", s)
        return [p.strip() for p in parts if p.strip()]

    return []


def parse_services_capacity(x: Any, service_map: Optional[Mapping[str, int]] = None) -> List[Union[float, Dict[str, float]]]:
    """Нормализовать ``services_capacity``.

    Возвращаем либо список чисел, либо список dict ``{"name": <str>, "value": <float>}``.
    Если передан числовой вектор и есть ``service_map``, превращаем в [{name,value}] для ненулевых.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, (list, tuple, np.ndarray)):
        arr = list(x)
        is_num = all((v is None) or isinstance(v, (int, float, np.integer, np.floating)) for v in arr)
        if service_map and is_num:
            idx2name = {int(i): name for name, i in service_map.items()}
            out: List[Dict[str, float]] = []
            for i, v in enumerate(arr):
                try:
                    val = float(0.0 if v is None else v)
                except Exception:
                    val = 0.0
                if val != 0.0 and i in idx2name:
                    out.append({"name": idx2name[i], "value": val})
            return out
        # смешанный или числовой без карты → вернуть числа/словарики как есть
        out2: List[Union[float, Dict[str, float]]] = []
        for v in arr:
            if isinstance(v, dict):
                nm = v.get("name")
                try:
                    val = float(v.get("value", 0.0))
                except Exception:
                    val = 0.0
                out2.append({"name": (str(nm) if nm is not None else None), "value": val})
            else:
                try:
                    out2.append(float(0.0 if v is None else v))
                except Exception:
                    out2.append(0.0)
        return out2

    if isinstance(x, dict):
        out = []
        for k, v in x.items():
            try:
                out.append({"name": str(k), "value": float(0.0 if v is None else v)})
            except Exception:
                out.append({"name": str(k), "value": 0.0})
        return out

    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        try:
            v = json.loads(s)
            return parse_services_capacity(v, service_map)
        except Exception:
            pass
        try:
            parts = [p.strip() for p in re.split(r"[,;|\t]+", s) if p.strip()]
            return [float(p) for p in parts]
        except Exception:
            return []

    return []


# -------------------------
# Нормализация колонок DataFrame зданий
# -------------------------

def normalize_services_columns(
    bldgs: pd.DataFrame,
    service_map: Optional[Mapping[str, int]] = None,
    lowercase: bool = True,
) -> pd.DataFrame:
    """Нормализовать/создать ``services_present`` и ``services_capacity`` в DF зданий.

    * Если ``services_present`` бинарный вектор и передана карта — конвертируется в список имён.
    * Если ``services_capacity`` числовой вектор и есть карта — конвертируется в [{name,value}] для ненулевых.
    * Имена канонизируются по карте (регистронезависимо); неизвестные имена отбрасываются.
    """
    # present
    if "services_present" in bldgs.columns:
        bldgs["services_present"] = bldgs["services_present"].apply(
            lambda v: parse_services_present(v, service_map)
        )
    else:
        bldgs["services_present"] = [[] for _ in range(len(bldgs))]

    # capacity
    if "services_capacity" in bldgs.columns:
        bldgs["services_capacity"] = bldgs["services_capacity"].apply(
            lambda v: parse_services_capacity(v, service_map)
        )
    else:
        bldgs["services_capacity"] = [[] for _ in range(len(bldgs))]

    if service_map:
        lower2canon = {k.lower(): k for k in service_map.keys()}

        def _canon_names(lst: Iterable[str]) -> List[str]:
            out: List[str] = []
            for name in lst:
                key = name.lower() if isinstance(name, str) and lowercase else name
                canon = lower2canon.get(key, None)
                if canon is not None:
                    out.append(canon)
            # уникализация с сохранением порядка
            seen = set(); uniq: List[str] = []
            for n in out:
                if n not in seen:
                    uniq.append(n); seen.add(n)
            return uniq

        def _canon_caps(obj: Any) -> Any:
            if isinstance(obj, list):
                out: List[Union[float, Dict[str, float]]] = []
                for v in obj:
                    if isinstance(v, dict):
                        nm = v.get("name")
                        key = nm.lower() if isinstance(nm, str) and lowercase else nm
                        if key in lower2canon:
                            out.append({"name": lower2canon[key], "value": float(v.get("value", 0.0))})
                    else:
                        try:
                            out.append(float(0.0 if v is None else v))
                        except Exception:
                            out.append(0.0)
                return out
            return obj

        bldgs["services_present"] = bldgs["services_present"].apply(_canon_names)
        bldgs["services_capacity"] = bldgs["services_capacity"].apply(_canon_caps)

    return bldgs


# -------------------------
# Хелперы для воркера (wrap-функции)
# -------------------------

def services_present_list(x: Any, service_map: Optional[Mapping[str, int]] = None) -> List[str]:
    """Сокращённый вызов для воркера per-block."""
    return normalize_services_columns(
        pd.DataFrame({"tmp": [x]}), service_map
    )["services_present"].iloc[0]


def services_capacity_list(x: Any, service_map: Optional[Mapping[str, int]] = None) -> List[Union[float, Dict[str, float]]]:
    """Сокращённый вызов для воркера per-block."""
    return normalize_services_columns(
        pd.DataFrame({"tmp": [x]}), service_map
    )["services_capacity"].iloc[0]


if __name__ == "__main__":
    # Небольшой self-test
    mp = {"school": 1, "clinic": 3, "market": 5}
    bin_vec = [0, 1, 0, 1, 0, 0]
    names = parse_services_present(bin_vec, mp)
    assert names == ["school", "clinic"], names
    caps = parse_services_capacity([0, 120, 0, 40, 0, 0], mp)
    assert {d["name"] for d in caps} == {"school", "clinic"}
    print("services_utils.py self-test OK")
