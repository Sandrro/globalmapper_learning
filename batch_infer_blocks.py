#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_infer_blocks.py — пакетная генерация застройки по набору кварталов через train.py --mode infer

Что делает:
- Читает FeatureCollection с полигонами кварталов; метка зоны в properties[--zone-attr] (по умолчанию "zone").
- Для каждой зоны берёт цели из --targets-by-zone: { "zone": {"la": <м2>, "floors_avg": <этажи>} }.
  Синонимы ключей внутри зоны: la|living_area|la_target и floors_avg|floors.
- Если для residential не задана la, можно fallback: la = 15 м² * people (из properties или --people).
- НЕ передаёт --zones-json/--services-json, если вы их явно не указали,
  чтобы train.py использовал артефакты из каталога рядом с чекпойнтом: <ckpt_dir>/artifacts/{zones.json,services.json}.

Пример:
  python batch_infer_blocks.py \
    --blocks ./zones.geojson \
    --out ./buildings_gen.geojson \
    --train-script ./train.py \
    --model-ckpt ./out_2/checkpoints/graphgen_hcanon_v1.pt \
    --zone-attr zone \
    --targets-by-zone ./targets_by_zone.json \
    --infer-slots 256 --infer-knn 8
"""

from __future__ import annotations
import os, sys, json, argparse, tempfile, subprocess, shutil
from typing import Dict, Any, List
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # без прогресса, если tqdm не установлен


# ---- утилиты ----

SYNONYMS = {
    "school":       ["school", "школа", "общеобразовательная школа", "образовательная организация"],
    "kindergarten": ["kindergarten", "детский сад", "детсад", "дошкольное", "д/с"],
    "polyclinic":   ["polyclinic", "clinic", "поликлиника", "амбулатория", "клиника"],
}

def load_services_vocab_from_artifacts(model_ckpt: str) -> Dict[str, int] | None:
    """Пробуем прочитать services.json из artifacts рядом с чекпойнтом."""
    aux_dir = os.path.join(os.path.dirname(model_ckpt) or ".", "artifacts")
    sj = os.path.join(aux_dir, "services.json")
    if os.path.exists(sj):
        try:
            with open(sj, "r", encoding="utf-8") as f:
                return json.load(f)  # name -> id
        except Exception:
            return None
    return None

def pick_service_keys(vocab: Dict[str,int] | None) -> Dict[str,str]:
    """
    Возвращает {canonical_key: actual_vocab_key} для ['school','kindergarten','polyclinic'].
    Если vocab неизвестен — возвращает английские ключи как есть.
    """
    out = {"school":"school", "kindergarten":"kindergarten", "polyclinic":"polyclinic"}
    if not vocab:
        return out
    names = list(vocab.keys())
    lowered = [s.casefold() for s in names]
    for canon, syns in SYNONYMS.items():
        for syn in syns:
            cf = syn.casefold()
            # точное совпадение
            if cf in lowered:
                out[canon] = names[lowered.index(cf)]
                break
            # подстрока
            idx = next((i for i,s in enumerate(lowered) if cf in s), None)
            if idx is not None:
                out[canon] = names[idx]
                break
    return out

def read_geojson(path: str) -> Dict[str,Any]:
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    if gj.get("type") != "FeatureCollection":
        # обернём единичный объект как FC
        if gj.get("type") == "Feature":
            feats = [gj]
        else:
            feats = [{"type":"Feature","geometry":gj,"properties":{}}]
        gj = {"type":"FeatureCollection", "features": feats}
    return gj

def write_geojson(path: str, fc: Dict[str,Any]) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f, ensure_ascii=False, indent=2)

def get_people(props: Dict[str,Any], default_people: int) -> int:
    for k in ("population","people","POPULATION","num_people"):
        if k in props:
            try:
                v = int(float(props[k]))
                if v > 0:
                    return v
            except Exception:
                pass
    return int(default_people)

def zone_label_of(feat: Dict[str,Any], zone_attr: str, fallback: str|None=None) -> str|None:
    props = feat.get("properties") or {}
    z = props.get(zone_attr)
    if isinstance(z, str):
        return z
    if fallback:
        return fallback
    return None

def load_json_maybe(path_or_json: str) -> Any:
    """Читает JSON из файла, если путь существует; иначе парсит как строку JSON."""
    try:
        if os.path.exists(path_or_json):
            with open(path_or_json, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            return json.loads(path_or_json)
    except Exception as e:
        print(f"[WARN] failed to read JSON '{path_or_json}': {e}", file=sys.stderr)
        return None

def normalize_targets_map(raw: Any) -> Dict[str, Dict[str, float]]:
    """
    Приводит словарь целей по зонам к виду:
      { zone_label: { "la": <float or None>, "floors_avg": <float or None> } }
    Поддерживаем синонимы ключей: la|living_area|la_target и floors_avg|floors.
    """
    out: Dict[str, Dict[str, float]] = {}
    if not isinstance(raw, dict):
        return out
    for zone, val in raw.items():
        if not isinstance(zone, str) or not isinstance(val, dict):
            continue
        la = None
        fl = None
        # площадь
        for k in ("la", "living_area", "la_target"):
            if k in val and val[k] is not None:
                try:
                    la = float(val[k]); break
                except Exception:
                    pass
        # этажность
        for k in ("floors_avg", "floors"):
            if k in val and val[k] is not None:
                try:
                    fl = float(val[k]); break
                except Exception:
                    pass
        out[zone] = {"la": la, "floors_avg": fl}
    return out


# ---- основной скрипт ----

def main():
    ap = argparse.ArgumentParser(description="Batch infer buildings for blocks via train.py --mode infer")
    ap.add_argument("--blocks", required=True, help="Входной GeoJSON (FeatureCollection) кварталов с полигонами")
    ap.add_argument("--out", required=True, help="Выходной GeoJSON зданий")
    ap.add_argument("--train-script", default="./train.py", help="Путь к train.py")
    ap.add_argument("--model-ckpt", required=True, help="Путь к чекпойнту модели (.pt)")
    # ВАЖНО: по умолчанию НЕ передаём эти пути — train.py возьмёт артефакты рядом с чекпойнтом
    ap.add_argument("--zones-json", default=None, help="(опц.) путь к zones.json; по умолчанию ckpt_dir/artifacts/zones.json")
    ap.add_argument("--services-json", default=None, help="(опц.) путь к services.json; по умолчанию ckpt_dir/artifacts/services.json")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None, help="cuda|cpu (пробрасывается в train.py как --device)")
    ap.add_argument("--zone-attr", default="zone", help="Имя свойства с меткой зоны в кварталах")
    ap.add_argument("--people", type=int, default=1000, help="Число жителей на квартал (fallback для residential)")
    ap.add_argument("--min-services", action="store_true", default=True,
                    help="Требовать минимум по одному: школа, поликлиника, детсад (в residential)")
    # инференс-параметры 1:1 с train.py
    ap.add_argument("--infer-slots", type=int, default=256)
    ap.add_argument("--infer-knn", type=int, default=8)
    ap.add_argument("--infer-e-thr", type=float, default=0.5)
    ap.add_argument("--infer-il-thr", type=float, default=0.5)
    ap.add_argument("--infer-sv1-thr", type=float, default=0.5)

    # Карта целей по зонам
    ap.add_argument("--targets-by-zone", default=None,
                    help="Путь к JSON или JSON-строка вида "
                         "{\"residential\": {\"la\": 12000, \"floors_avg\": 8}, \"business\": {\"la\": 8000}}")
    # Устаревшее (оставлено для совместимости): только la по зонам
    ap.add_argument("--la-by-zone", default=None,
                    help="DEPRECATED: Путь/JSON с {\"residential\": 12000, ...}; используйте --targets-by-zone")

    args = ap.parse_args()

    # services vocab (для синонимов имен сервисов)
    vocab = load_services_vocab_from_artifacts(args.model_ckpt)
    svc_keys = pick_service_keys(vocab)

    # Загрузка словаря целей по зонам
    targets_by_zone: Dict[str, Dict[str, float]] = {}
    if args.targets_by_zone:
        raw = load_json_maybe(args.targets_by_zone)
        targets_by_zone = normalize_targets_map(raw)
    elif args.la_by_zone:
        raw_la = load_json_maybe(args.la_by_zone)
        if isinstance(raw_la, dict):
            targets_by_zone = {z: {"la": (float(v) if v is not None else None), "floors_avg": None}
                               for z, v in raw_la.items()}

    fc = read_geojson(args.blocks)
    out_feats: List[Dict[str,Any]] = []

    tmpdir = tempfile.mkdtemp(prefix="ked_infer_")
    try:
        for i, feat in enumerate(tqdm(fc.get("features", []), desc="Blocks")):
            z = zone_label_of(feat, args.zone_attr)
            if not z:
                print(f"[WARN] feature #{i}: не найдена зона в properties['{args.zone_attr}'] — пропуск", file=sys.stderr)
                continue

            # цели по зоне
            t_z = targets_by_zone.get(z, {})
            la_target = t_z.get("la")
            floors_avg = t_z.get("floors_avg")

            # fallback для residential по людям (только если la не задана в targets-by-zone)
            services_target: Dict[str, float] = {}
            if (la_target is None) and (str(z).casefold() == "residential"):
                people = get_people(feat.get("properties") or {}, args.people)
                la_target = float(15.0 * people)
                if args.min_services:
                    services_target[svc_keys["school"]]       = 1.0
                    services_target[svc_keys["polyclinic"]]   = 1.0
                    services_target[svc_keys["kindergarten"]] = 1.0

            # временные файлы для данного квартала
            in_path  = os.path.join(tmpdir, f"blk_{i:06d}.geojson")
            out_path = os.path.join(tmpdir, f"blk_{i:06d}_out.geojson")
            write_geojson(in_path, {"type":"FeatureCollection","features":[feat]})

            # команда вызова train.py --mode infer
            cmd = [
                sys.executable, args.train_script,
                "--mode", "infer",
                "--model-ckpt", args.model_ckpt,
                "--infer-geojson-in", in_path,
                "--infer-out", out_path,
                "--zone", str(z),
                "--infer-slots", str(args.infer_slots),
                "--infer-knn",   str(args.infer_knn),
                "--infer-e-thr", str(args.infer_e_thr),
                "--infer-il-thr", str(args.infer_il_thr),
                "--infer-sv1-thr", str(args.infer_sv1_thr),
            ]
            if args.device:
                cmd += ["--device", args.device]
            if la_target is not None:
                cmd += ["--la-target", str(la_target)]
            if floors_avg is not None:
                cmd += ["--floors-avg", str(floors_avg)]
            if services_target:
                cmd += ["--services-target", json.dumps(services_target, ensure_ascii=False)]
            if args.config:
                cmd += ["--config", args.config]

            # ВАЖНО: zones/services НЕ передаём, пока вы явно не попросите — избежим рассинхрона с чекпойнтом
            if args.zones_json:
                cmd += ["--zones-json", args.zones_json]
            if args.services_json:
                cmd += ["--services-json", args.services_json]

            # инференс одного квартала
            try:
                subprocess.run(cmd, check=True, stdout=sys.stdout, stderr=sys.stderr)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] infer failed for block #{i} (zone={z}): {e}", file=sys.stderr)
                continue

            # читаем результат и добавляем трассировку источника
            try:
                blk_out = read_geojson(out_path)
                for b in blk_out.get("features", []):
                    props = dict(b.get("properties") or {})
                    props.setdefault("_source_zone", z)
                    props.setdefault("_source_block_index", i)
                    b["properties"] = props
                    out_feats.append(b)
            except Exception as e:
                print(f"[WARN] failed to read output for block #{i}: {e}", file=sys.stderr)
                continue

        # общий выход
        write_geojson(args.out, {"type":"FeatureCollection","features": out_feats})
        print(f"[OK] saved {len(out_feats)} buildings → {args.out}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
