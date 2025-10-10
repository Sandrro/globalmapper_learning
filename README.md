# GenBuilder

GenBuilder — алгоритм для обучения модели предсказания местоположения зданий и генерации застройки и ее атрибутов. Репозиторий включает утилиты для подготовки данных, обучения
графового генератора и пакетного инференса с последующей постобработкой.

## Подготовка окружения

Минимальная рабочая конфигурация описана в `environment.yml`: Python 3.7,
PyTorch 1.8 с CUDA 11.1, geopandas/shapely и вспомогательные библиотеки для
машинного обучения и работы с геоданными.

```bash
conda env create -f environment.yml
conda activate globalmapper
```

## Требования к данным

### Источники для трансформации

Этап `transform` ожидает слои кварталов и зданий в CRS, которые можно привести
к метрической системе координат, а также (опционально) точки сервисов. Ключевые
атрибуты перечислены ниже.

* **Кварталы (`--blocks`)** — многоугольники с уникальным `block_id` (если нет,
  будет сгенерирован). Необязательный столбец с типом зоны (`--zone-column` или
  один из автоопределяемых: `functional_zone_type_name`, `zone`, `class` и т.п.).
* **Здания (`--buildings`)** — полигоны с идентификатором здания. Желательно
  иметь этажность (`floors_num` или `storeys_count`), жилую площадь и флаги
  `is_living`/`has_floors`; при отсутствии будут добавлены дефолтные значения.
* **Сервисы (`--services`)**  — точки с типом объекта и полями
  вместимости. Перечень используемых полей можно задать через
  `--services-capacity-fields` и `--services-exclude`.

### Структура датасета для обучения

Скрипт обучения ожидает каталог `data_dir` с четырьмя parquet-файлами, которые
возвращает трансформационный конвейер, а также маски кварталов и JSON со списком
зон и сервисов.
```
data_dir/
  blocks.parquet      # block_id, zone, scale_l, mask_path
  branches.parquet    # block_id, branch_local_id, length
  nodes_fixed.parquet # block_id, slot_id, признаки зданий и сервисов
  edges.parquet       # block_id, src_slot, dst_slot
out/masks/            # <block_id>.png — бинарные маски кварталов
services.json         # описание сервисов (опционально)
zones.json            # справочник зон (если используется)
```

## Этап transform: подготовка данных

Файл `preprocessing.py` оркестрирует все стадии: присвоение зданий кварталам,
интеграцию сервисов, генерацию канонических признаков и сборку parquet-датасета.
Основной CLI выглядит так:

```bash
python preprocessing.py \
  --blocks data/blocks.gpkg \
  --buildings data/buildings.parquet \
  --target-crs EPSG:3857 \
  --out-dir out \
  --N 120 --mask-size 64 --num-workers 8 --block-timeout-sec 300 \
  --services data/services.geojson --out-services-json out/services.json \
  --min-branch-len 5.0 --branch-simplify-tol 0.2 \
  --knn-k 6
```

Выходные данные появляются в `--out-dir`:

* `blocks.parquet`, `branches.parquet`, `nodes_fixed.parquet`, `edges.parquet` —
  обучающий граф квартала.
* `masks/<block_id>.png` — маски кварталов для CNN-блока модели.
* `services.json` — схема сервисов, если они были переданы.
* Отчётные GeoJSON: `orphaned_buildings.geojson`, `orphaned_blocks.geojson`,
  `timeout.geojson` для диагностики качества данных.

## Этап train: обучение модели

Управляющий скрипт `train.py` поддерживает конфиги YAML (пример —
`train_gnn.yaml`) и параметры CLI. Ключевые опции: пути к датасету и маскам,
гиперпараметры обучения, а также режим работы (`train` или `infer`).

```bash
python train.py \
  --config train_gnn.yaml \
  --data-dir ./out \
  --mask-root ./out/masks \
  --model-ckpt ./out/model_graphgen.pt \
  --epochs 50 --batch-size 8 --device cuda
```

В процессе обучения логируются метрики (stdout + TensorBoard), а итоговая модель
и артефакты складываются по путям из конфига (`model_ckpt`, `./runs/<exp>_*`). При
наличии токена Hugging Face можно указать `--hf.repo-id`/`--hf.token`, чтобы
автоматически выгрузить модель.

## Этап inference: генерация кварталов

Скрипт `inference.py` выполняет пакетный инференс по набору кварталов, вызывает
`train.py` в режиме `--mode infer`, а затем формирует регулярную сетку, изолинии
плотности и итоговые полигоны зданий.

```bash
python inference.py \
  --blocks blocks.geojson \
  --model-ckpt ./out/model_graphgen.pt \
  --config train_gnn.yaml \
  --zones-json ./out/zones.json \
  --services-json ./out/services.json \
  --out-buildings buildings.geojson \
  --out-centroids centroids.geojson \
  --grid-size 15
```

Требования к входу:

* `--blocks` — GeoJSON с геометриями кварталов, `properties.block_id` и
  (по умолчанию) `properties.zone`. При отсутствии зоны можно указать иной атрибут
  через `--zone-attr`.
* Модель должна быть обучена на совместимом датасете, иначе распределение
  признаков не совпадёт.
* Опционально: JSON-словарь целевых показателей по зонам (`--targets-by-zone` или
  `--la-by-zone`), минимальные требования к сервисам (`--min-services`).

Результаты сохраняются в указанную директорию: прямоугольники зданий
`buildings.geojson`, сервисные территории, сетка и изолинии плотности, а также
центроиды (если передан `--out-centroids`).

## Быстрый чек-лист

1. Подготовьте окружение и исходные геоданные.
2. Запустите `preprocessing.py`, чтобы собрать parquet-датасет и маски.
3. Обучите модель `train.py` с нужными гиперпараметрами.
4. Выполните `inference.py` по новым кварталам и получите сгенерированные
   здания и дополнительные геометрические слои.
