from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union


# =============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================

def _round_int(x: float) -> int:
    return int(np.floor(x + 0.5))


def _to_metric_crs(
    gdf: gpd.GeoDataFrame,
    like: gpd.GeoDataFrame | None = None,
    fallback_epsg: int = 3857,
) -> gpd.GeoDataFrame:
    """Гарантирует метрическую CRS: приводим к CRS `like` или назначаем/проецируем в fallback."""
    if like is not None and like.crs is not None:
        try:
            return gdf.to_crs(like.crs)
        except Exception:
            # если у входа нет CRS — назначим
            if gdf.crs is None:
                return gdf.set_crs(like.crs, allow_override=True)
            raise
    if gdf.crs is None:
        return gdf.set_crs(epsg=fallback_epsg, allow_override=True)
    if getattr(gdf.crs, "is_projected", None) is True:
        return gdf
    return gdf.to_crs(epsg=fallback_epsg)


def _zone_cols(zones: gpd.GeoDataFrame) -> tuple[str, str]:
    """Определяем имена колонок идентификатора и имени зоны, создаём при отсутствии."""
    zid = "zone_id" if "zone_id" in zones.columns else ("id" if "id" in zones.columns else "ZONE_ID")
    if zid not in zones.columns:
        zones[zid] = np.arange(len(zones))
    zname = "zone"
    if zname not in zones.columns:
        for alt in ["zone_name", "zone_type", "functional_zone_type_name"]:
            if alt in zones.columns:
                zname = alt
                break
        else:
            zones["zone"] = "unknown"
            zname = "zone"
    return zid, zname


def _waterfill_with_caps(weights: np.ndarray, caps: np.ndarray, demand: float, eps: float = 1e-9) -> np.ndarray:
    """
    Распределяем объём `demand` пропорционально `weights` с верхними ограничениями `caps` для каждого элемента.
    Возвращает массив x, 0<=x<=caps, sum(x)<=demand. Итеративная схема с выбыванием насыщённых элементов.
    """
    n = len(weights)
    x = np.zeros(n, dtype=float)
    remain = np.arange(n)
    D = float(max(demand, 0.0))
    w = np.asarray(weights, dtype=float).copy()
    w[w < 0] = 0.0
    caps = np.asarray(caps, dtype=float)
    while len(remain) > 0 and D > eps:
        sw = float(w[remain].sum())
        if sw <= eps:
            break
        inc = np.zeros_like(x)
        # раздаём пропорционально весам, но не превышая оставшиеся локальные капы
        for i in remain:
            quota = D * (w[i] / sw)
            inc[i] = min(quota, caps[i] - x[i])
        inc_sum = float(inc[remain].sum())
        if inc_sum <= eps:
            break
        x += inc
        D -= inc_sum
        remain = np.array([i for i in remain if (caps[i] - x[i]) > eps], dtype=int)
    return x


# =============================
# ОСНОВНОЙ КЛАСС
# =============================

@dataclass
class LivingAreaAllocator:
    """
    Расчёт этажности и распределение жилой площади по жилым домам.

    Правила:
      1) Для жилых: floors_count = round(area_m2 / area_per_floor_m2), затем клип в [1, max_floor(zone)].
      2) Жилая площадь: living_area = K * area_m2 * floors_count, где K ∈ [K_min, K_max].
      3) Таргет по зоне (`target_living_area`) распределяется по весам ~ 1 / distance_to_zone_boundary
         (точка представителя полигона), с соблюдением K-ограничений (водонапорная модель).
      4) Считаем ТОЛЬКО для зон, где указан target_living_area; прочие — living_area=0, K=NaN.
      5) Этажность сервисов фиксирована (kindergarten=2, school=3, polyclinics=4), но не превышает max_floor(zone).

    Вход метода `fit_transform`:
      - buildings_gdf: GeoDataFrame (колонки: geometry, service (living_house/school/kindergarten/polyclinics/...),
        опционально zone_id/zone и т.д.).
      - zones_gdf: GeoDataFrame зон; будет приведён к CRS зданий; zone_id/zone определяются автоматически.

    Выход:
      - dict с ключами:
          * buildings: GeoDataFrame с добавленными колонками area_m2, floors_count, K, living_area, _alloc_status
          * summary:   DataFrame по зонам: target_requested, target_used, cap_min, cap_max, n_buildings, mean_K, sum_living_area, status
    """

    # Базовые параметры
    area_per_floor_m2: float = 100.0    # для перевода площади пятна в этажность (правило 1)
    K_min: float = 0.5                   # нижняя граница K
    K_max: float = 0.75                  # верхняя граница K
    default_zone_max_floor: int = 9      # если по зоне нет max_floor

    # Фиксированные этажности сервисов
    service_fixed_floors: Dict[str, int] = field(default_factory=lambda: {
        "kindergarten": 2,
        "school": 3,
        "polyclinics": 4,
    })

    # Параметры зон (любой из словарей может быть пустым)
    zone_params_by_id: Dict[int, Dict[str, float | int]] = field(default_factory=dict)  # {zid: {"target_living_area": float, "max_floor": int}}
    zone_params_by_name: Dict[str, Dict[str, float | int]] = field(default_factory=dict)  # {zname_lower: {...}}

    # Численная устойчивость
    dist_eps: float = 1e-6

    # CRS
    fallback_epsg: int = 3857

    # Логирование
    verbose: bool = True

    # -----------------
    # ПУБЛИЧНО: API
    # -----------------
    def fit_transform(
        self,
        buildings_gdf: gpd.GeoDataFrame,
        zones_gdf: gpd.GeoDataFrame,
    ) -> Dict[str, object]:
        # CRS/копии
        buildings = buildings_gdf.copy()
        zones = _to_metric_crs(zones_gdf.copy(), like=buildings, fallback_epsg=self.fallback_epsg)
        buildings = _to_metric_crs(buildings, like=zones, fallback_epsg=self.fallback_epsg)

        # Определяем колонки зон
        zid_col, zname_col = _zone_cols(zones)

        # Если в buildings нет zone_id/name или есть пропуски — подставим по representative_point within зоны
        need_zone_join = (zid_col not in buildings.columns) or buildings[zid_col].isna().any()
        if need_zone_join:
            cent = buildings.geometry.representative_point()
            j = gpd.sjoin(
                gpd.GeoDataFrame({"i": np.arange(len(buildings))}, geometry=cent, crs=buildings.crs),
                zones[[zid_col, zname_col, "geometry"]],
                how="left", predicate="within"
            ).drop_duplicates("i").set_index("i")[[zid_col, zname_col]]
            buildings = buildings.drop(columns=[zid_col, zname_col], errors="ignore").join(j, how="left")

        # Нормализуем имя зоны
        buildings[zname_col] = buildings[zname_col].astype(str).str.lower().str.strip()

        # Тип объекта
        service_series = buildings.get("service").astype(str).str.lower()
        is_living = service_series.eq("living_house")

        # Геометрическая площадь
        buildings["area_m2"] = buildings.geometry.area.astype(float)

        # Параметры зоны по каждой записи: target (может быть NaN), max_floor (int)
        def _resolve_zone_params(row) -> tuple[float, int]:
            zid = row.get(zid_col)
            zname = row.get(zname_col)
            targ = None
            mf = None
            if pd.notna(zid):
                zid_int = int(zid)
                if zid_int in self.zone_params_by_id:
                    p = self.zone_params_by_id[zid_int]
                    targ = p.get("target_living_area", targ)
                    mf = p.get("max_floor", mf)
            if (targ is None or mf is None) and pd.notna(zname):
                p = self.zone_params_by_name.get(str(zname).lower())
                if p:
                    if targ is None:
                        targ = p.get("target_living_area")
                    if mf is None:
                        mf = p.get("max_floor")
            if mf is None:
                mf = self.default_zone_max_floor
            return (float(targ) if targ is not None else np.nan), int(mf)

        params = buildings.apply(_resolve_zone_params, axis=1, result_type="expand")
        buildings["target_zone_area"], buildings["max_floor_zone"] = params[0].values, params[1].values

        # --- ЭТАЖНОСТЬ ---
        # 1) Жилые: floors = round(area/area_per_floor_m2), клип [1, max_floor_zone]
        floors_living = np.maximum(
            1,
            np.minimum(
                buildings.loc[is_living, "max_floor_zone"].astype(int).values,
                np.vectorize(_round_int)(buildings.loc[is_living, "area_m2"].values / float(self.area_per_floor_m2)),
            ),
        )
        buildings.loc[is_living, "floors_count"] = floors_living

        # 2) Сервисы: фиксированные этажности (клип по max_floor)
        for svc_name, fixed in self.service_fixed_floors.items():
            m = service_series.eq(svc_name)
            if m.any():
                maxf = buildings.loc[m, "max_floor_zone"].astype(int)
                buildings.loc[m, "floors_count"] = np.minimum(int(fixed), maxf).astype(int)

        # Приводим тип floors_count
        buildings["floors_count"] = pd.to_numeric(buildings["floors_count"], errors="coerce").astype("Int64")

        # --- РАСПРЕДЕЛЕНИЕ ЖИЛОЙ ПЛОЩАДИ ---
        buildings["K"] = np.nan
        buildings["living_area"] = 0.0

        # Границы зон, для расчёта весов
        zones["_boundary"] = zones.geometry.boundary

        summary_rows: List[Dict] = []

        valid_living = buildings[is_living & buildings[zid_col].notna()].copy()
        for zid, sub in valid_living.groupby(valid_living[zid_col].astype(int)):
            zid_int = int(zid)
            Z = zones.loc[zones[zid_col] == zid_int]
            if len(Z) == 0:
                continue
            # Целевая площадь для зоны (явно заданная)
            targ = sub["target_zone_area"].dropna().iloc[0] if sub["target_zone_area"].notna().any() else np.nan
            if np.isnan(targ) or targ <= 0:
                # Нет таргета → living_area=0, K=NaN (для всех домов в зоне)
                buildings.loc[sub.index, "_alloc_status"] = "no_target → living=0"
                summary_rows.append({
                    "zone_id": zid_int,
                    "target_requested": float(targ) if not np.isnan(targ) else None,
                    "target_used": 0.0,
                    "cap_min": float((self.K_min * sub["area_m2"] * sub["floors_count"]).sum()),
                    "cap_max": float((self.K_max * sub["area_m2"] * sub["floors_count"]).sum()),
                    "n_buildings": int(len(sub)),
                    "mean_K": np.nan,
                    "sum_living_area": 0.0,
                    "status": "no_target",
                })
                continue

            # Капасити по K
            cap_min = self.K_min * sub["area_m2"] * sub["floors_count"]
            cap_max = self.K_max * sub["area_m2"] * sub["floors_count"]
            cap_min_tot = float(cap_min.sum())
            cap_max_tot = float(cap_max.sum())

            # Ограничиваем таргет, чтобы остаться в [cap_min, cap_max]
            T = float(np.clip(targ, cap_min_tot, cap_max_tot))

            # Веса: ближе к границе → больше вес
            boundary = unary_union(list(Z["_boundary"].values))
            dists = sub.geometry.representative_point().distance(boundary).astype(float).values
            weights = 1.0 / (self.dist_eps + dists)
            if not np.isfinite(weights).any() or weights.sum() == 0.0:
                weights = np.ones_like(dists, dtype=float)

            deltas = (cap_max - cap_min).astype(float).values
            D = T - cap_min_tot
            x = _waterfill_with_caps(weights, deltas, D)

            living = cap_min.values + x
            denom = (sub["area_m2"].values * sub["floors_count"].values)
            K = np.divide(living, denom, out=np.zeros_like(living), where=denom > 0)
            K = np.clip(K, self.K_min, self.K_max)

            buildings.loc[sub.index, "K"] = K
            buildings.loc[sub.index, "living_area"] = living
            status = f"T_used={T:.2f}; cap_min={cap_min_tot:.2f}; cap_max={cap_max_tot:.2f}"
            buildings.loc[sub.index, "_alloc_status"] = status

            summary_rows.append({
                "zone_id": zid_int,
                "target_requested": float(targ),
                "target_used": float(T),
                "cap_min": cap_min_tot,
                "cap_max": cap_max_tot,
                "n_buildings": int(len(sub)),
                "mean_K": float(np.mean(K)) if len(K) else np.nan,
                "sum_living_area": float(np.sum(living)),
                "status": status,
            })

        # Для сервисов — статус по умолчанию
        is_service = service_series.isin(list(self.service_fixed_floors.keys()))
        buildings.loc[is_service, "_alloc_status"] = buildings.loc[is_service, "_alloc_status"].fillna("service → living=0")

        summary_df = pd.DataFrame(summary_rows)

        # Возвращаем без временных колонок в зонах
        zones.drop(columns=["_boundary"], inplace=True, errors="ignore")

        if self.verbose:
            meanK = summary_df["mean_K"].mean(skipna=True) if len(summary_df) > 0 else float("nan")
            print(
                f"OK | objects={len(buildings)}, living_houses={int(is_living.sum())}, "
                f"zones_with_target={summary_df.shape[0]} | mean(K)={meanK:.3f}"
            )

        return {
            "buildings": buildings,
            "summary": summary_df,
        }
