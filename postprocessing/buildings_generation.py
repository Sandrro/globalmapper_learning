from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from shapely.ops import unary_union

# ------------------------------
# ВСПОМОГАТЕЛЬНЫЕ ОБЁРТКИ SJOIN
# ------------------------------
def _sjoin(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, predicate: str, how: str = "left", **kwargs) -> gpd.GeoDataFrame:
    """Совместимость gpd.sjoin для разных версий geopandas."""
    try:
        return gpd.sjoin(left, right, predicate=predicate, how=how, **kwargs)
    except TypeError:  # старые версии
        return gpd.sjoin(left, right, op=predicate, how=how, **kwargs)


def _make_valid(g):
    try:
        gg = g.buffer(0)
        return gg if not gg.is_empty else g
    except Exception:
        return g


@dataclass
class BuildingPlacer:
    """
    Размещение сервисных зданий и участков на регулярной сетке клеток.

    Входные данные:
        - cells_gdf: GeoDataFrame клеток. Требуемые колонки:
            * geometry (Polygon ячейки)
            * inside_iso_closed: bool — принадлежность «застраиваемой» территории (после правила 3 сторон)
            * iso_level: int/float — уровень кольца (может быть NaN вне изолиний)
          Необязательные: zone_id/zone/name и т.п. (по желанию).

        - zones_gdf: полигоны кварталов с идентификатором (автодетект), будут приклеены к клеткам `within`.

    Выход:
        dict(
            cells=GeoDataFrame,             # клетки с финальными флагами и назначенными сервисами
            buildings_rects=GeoDataFrame,   # корпуса (жилые + сервисы) до merge
            buildings_merged=GeoDataFrame,  # объединённые полигоны по типу service
            service_sites=GeoDataFrame,     # прямоугольные участки сервисов
            metrics=pd.DataFrame            # метрики равномерности по участкам
        )

    Параметры и правила поддерживают:
        - Прямоугольные участки (site) как набор клеток, корпус внутри core окна участка (внутренний буфер >= 1 клетка).
        - Зазоры Chebyshev: до домов >= 1 клетка (т.е. d>=2), между любыми участками >=1 клетка, между участками одного типа >=10 клеток.
        - Раунд‑робин по типам: за один цикл ставится по одному объекту каждого типа в зоне.
        - Кандидаты жилых клеток фиксируются до размещения сервисов.

    Примечание: все расстояния и размеры выражены в «клетках»/метрах согласно CRS сетки.
    """

    # --- Параметры алгоритма ---
    max_run: int = 8
    neigh_empty_thr: int = 3
    cell_size_m: float = 15.0
    edge_share_frac: float = 0.2  # порог для соседства по стороне (доля характерного ребра)

    # Merge итоговых корпусов
    merge_predicate: str = "intersects"
    merge_fix_eps: float = 0.0

    # Лимиты сервисов на квартал
    max_services_per_zone: Dict[str, int] = field(default_factory=lambda: {"school": 3, "kindergarten": 5, "polyclinics": 1})

    # Рандомизация форм корпусов/участков
    randomize_service_forms: bool = True
    service_random_seed: int = 42

    # Зазоры (в клетках Чебышёва)
    gap_to_houses_cheb: int = 2         # до домов >=1 клетка (Chebyshev >=2)
    gap_between_sites_cheb: int = 2     # между любыми участками
    same_type_site_gap_cheb: int = 10   # между участками одного типа (НОВОЕ)

    # Корпус внутри участка с внутренним отступом >= 1 клетка
    inner_margin_cells: int = 1

    # Нормативы участков (правила площадей/вместимости)
    service_site_rules: Dict[Tuple[str, str], Dict[str, float | int]] = field(default_factory=lambda: {
        ("school", "RECT_5x2_WITH_OPEN_3"):   {"capacity": 600,  "site_area_m2": 33000.0},
        ("school", "H_5x4"):                   {"capacity": 800,  "site_area_m2": 36000.0},
        ("school", "RING_5x5_WITH_COURTYARD"): {"capacity": 1100, "site_area_m2": 39600.0},
        ("kindergarten", "LINE3"):             {"capacity": 60,   "site_area_m2": 2640.0},
        ("kindergarten", "W5"):                {"capacity": 100,  "site_area_m2": 4400.0},
        ("kindergarten", "H7"):                {"capacity": 150,  "site_area_m2": 5700.0},
        ("polyclinics", "RECT_2x4"):           {"capacity": 300,  "site_area_m2": 3000.0},
    })

    # Сервисный порядок в раунде
    svc_order: List[str] = field(default_factory=lambda: ["school", "kindergarten", "polyclinics"])

    # Колонки зон
    zone_id_col: str = "zone_id"
    zone_name_col: str = "zone"

    # Логирование
    verbose: bool = True

    # --------------------------
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # --------------------------
    def fit_transform(
        self,
        cells_gdf: gpd.GeoDataFrame,
        zones_gdf: gpd.GeoDataFrame,
        zone_name_aliases: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Главный метод: размещает участки и корпуса сервисов, агрегирует жилые и сервисные полигоны
        и возвращает все результаты без файлового ввода/вывода.
        """
        if zone_name_aliases is None:
            zone_name_aliases = ["functional_zone_type_name", "zone_type", "zone_name"]

        # Копии и базовая проверка
        cells = cells_gdf.copy().reset_index(drop=True)
        if "inside_iso_closed" not in cells.columns:
            raise ValueError("Во входном слое клеток отсутствует колонка 'inside_iso_closed'.")
        cells["inside_iso_closed"] = cells["inside_iso_closed"].fillna(False).astype(bool)
        cells["iso_level"] = pd.to_numeric(cells.get("iso_level"), errors="coerce")
        cells["service"] = None  # будет заполнено

        # Индексы сетки
        row_i, col_j, x0, y0, step_est = self._grid_indices(cells)
        cells["row_i"], cells["col_j"] = row_i, col_j

        # Подготовка и приклейка зон
        zones = zones_gdf.to_crs(cells.crs).reset_index(drop=True)
        zid_col = self.zone_id_col
        if self.zone_id_col not in zones.columns:
            if "id" in zones.columns:
                zones[zid_col] = zones["id"]
            else:
                zones[zid_col] = np.arange(len(zones))
        zname_col = self.zone_name_col
        if zname_col not in zones.columns:
            for alt in zone_name_aliases:
                if alt in zones.columns:
                    zones[zname_col] = zones[alt]
                    break
        if zname_col not in zones.columns:
            zones[zname_col] = "unknown"

        cells_zone = _sjoin(
            cells[["geometry"]].reset_index().rename(columns={"index": "cell_idx"}),
            zones[[zid_col, zname_col, "geometry"]],
            predicate="within", how="left"
        ).drop_duplicates("cell_idx")

        cells = cells.merge(
            cells_zone[["cell_idx", zid_col, zname_col]],
            left_index=True, right_on="cell_idx", how="left"
        ).drop(columns=["cell_idx"])

        cells[zname_col] = cells[zname_col].astype(str).str.lower().str.strip()
        cells["is_residential_zone"] = cells[zname_col].eq("residential")

        # Соседства/внешность
        neighbors_all, neighbors_side, neighbors_diag, empty_neighs, missing = self._compute_neighbors(cells)

        # 1) Фиксация жилых клеток
        inside_mask = cells["inside_iso_closed"].values
        is_external = np.zeros(len(cells), dtype=bool)
        for i in range(len(cells)):
            if not inside_mask[i]:
                continue
            is_external[i] = (empty_neighs[i] >= self.neigh_empty_thr) or (missing[i] > 0)
        cells["candidate_building"] = inside_mask & is_external
        cells["candidate_building"] = self._enforce_line_blocks(cells, "row_i", "col_j", "candidate_building", self.max_run)
        cells["is_building"] = self._enforce_line_blocks(cells, "col_j", "row_i", "candidate_building", self.max_run)

        # Промо внутренних соседей для «малых граничных»
        bbox = np.array([
            cells.geometry.bounds.minx.values,
            cells.geometry.bounds.miny.values,
            cells.geometry.bounds.maxx.values,
            cells.geometry.bounds.maxy.values,
        ]).T
        w = bbox[:, 2] - bbox[:, 0]
        h = bbox[:, 3] - bbox[:, 1]
        is_small = (w < self.cell_size_m - 1e-6) | (h < self.cell_size_m - 1e-6)
        external_score = empty_neighs + missing

        is_b = cells["is_building"].to_numpy().astype(bool)
        promote_targets: List[int] = []
        for i in range(len(cells)):
            if not (is_b[i] and is_external[i] and is_small[i]):
                continue
            cand = [j for j in neighbors_side.get(i, []) if inside_mask[j]]
            if not cand:
                cand = [j for j in neighbors_all.get(i, []) if inside_mask[j]]
            cand = [j for j in cand if not is_b[j]]
            if not cand:
                continue
            j_best = min(cand, key=lambda j: (external_score[j], -empty_neighs[j]))
            promote_targets.append(j_best)
        if promote_targets:
            for j in promote_targets:
                is_b[j] = True
            cells["is_building"] = is_b
            cells["is_building"] = self._enforce_line_blocks(cells, "row_i", "col_j", "is_building", self.max_run)
            cells["is_building"] = self._enforce_line_blocks(cells, "col_j", "row_i", "is_building", self.max_run)

        # 2) Группировка по зонам и подготовка
        idx_by_rc = {(int(r), int(c)): i for i, (r, c) in enumerate(zip(cells["row_i"], cells["col_j"]))}
        is_res = cells["is_residential_zone"].fillna(False).values
        not_house = ~cells["is_building"].fillna(False).values
        inside_true = cells["inside_iso_closed"].fillna(False).values
        inside_false = ~inside_true
        ok_iso = (cells["iso_level"].fillna(0).values >= 0)

        zone_groups: Dict[int, Dict] = {}
        zids_series = cells[zid_col].astype("Int64") if zid_col in cells.columns else pd.Series([None] * len(cells))
        for zid, sub_all in cells[pd.notna(zids_series)].groupby(zids_series):
            zid_int = int(zid)
            idxs = sub_all.index
            in_ids = sub_all.index[(is_res[idxs]) & (inside_true[idxs]) & ok_iso[idxs] & not_house[idxs]].to_list()
            out_ids = sub_all.index[(is_res[idxs]) & (inside_false[idxs]) & not_house[idxs]].to_list()
            if not in_ids and not out_ids:
                continue
            sub_res = sub_all.index[is_res[idxs]].to_list()
            r_center = float(cells.loc[sub_res, "row_i"].median()) if len(sub_res) else float(sub_all["row_i"].median())
            c_center = float(cells.loc[sub_res, "col_j"].median()) if len(sub_res) else float(sub_all["col_j"].median())
            Lmax = int(pd.to_numeric(cells.loc[in_ids, "iso_level"], errors="coerce").fillna(0).max()) if in_ids else 0
            zone_groups[zid_int] = {
                "inside_ids": in_ids,
                "outside_ids": out_ids,
                "r_center": r_center,
                "c_center": c_center,
                "Lmax": Lmax,
            }

        svc_count_by_zone = {int(z): {s: 0 for s in self.svc_order} for z in zone_groups.keys()}
        rng = random.Random(self.service_random_seed)

        # Паттерны корпусов и вариации форм
        lib = self._pattern_library()
        shape_variants: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]] = {svc: [] for svc in lib.keys()}
        for svc, shapes in lib.items():
            items = list(shapes)
            if self.randomize_service_forms:
                rng.shuffle(items)
            for name, offsets, allow_rot in items:
                vars_ = self._shape_variants(offsets, allow_rot)
                if self.randomize_service_forms:
                    rng.shuffle(vars_)
                shape_variants[svc].append((name, vars_))

        # Глобальные наборы занятых клеток
        reserved_site_cells: set[int] = set()     # клетки — территории
        reserved_service_cells: set[int] = set()  # клетки — корпуса

        # Коллекторы геометрий
        service_sites_geom: List = []
        service_sites_attrs: List[Dict] = []
        service_polys_geom: List = []
        service_polys_attrs: List[Dict] = []

        placed_site_sets: List[List[int]] = []
        placed_sites_by_type: Dict[str, List[List[int]]] = {s: [] for s in self.svc_order}

        # --- Основной цикл: по зонам и раунд‑робин по типам ---
        for zid, meta in zone_groups.items():
            r_cen, c_cen = meta["r_center"], meta["c_center"]
            Lmax = meta["Lmax"]
            z_inside = meta["inside_ids"]
            z_outside = meta["outside_ids"]

            limits = {svc: self.max_services_per_zone.get(svc, 0) for svc in self.svc_order}

            while True:
                progress = False
                for svc in self.svc_order:
                    if svc_count_by_zone[zid][svc] >= limits[svc]:
                        continue
                    placed_here = False

                    # Пробуем внутри изолиний по уровням (от Lmax вниз)
                    for L in range(Lmax, -1, -1):
                        allowed_ids = [
                            i for i in z_inside
                            if (i not in reserved_site_cells)
                            and (not cells.at[i, "is_building"])  # не дом
                            and pd.notna(cells.at[i, "iso_level"]) and (int(cells.at[i, "iso_level"]) >= L)
                        ]
                        if not allowed_ids:
                            continue
                        if self._try_place_site_and_service_in_zone_level(
                            cells, zid, svc, allowed_ids, r_cen, c_cen,
                            placed_site_sets, placed_sites_by_type, rng,
                            shape_variants, idx_by_rc,
                            reserved_site_cells, reserved_service_cells,
                            neighbors_all,
                            service_sites_geom, service_sites_attrs,
                            service_polys_geom, service_polys_attrs,
                        ):
                            svc_count_by_zone[zid][svc] += 1
                            placed_here = True
                            break

                    # Если не получилось внутри — пытаемся снаружи
                    if not placed_here and z_outside:
                        if self._try_place_site_and_service_fallback_outside(
                            cells, zid, svc, z_outside, r_cen, c_cen,
                            placed_site_sets, placed_sites_by_type, rng,
                            shape_variants, idx_by_rc,
                            reserved_site_cells, reserved_service_cells,
                            neighbors_all,
                            service_sites_geom, service_sites_attrs,
                            service_polys_geom, service_polys_attrs,
                        ):
                            svc_count_by_zone[zid][svc] += 1
                            placed_here = True

                    if placed_here:
                        progress = True  # За проход ровно один объект данного типа

                if not progress:
                    break

        # --- Сборка жилых полигонов ---
        is_b = cells["is_building"].to_numpy().astype(bool)
        # Разбиение на диагональные компоненты (без реберных соседей)
        diag_only_nodes: List[int] = []
        for i in range(len(cells)):
            if not is_b[i]:
                continue
            side_in = any(is_b[j] for j in neighbors_side.get(i, []))
            diag_in = any(is_b[j] for j in neighbors_all.get(i, []) if j in neighbors_diag.get(i, []))
            if (not side_in) and diag_in:
                diag_only_nodes.append(i)

        diag_adj = {i: [j for j in neighbors_diag.get(i, []) if is_b[j]] for i in range(len(cells))}
        diag_components = self._components(diag_only_nodes, diag_adj)
        diag_components = [comp for comp in diag_components if len(comp) >= 2]

        cells["is_diag_only"] = False
        for comp in diag_components:
            for i in comp:
                cells.at[i, "is_diag_only"] = True

        records_geom, records_attrs = [], []
        try:
            from shapely.geometry import CAP_STYLE, JOIN_STYLE
            cap_style = CAP_STYLE.flat
            join_style = JOIN_STYLE.mitre
        except Exception:
            cap_style = 2
            join_style = 2

        buf_dist = float(self.cell_size_m) / 2.0
        centroids = cells.geometry.centroid

        # Диагональные жилые → полоса-буфер по PCA-упорядоченным центроидам
        for k, comp in enumerate(diag_components, start=1):
            pts = np.array([[centroids[i].x, centroids[i].y] for i in comp], dtype=float)
            if len(pts) < 2:
                continue
            order = self._pca_order(pts)
            ordered_pts = pts[order]
            ordered_pts = ordered_pts[np.concatenate(([True], np.any(np.diff(ordered_pts, axis=0) != 0, axis=1)))]
            if len(ordered_pts) < 2:
                continue
            line = LineString(ordered_pts)
            poly = line.buffer(buf_dist, cap_style=cap_style, join_style=join_style)
            records_geom.append(_make_valid(poly))
            records_attrs.append({
                "building_id": f"D{str(k).zfill(5)}",
                "type": "diag_buffer",
                "service": "living_house",
                "n_cells": int(len(comp)),
                "width_m": float(self.cell_size_m),
            })

        # Простые жилые клетки
        simple_ids = [i for i in range(len(cells)) if cells.at[i, "is_building"] and not cells.at[i, "is_diag_only"]]
        for i in simple_ids:
            records_geom.append(_make_valid(cells.geometry[i]))
            zid_mode = int(cells.at[i, zid_col]) if zid_col in cells.columns and not pd.isna(cells.at[i, zid_col]) else None
            records_attrs.append({
                "building_id": f"C{str(i).zfill(6)}",
                "type": "cell",
                "service": "living_house",
                "n_cells": 1,
                "width_m": float(self.cell_size_m),
                "row_i": int(cells.at[i, "row_i"]),
                "col_j": int(cells.at[i, "col_j"]),
                zid_col: zid_mode,
            })

        # Добавляем корпуса сервисов
        records_geom.extend(service_polys_geom)
        records_attrs.extend(service_polys_attrs)

        buildings_rects = gpd.GeoDataFrame(records_attrs, geometry=records_geom, crs=cells.crs).reset_index(drop=True)

        # --- MERGE по типам (не склеиваем разные service) ---
        left = buildings_rects[["geometry"]].reset_index().rename(columns={"index": "i"})
        right = buildings_rects[["geometry"]].reset_index().rename(columns={"index": "j"})
        pairs = _sjoin(left, right, predicate=self.merge_predicate)
        pairs = pairs[(pairs["i"] != pairs["j"]) & pairs["j"].notna()].copy()
        pairs["j"] = pairs["j"].astype(int)

        svc_vals = buildings_rects.get("service", pd.Series(["living_house"] * len(buildings_rects))).astype(object).tolist()
        def _same_group(i, j):
            return svc_vals[i] == svc_vals[j]

        pairs = pairs[pairs.apply(lambda r: _same_group(int(r["i"]), int(r["j"])), axis=1)]
        adj = {i: [] for i in range(len(buildings_rects))}
        for a, b in pairs[["i", "j"]].itertuples(index=False):
            a = int(a); b = int(b)
            adj[a].append(b); adj[b].append(a)
        groups = self._components(list(adj.keys()), adj)

        merged_geoms, merged_attrs = [], []
        for gid, comp in enumerate(groups):
            geoms = buildings_rects.geometry.iloc[comp].tolist()
            if self.merge_fix_eps and self.merge_fix_eps > 0:
                u = unary_union([_make_valid(g.buffer(self.merge_fix_eps)) for g in geoms]).buffer(-self.merge_fix_eps)
            else:
                u = unary_union([_make_valid(g) for g in geoms])
            comp_svc = list({svc_vals[i] for i in comp})
            merged_service = comp_svc[0] if len(comp_svc) == 1 else "mixed"
            types = ",".join(sorted(set(buildings_rects.loc[comp, "type"].astype(str).tolist())))
            zid_mode = None
            if zid_col in buildings_rects.columns:
                zids = buildings_rects.loc[comp, zid_col].dropna().astype(int)
                if len(zids) > 0:
                    zid_mode = int(zids.value_counts().index[0])
            merged_geoms.append(_make_valid(u))
            merged_attrs.append({
                "group_id": int(gid),
                "n_members": int(len(comp)),
                "n_cells_sum": int(np.nansum(buildings_rects.loc[comp, "n_cells"].values)) if "n_cells" in buildings_rects else None,
                "types": types,
                "service": merged_service,
                zid_col: zid_mode,
            })

        buildings_merged = gpd.GeoDataFrame(merged_attrs, geometry=merged_geoms, crs=cells.crs).reset_index(drop=True)
        buildings_merged = buildings_merged[buildings_merged.area > self.cell_size_m * self.cell_size_m]

        # --- Метрики равномерности по участкам ---
        service_sites_gdf = gpd.GeoDataFrame(service_sites_attrs, geometry=service_sites_geom, crs=cells.crs).reset_index(drop=True)
        metrics_rows: List[Dict] = []
        if len(service_sites_gdf) > 0:
            service_sites_gdf["centroid"] = service_sites_gdf.geometry.representative_point()
            for (zid, svc), sub in service_sites_gdf.groupby([zid_col, "service"], dropna=True):
                pts = list(sub["centroid"])
                n = len(pts)
                if n == 1:
                    min_nn = float("inf"); mean_nn = float("inf"); score = 1.0
                else:
                    dmat = np.zeros((n, n), dtype=float)
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = pts[i].distance(pts[j])
                            dmat[i, j] = dmat[j, i] = d
                    nn = np.min(np.where(dmat == 0, np.inf, dmat), axis=1)
                    min_nn = float(np.min(nn)); mean_nn = float(np.mean(nn))
                    score = float(mean_nn / (self.cell_size_m))
                metrics_rows.append({
                    zid_col: int(zid) if pd.notna(zid) else None,
                    "service": svc,
                    "sites_count": int(n),
                    "min_nn_m": None if math.isinf(min_nn) else float(min_nn),
                    "mean_nn_m": None if math.isinf(mean_nn) else float(mean_nn),
                    "uniformity_score": score,
                })
        metrics_df = pd.DataFrame(metrics_rows)

        # Финальные статусы в клетках
        cells.loc[cells["is_building"].fillna(False), "service"] = cells.loc[cells["is_building"].fillna(False), "service"].fillna("living_house")

        if self.verbose:
            svc_rects = buildings_rects[buildings_rects.get("service").isin(self.svc_order)]
            print(
                f"OK | cells={len(cells)}, living={int(cells['is_building'].sum())}, diag-only={int(cells['is_diag_only'].sum())}\n"
                f"Rects={len(buildings_rects)} (service rects: {len(svc_rects)}) | Merged={len(buildings_merged)} | Sites={len(service_sites_gdf)}"
            )

        return {
            "cells": cells,
            "buildings_rects": buildings_rects,
            "buildings_merged": buildings_merged,
            "service_sites": service_sites_gdf.drop(columns=["centroid"], errors="ignore"),
            "metrics": metrics_df,
        }

    # --------------------------
    # ХЕЛПЕРЫ: геометрия/индексы
    # --------------------------
    def _grid_indices(self, gdf: gpd.GeoDataFrame):
        c = gdf.geometry.centroid
        x = c.x.values; y = c.y.values
        cell_step = np.median(np.sqrt(np.maximum(gdf.geometry.area.values, 1e-9)))
        x0, y0 = float(np.min(x)), float(np.min(y))
        col = np.rint((x - x0) / cell_step).astype(int)
        row = np.rint((y - y0) / cell_step).astype(int)
        return row, col, x0, y0, float(cell_step)

    def _compute_neighbors(self, cells: gpd.GeoDataFrame):
        """Соседство по стороне/диагонали и подсчёт пустых соседей. Надёжная версия с проверкой длины общего ребра."""
        left = cells[["geometry"]].reset_index().rename(columns={"index": "ida"})
        right = cells[["geometry"]].reset_index().rename(columns={"index": "idb"})
        pairs = _sjoin(left, right, predicate="touches", how="left")
        if "idb" not in pairs.columns and "index_right" in pairs.columns:
            pairs = pairs.rename(columns={"index_right": "idb"})
        pairs = pairs[(pairs["ida"] != pairs["idb"]) & pairs["idb"].notna()].copy()

        neighbors_all = {i: [] for i in range(len(cells))}
        neighbors_side = {i: [] for i in range(len(cells))}
        neighbors_diag = {i: [] for i in range(len(cells))}

        if len(pairs) > 0:
            pairs["idb"] = pairs["idb"].astype(int)
            geom_list = list(cells.geometry.values)
            edge_len_est = np.sqrt(np.maximum(cells.geometry.area.values, 1e-9))
            thr_len = self.edge_share_frac * edge_len_est

            def _is_edge_neighbor(a: int, b: int) -> bool:
                try:
                    inter = geom_list[a].boundary.intersection(geom_list[b].boundary)
                    length = getattr(inter, "length", 0.0)
                except Exception:
                    length = 0.0
                return length >= min(thr_len[a], thr_len[b])

            for a, b in pairs[["ida", "idb"]].itertuples(index=False):
                a = int(a); b = int(b)
                eok = _is_edge_neighbor(a, b)
                neighbors_all[a].append(b)
                (neighbors_side if eok else neighbors_diag)[a].append(b)

            # симметричное наполнение
            for i in range(len(cells)):
                for dct in (neighbors_all, neighbors_side, neighbors_diag):
                    for j in list(dct[i]):
                        if i not in dct[j]:
                            dct[j].append(i)

        inside = cells["inside_iso_closed"].fillna(False).to_numpy().astype(bool)
        empty_neighs = np.zeros(len(cells), dtype=int)
        missing = np.zeros(len(cells), dtype=int)
        for i in range(len(cells)):
            nn = list(set(neighbors_all.get(i, [])))
            empty_neighs[i] = sum(1 for j in nn if not inside[j])
            missing[i] = max(0, 8 - len(nn))
        return neighbors_all, neighbors_side, neighbors_diag, empty_neighs, missing

    def _enforce_line_blocks(self, df: gpd.GeoDataFrame, line_key: str, order_key: str, mask_key: str, max_run: int) -> pd.Series:
        out = df[mask_key].copy()
        for _, sub in df.loc[df[mask_key]].groupby(line_key):
            sub = sub.sort_values(order_key)
            idx = sub.index.to_numpy(); ordv = sub[order_key].to_numpy()
            breaks = np.where(np.diff(ordv) != 1)[0] + 1
            segments = np.split(np.arange(len(ordv)), breaks)
            for seg in segments:
                if len(seg) <= max_run:
                    continue
                run = 0; place_gap = False
                for k in seg:
                    i = idx[k]
                    if place_gap:
                        out.loc[i] = False; place_gap = False; run = 0
                    else:
                        if run < max_run:
                            out.loc[i] = True; run += 1
                            if run == max_run:
                                place_gap = True
                        else:
                            out.loc[i] = False; run = 0
        return out

    def _components(self, nodes: List[int], adj: Dict[int, List[int]]) -> List[List[int]]:
        node_set = set(nodes); seen, comps = set(), []
        for v in nodes:
            if v in seen:
                continue
            stack, comp = [v], [v]
            seen.add(v)
            while stack:
                u = stack.pop()
                for w in adj.get(u, []):
                    if w in node_set and w not in seen:
                        seen.add(w); stack.append(w); comp.append(w)
            comps.append(comp)
        return comps

    def _pca_order(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) <= 2:
            return np.argsort(pts[:, 0] + pts[:, 1])
        P = pts - pts.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(P, full_matrices=False)
        axis = Vt[0]
        t = P @ axis
        return np.argsort(t)

    # --------------------------
    # ПАТТЕРНЫ И ФОРМЫ
    # --------------------------
    def _pattern_library(self) -> Dict[str, List[Tuple[str, List[Tuple[int, int]], bool]]]:
        lib: Dict[str, List[Tuple[str, List[Tuple[int,int]], bool]]] = {}
        # Kindergarten
        k_h7 = [(-1,-1),(0,-1),(1,-1),(0,0),(-1,1),(0,1),(1,1)]
        k_w5 = [(0,0),(1,1),(0,2),(1,3),(0,4)]
        line3 = [(0,0),(0,1),(0,2)]
        lib["kindergarten"] = [
            ("H7", k_h7, True),
            ("W5", k_w5, True),
            ("LINE3", line3, True),
        ]
        # Polyclinics
        rect_2x4 = [(r, c) for r in range(2) for c in range(4)]
        lib["polyclinics"] = [("RECT_2x4", rect_2x4, True)]
        # School
        s_h_5x4 = [(r,0) for r in range(5)] + [(r,3) for r in range(5)] + [(2,c) for c in range(4)]
        ring = []
        for r in range(5):
            for c in range(5):
                if (r in {0,4} or c in {0,4}) and not (r in {0,4} and c in {0,4}):
                    ring.append((r,c))
        s_5x2_open = [(1,c) for c in range(5)] + [(0,0),(0,4)]
        lib["school"] = [
            ("H_5x4", s_h_5x4, True),
            ("RING_5x5_WITH_COURTYARD", ring, False),
            ("RECT_5x2_WITH_OPEN_3", s_5x2_open, True),
        ]
        return lib

    def _transform_offsets(self, offsets: List[Tuple[int, int]], rot_k: int, mirror: bool) -> List[Tuple[int, int]]:
        out = []
        for (dr, dc) in offsets:
            r, c = dr, dc
            for _ in range(rot_k % 4):
                r, c = c, -r
            if mirror:
                c = -c
            out.append((r, c))
        minr = min(r for r, _ in out); minc = min(c for _, c in out)
        return [(r - minr, c - minc) for (r, c) in out]

    def _shape_variants(self, offsets: List[Tuple[int, int]], allow_rotations: bool) -> List[List[Tuple[int, int]]]:
        variants = set()
        rots = [0, 1, 2, 3] if allow_rotations else [0]
        mirrors = [False, True] if allow_rotations else [False]
        for k in rots:
            for m in mirrors:
                var = tuple(sorted(self._transform_offsets(offsets, k, m)))
                variants.add(var)
        return [list(v) for v in variants]

    def _shape_length(self, var: List[Tuple[int, int]]) -> int:
        rs = [dr for dr, _ in var]; cs = [dc for _, dc in var]
        return max(max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)

    # --------------------------
    # ТЕРРИТОРИИ (прямоугольники)
    # --------------------------
    def _site_cells_required(self, area_m2: float) -> int:
        return int(math.ceil(max(area_m2, 1.0) / (self.cell_size_m * self.cell_size_m)))

    def _rect_variants_for_cells(self, ncells: int, max_variants: int = 12, ar_min: float = 0.33, ar_max: float = 3.0):
        base = int(round(math.sqrt(ncells)))
        pairs = []
        span = max(1, base) + 12
        for r in range(1, span + 1):
            c = int(math.ceil(ncells / r))
            ar = max(r, c) / max(1.0, min(r, c))
            if ar_min <= ar <= ar_max:
                pairs.append((r, c, r * c - ncells))
        pairs = sorted(pairs, key=lambda t: (t[2], abs(t[0] - t[1])))
        pairs = pairs[:max_variants]
        variants = []
        for r, c, _ in pairs:
            offs = [(dr, dc) for dr in range(r) for dc in range(c)]
            variants.append((f"RECT_{r}x{c}", offs))
        return variants

    def _territory_shape_variants(self, area_m2: float) -> List[Tuple[str, List[Tuple[int, int]]]]:
        ncells = self._site_cells_required(area_m2)
        return self._rect_variants_for_cells(ncells, max_variants=12)

    # --------------------------
    # ЛОКАЛЬНЫЕ УТИЛИТЫ
    # --------------------------
    def _service_site_spec(self, svc: str, pattern_name: str) -> Tuple[float, int]:
        spec = self.service_site_rules.get((svc, pattern_name))
        if spec:
            return float(spec["site_area_m2"]), int(spec["capacity"])  # area_m2, capacity
        # дефолты
        if svc == "school":
            return 33000.0, 600
        if svc == "kindergarten":
            return 4400.0, 100
        if svc == "polyclinics":
            return 3000.0, 300
        return 2000.0, 0

    def _min_cheb_between_sets(self, cells: gpd.GeoDataFrame, A: List[int], B: List[int]) -> int:
        if not A or not B:
            return 10 ** 9
        ar = cells.loc[A, "row_i"].to_numpy(); ac = cells.loc[A, "col_j"].to_numpy()
        br = cells.loc[B, "row_i"].to_numpy(); bc = cells.loc[B, "col_j"].to_numpy()
        best = 10 ** 9
        for i in range(len(ar)):
            dr = np.abs(br - ar[i]); dc = np.abs(bc - ac[i])
            d = int(np.min(np.maximum(dr, dc)))
            if d < best:
                best = d
            if best == 0:
                break
        return best

    def _positions_from_center_or_edges(self, svc: str, rmin: int, rmax: int, cmin: int, cmax: int, r_center: float, c_center: float, invert: bool = False) -> List[Tuple[int, int]]:
        pos = [(r0, c0) for r0 in range(rmin, rmax + 1) for c0 in range(cmin, cmax + 1)]
        def d2(rc):
            return (rc[0] - r_center) ** 2 + (rc[1] - c_center) ** 2
        pos.sort(key=d2, reverse=invert and (svc == "kindergarten"))
        return pos

    def _min_site_cells_for_service_with_margin(self, svc: str, shape_variants: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]], inner_margin_cells: int = 1) -> int:
        need = 0
        if svc not in shape_variants:
            return 0
        best = None
        for _pat, vars_ in shape_variants[svc]:
            for var in vars_:
                vr = [dr for dr, _ in var]; vc = [dc for _, dc in var]
                h = (max(vr) - min(vr) + 1) + 2 * inner_margin_cells
                w = (max(vc) - min(vc) + 1) + 2 * inner_margin_cells
                cells_needed = h * w
                best = cells_needed if best is None else min(best, cells_needed)
        return 0 if best is None else best

    def _sort_variants_by_core_fit(self, variants: List[List[Tuple[int, int]]], core_h: int, core_w: int) -> List[List[Tuple[int, int]]]:
        if core_h <= 0 or core_w <= 0:
            return []
        core_ar = core_w / core_h if core_h > 0 else 1.0
        def dims(var):
            vr = [dr for (dr, dc) in var]; vc = [dc for (dr, dc) in var]
            h = max(vr) - min(vr) + 1
            w = max(vc) - min(vc) + 1
            return h, w
        scored = []
        for var in variants:
            h, w = dims(var)
            var_ar = w / h if h > 0 else 1.0
            same_orient = int(not ((core_w >= core_h) ^ (w >= h)))
            ar_diff = abs(math.log(max(var_ar, 1e-6) / max(core_ar, 1e-6)))
            scored.append((0 if same_orient else 1, ar_diff, h * w, var))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        return [t[3] for t in scored]

    # --------------------------
    # РАЗМЕЩЕНИЕ КОРПУСА ВНУТРИ УЧАСТКА
    # --------------------------
    def _try_place_service_inside_site(
        self,
        cells: gpd.GeoDataFrame,
        svc: str, zid: int, site_idxs: List[int], site_id: str,
        shape_variants: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]],
        reserved_service_cells: set[int], rng: random.Random,
        idx_by_rc: Dict[Tuple[int, int], int],
        inner_margin_cells: int,
    ) -> Tuple[bool, List[int], str]:
        site_set = set(site_idxs)
        rvals = cells.loc[site_idxs, "row_i"].to_numpy(); cvals = cells.loc[site_idxs, "col_j"].to_numpy()
        rmin, rmax = int(rvals.min()), int(rvals.max())
        cmin, cmax = int(cvals.min()), int(cvals.max())
        rmin_core = rmin + inner_margin_cells; rmax_core = rmax - inner_margin_cells
        cmin_core = cmin + inner_margin_cells; cmax_core = cmax - inner_margin_cells
        if (rmin_core > rmax_core) or (cmin_core > cmax_core):
            return False, [], ""
        core_h = rmax_core - rmin_core + 1; core_w = cmax_core - cmin_core + 1
        cen_r = 0.5 * (rmin_core + rmax_core); cen_c = 0.5 * (cmin_core + cmax_core)

        for (pat_name, variants) in shape_variants.get(svc, []):
            vars_iter = list(variants)
            if self.randomize_service_forms:
                rng.shuffle(vars_iter)
            vars_iter = self._sort_variants_by_core_fit(vars_iter, core_h, core_w)
            for var in vars_iter:
                vr = [dr for (dr, dc) in var]; vc = [dc for (dr, dc) in var]
                h, w = (max(vr) - min(vr) + 1), (max(vc) - min(vc) + 1)
                if h > core_h or w > core_w:
                    continue
                positions = [(r0, c0) for r0 in range(rmin_core, rmax_core - h + 2) for c0 in range(cmin_core, cmax_core - w + 2)]
                positions.sort(key=lambda rc: (rc[0] - cen_r) ** 2 + (rc[1] - cen_c) ** 2)
                for (r0, c0) in positions:
                    idxs = []
                    ok = True
                    for (dr, dc) in var:
                        rr, cc = r0 + dr, c0 + dc
                        idx = idx_by_rc.get((rr, cc))
                        if (idx is None) or (idx in reserved_service_cells) or (idx not in site_set):
                            ok = False; break
                        idxs.append(idx)
                    if not ok:
                        continue
                    return True, idxs, pat_name
        return False, [], ""

    # --------------------------
    # УСЛОВИЯ ЗАЗОРОВ
    # --------------------------
    def _house_indices(self, cells: gpd.GeoDataFrame) -> np.ndarray:
        return np.where(cells["is_building"].fillna(False).to_numpy())[0]

    def _cheb_gap_ok_to_houses(self, cells: gpd.GeoDataFrame, candidate_idxs: List[int]) -> bool:
        H = self._house_indices(cells)
        if len(H) == 0 or len(candidate_idxs) == 0:
            return True
        d = self._min_cheb_between_sets(cells, candidate_idxs, list(H))
        return d >= self.gap_to_houses_cheb

    def _cheb_gap_ok_to_sites(self, cells: gpd.GeoDataFrame, candidate_idxs: List[int], placed_site_sets: List[List[int]], placed_sites_by_type: Dict[str, List[List[int]]], svc: str) -> bool:
        # Общий зазор со всеми участками
        for S in placed_site_sets:
            if self._min_cheb_between_sets(cells, candidate_idxs, S) < self.gap_between_sites_cheb:
                return False
        # Усиленный зазор для участков того же типа
        for S in placed_sites_by_type.get(svc, []):
            if self._min_cheb_between_sets(cells, candidate_idxs, S) < self.same_type_site_gap_cheb:
                return False
        return True

    # --------------------------
    # ПОПЫТКИ РАЗМЕЩЕНИЯ ВНУТРИ/СНАРУЖИ
    # --------------------------
    def _try_place_site_and_service_in_zone_level(
        self,
        cells: gpd.GeoDataFrame,
        zid: int, svc: str, allowed_ids: List[int], r_cen: float, c_cen: float,
        placed_site_sets: List[List[int]], placed_sites_by_type: Dict[str, List[List[int]]], rng: random.Random,
        shape_variants: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]], idx_by_rc: Dict[Tuple[int, int], int],
        reserved_site_cells: set[int], reserved_service_cells: set[int], neighbors_all: Dict[int, List[int]],
        service_sites_geom: List, service_sites_attrs: List[Dict], service_polys_geom: List, service_polys_attrs: List[Dict],
    ) -> bool:
        if not allowed_ids:
            return False
        allowed_set = set(allowed_ids)
        coord_to_idx = {(int(cells.at[i, "row_i"]), int(cells.at[i, "col_j"])): i for i in allowed_ids}
        sub = cells.loc[allowed_ids, ["row_i", "col_j"]]
        rmin, rmax = int(sub["row_i"].min()), int(sub["row_i"].max())
        cmin, cmax = int(sub["col_j"].min()), int(sub["col_j"].max())
        positions = self._positions_from_center_or_edges(svc, rmin, rmax, cmin, cmax, r_cen, c_cen, invert=False)

        service_variants = list(shape_variants.get(svc, []))
        if self.randomize_service_forms:
            rng.shuffle(service_variants)

        for (pat_name, _service_vars) in service_variants:
            site_area_m2, capacity = self._service_site_spec(svc, pat_name)
            territory_variants = self._territory_shape_variants(site_area_m2)
            if self.randomize_service_forms:
                rng.shuffle(territory_variants)

            for (site_form_name, site_offsets) in territory_variants:
                vrr = [dr for (dr, dc) in site_offsets]; vcc = [dc for (dr, dc) in site_offsets]
                Hs, Ws = (max(vrr) - min(vrr) + 1), (max(vcc) - min(vcc) + 1)
                for (r0, c0) in positions:
                    if r0 + Hs - 1 > rmax or c0 + Ws - 1 > cmax:
                        continue
                    site_idxs = []
                    ok = True
                    for (dr, dc) in site_offsets:
                        rr, cc = r0 + dr, c0 + dc
                        idx = coord_to_idx.get((rr, cc))
                        if (idx is None) or (idx in reserved_site_cells) or (idx in reserved_service_cells):
                            ok = False; break
                        if idx not in allowed_set:
                            ok = False; break
                        if cells.at[idx, "is_building"]:
                            ok = False; break
                        site_idxs.append(idx)
                    if not ok:
                        continue
                    if not self._cheb_gap_ok_to_houses(cells, site_idxs):
                        continue
                    if not self._cheb_gap_ok_to_sites(cells, site_idxs, placed_site_sets, placed_sites_by_type, svc):
                        continue
                    ok_svc, svc_cell_idxs, chosen_pat = self._try_place_service_inside_site(
                        cells, svc, zid, site_idxs, site_id="__tmp__",
                        shape_variants=shape_variants,
                        reserved_service_cells=reserved_service_cells,
                        rng=rng,
                        idx_by_rc=idx_by_rc,
                        inner_margin_cells=self.inner_margin_cells,
                    )
                    if not ok_svc:
                        continue

                    # Фиксируем участок и корпус
                    site_id = f"SITE_{svc.upper()}_{str(len(service_sites_attrs) + 1).zfill(4)}"
                    service_id = f"{svc.upper()}_{str(len(service_polys_attrs) + 1).zfill(4)}"
                    for idx in site_idxs:
                        reserved_site_cells.add(idx)
                        cells.loc[idx, "is_service_site"] = True
                        cells.loc[idx, "site_id"] = site_id
                        cells.loc[idx, "service_site_type"] = svc
                    for idx in svc_cell_idxs:
                        reserved_service_cells.add(idx)
                        cells.loc[idx, "service"] = svc

                    site_poly = _make_valid(unary_union([cells.geometry[i] for i in site_idxs]))
                    svc_poly = _make_valid(unary_union([cells.geometry[i] for i in svc_cell_idxs]))

                    service_sites_geom.append(site_poly)
                    service_sites_attrs.append({
                        "site_id": site_id,
                        "service": svc,
                        "zone_id": int(zid),
                        "site_form": site_form_name,
                        "pattern_for_norms": pat_name,
                        "site_cells": int(len(site_idxs)),
                        "site_area_target_m2": float(site_area_m2),
                        "site_area_actual_m2": float(getattr(site_poly, "area", 0.0)),
                        "capacity": int(capacity),
                    })
                    service_polys_geom.append(svc_poly)
                    service_polys_attrs.append({
                        "building_id": service_id,
                        "site_id": site_id,
                        "service": svc,
                        "pattern": chosen_pat,
                        "zone_id": int(zid),
                        "n_cells": int(len(svc_cell_idxs)),
                        "width_m": float(self.cell_size_m),
                    })

                    placed_site_sets.append(site_idxs)
                    placed_sites_by_type.setdefault(svc, []).append(site_idxs)
                    return True
        return False

    def _try_place_site_and_service_fallback_outside(
        self,
        cells: gpd.GeoDataFrame,
        zid: int, svc: str, outside_ids: List[int], r_cen: float, c_cen: float,
        placed_site_sets: List[List[int]], placed_sites_by_type: Dict[str, List[List[int]]], rng: random.Random,
        shape_variants: Dict[str, List[Tuple[str, List[List[Tuple[int, int]]]]]], idx_by_rc: Dict[Tuple[int, int], int],
        reserved_site_cells: set[int], reserved_service_cells: set[int], neighbors_all: Dict[int, List[int]],
        service_sites_geom: List, service_sites_attrs: List[Dict], service_polys_geom: List, service_polys_attrs: List[Dict],
    ) -> bool:
        if not outside_ids:
            return False
        allowed_ids = [i for i in outside_ids if (i not in reserved_site_cells) and (not cells.at[i, "is_building"])]
        if not allowed_ids:
            return False
        allowed_set = set(allowed_ids)
        coord_to_idx = {(int(cells.at[i, "row_i"]), int(cells.at[i, "col_j"])): i for i in allowed_ids}
        sub = cells.loc[allowed_ids, ["row_i", "col_j"]]
        rmin, rmax = int(sub["row_i"].min()), int(sub["row_i"].max())
        cmin, cmax = int(sub["col_j"].min()), int(sub["col_j"].max())
        positions = self._positions_from_center_or_edges(svc, rmin, rmax, cmin, cmax, r_cen, c_cen, invert=False)

        service_variants = list(shape_variants.get(svc, []))
        if self.randomize_service_forms:
            rng.shuffle(service_variants)

        for (pat_name, _service_vars) in service_variants:
            site_area_m2, capacity = self._service_site_spec(svc, pat_name)
            min_cells = self._min_site_cells_for_service_with_margin(svc, shape_variants, self.inner_margin_cells)
            ncells = max(self._site_cells_required(site_area_m2), min_cells)
            territory_variants = self._rect_variants_for_cells(ncells, max_variants=12)
            if self.randomize_service_forms:
                rng.shuffle(territory_variants)

            for (site_form_name, site_offsets) in territory_variants:
                vrr = [dr for (dr, dc) in site_offsets]; vcc = [dc for (dr, dc) in site_offsets]
                Hs, Ws = (max(vrr) - min(vrr) + 1), (max(vcc) - min(vcc) + 1)
                for (r0, c0) in positions:
                    if r0 + Hs - 1 > rmax or c0 + Ws - 1 > cmax:
                        continue
                    site_idxs = []
                    ok = True
                    for (dr, dc) in site_offsets:
                        rr, cc = r0 + dr, c0 + dc
                        idx = coord_to_idx.get((rr, cc))
                        if (idx is None) or (idx in reserved_site_cells) or (idx in reserved_service_cells):
                            ok = False; break
                        if idx not in allowed_set:
                            ok = False; break
                        if cells.at[idx, "is_building"]:
                            ok = False; break
                        site_idxs.append(idx)
                    if not ok:
                        continue
                    if not self._cheb_gap_ok_to_houses(cells, site_idxs):
                        continue
                    if not self._cheb_gap_ok_to_sites(cells, site_idxs, placed_site_sets, placed_sites_by_type, svc):
                        continue
                    ok_svc, svc_cell_idxs, chosen_pat = self._try_place_service_inside_site(
                        cells, svc, zid, site_idxs, site_id="__tmp__",
                        shape_variants=shape_variants,
                        reserved_service_cells=reserved_service_cells,
                        rng=rng,
                        idx_by_rc=idx_by_rc,
                        inner_margin_cells=self.inner_margin_cells,
                    )
                    if not ok_svc:
                        continue

                    site_id = f"SITE_{svc.upper()}_{str(len(service_sites_attrs) + 1).zfill(4)}"
                    service_id = f"{svc.upper()}_{str(len(service_polys_attrs) + 1).zfill(4)}"
                    for idx in site_idxs:
                        reserved_site_cells.add(idx)
                        cells.loc[idx, "is_service_site"] = True
                        cells.loc[idx, "site_id"] = site_id
                        cells.loc[idx, "service_site_type"] = svc
                    for idx in svc_cell_idxs:
                        reserved_service_cells.add(idx)
                        cells.loc[idx, "service"] = svc

                    site_poly = _make_valid(unary_union([cells.geometry[i] for i in site_idxs]))
                    svc_poly = _make_valid(unary_union([cells.geometry[i] for i in svc_cell_idxs]))

                    service_sites_geom.append(site_poly)
                    service_sites_attrs.append({
                        "site_id": site_id,
                        "service": svc,
                        "zone_id": int(zid),
                        "site_form": site_form_name,
                        "pattern_for_norms": pat_name,
                        "site_cells": int(len(site_idxs)),
                        "site_area_target_m2": float(site_area_m2),
                        "site_area_actual_m2": float(getattr(site_poly, "area", 0.0)),
                        "capacity": int(capacity),
                        "fallback_outside": True,
                    })
                    service_polys_geom.append(svc_poly)
                    service_polys_attrs.append({
                        "building_id": service_id,
                        "site_id": site_id,
                        "service": svc,
                        "pattern": chosen_pat,
                        "zone_id": int(zid),
                        "n_cells": int(len(svc_cell_idxs)),
                        "width_m": float(self.cell_size_m),
                        "fallback_outside": True,
                    })

                    placed_site_sets.append(site_idxs)
                    placed_sites_by_type.setdefault(svc, []).append(site_idxs)
                    return True
        return False
