from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

@dataclass
class GridGenerator:
    """
    Теггер клеток по изолиниям:
      1) Линии→полосы (buffer), оценка вложенности полос (iso_level).
      2) «Внутренние» полигоны колец из замкнутых линий + их вложенность (fill_level).
      3) Теггинг клеток: пересечение с полосами и попадание центроида во внутренние полигоны.
      4) Правило «3 стороны» (соседи по общему РЕБРУ).
      5) Назначение уровня промоутнутым клеткам (мода уровней соседей raw-True).

    Вход: grid_gdf (клетки, Polygon), isolines_gdf (LineString/MultiLineString/Polygon).
    Выход: тот же grid с полями:
        - cell_id: int
        - iso_pids: List[int] — id пересечённых полос
        - inside_iso_raw: bool — попали в полосу или “внутренний” полигон
        - inside_iso_closed: bool — после правила «3 стороны»
        - iso_level_raw: float — уровень от полос/внутренних полигонов
        - iso_level: float — финальный уровень (с учётом промоута)
        - fill_level: float — уровень “внутреннего” полигона, если был

    Параметры:
        iso_buffer_m: float — буфер линий изолиний для получения полос (м).
        edge_share_frac: float — порог длины общего ребра соседей (доля от характерного ребра клетки).
        auto_reproject: bool — привести изолинии к CRS сетки автоматически.
        fallback_epsg: int — назначить эту CRS сетке, если у неё нет CRS.
        verbose: bool — логировать шаги.
    """
    iso_buffer_m: float = 1.0
    edge_share_frac: float = 0.2
    auto_reproject: bool = True
    fallback_epsg: int = 32636
    verbose: bool = True

    # ---------------------------
    # ПУБЛИЧНЫЙ ИНТЕРФЕЙС
    # ---------------------------
    def fit_transform(
        self,
        grid_gdf: gpd.GeoDataFrame,
        isolines_gdf: gpd.GeoDataFrame,
        output_crs: Optional[str | int] = None
    ) -> gpd.GeoDataFrame:
        """Основной метод: возвращает размеченный grid."""
        grid = self._ensure_grid_crs(grid_gdf)
        iso = self._align_isolines_to_grid_crs(isolines_gdf, grid.crs)

        # 1) ЛИНИИ → ПОЛОСЫ (buffer) + iso_level (вложенность)
        iso_polys = self._isolines_to_polys(iso, buffer_m=self.iso_buffer_m)
        iso_polys["iso_pid"] = pd.to_numeric(iso_polys["iso_pid"], errors="coerce").astype("Int64")
        iso_polys = gpd.GeoDataFrame(iso_polys, geometry="geometry", crs=iso.crs)
        iso_polys = self._attach_nesting_level(iso_polys, id_col="iso_pid", out_level_col="iso_level")

        # 2) «Внутренние» полигоны колец из замкнутых линий + вложенность
        iso_fill = self._rings_to_fill_polys(iso)
        if len(iso_fill) > 0:
            iso_fill = self._attach_nesting_level(iso_fill, id_col="fill_id", out_level_col="fill_level")

        # 3) ТЕГГИНГ КЛЕТОК: пересечение с полосами (+ список iso_pids) и попадание центроидов во “внутренние” полигоны
        grid = grid.reset_index(drop=False).rename(columns={"index": "cell_id"})
        cells = grid[["cell_id", "geometry"]].copy()

        # 3.1 пересечения с полосами
        hit = self._sjoin(cells, iso_polys[["iso_pid", "iso_level", "geometry"]], predicate="intersects", how="left")
        hit_nonnull = hit[hit["iso_pid"].notna()].copy()
        agg_inter = (
            hit_nonnull.groupby("cell_id", dropna=False)
            .agg(
                iso_pids=("iso_pid", lambda s: sorted(set(int(x) for x in pd.to_numeric(s, errors="coerce").dropna().tolist()))),
                iso_level_raw=("iso_level", "max"),
            )
            .reset_index()
        )
        grid = grid.merge(agg_inter, on="cell_id", how="left")
        grid["iso_pids"] = grid["iso_pids"].apply(lambda v: v if isinstance(v, list) else [])
        grid["inside_iso_raw"] = grid["iso_pids"].apply(lambda v: len(v) > 0)
        grid["iso_level_raw"] = pd.to_numeric(grid["iso_level_raw"], errors="coerce").astype("Float64")

        # 3.2 попадание representative_point во «внутренние» полигоны
        if len(iso_fill) > 0:
            cells_pts = cells.copy()
            cells_pts["geometry"] = grid.geometry.representative_point().values
            hit_fill = self._sjoin(cells_pts, iso_fill[["fill_id", "fill_level", "geometry"]], predicate="within", how="left")
            hit_fill_nonnull = hit_fill[hit_fill["fill_id"].notna()].copy()
            agg_fill = (
                hit_fill_nonnull.groupby("cell_id", dropna=False)
                .agg(fill_level=("fill_level", "max"))
                .reset_index()
            )
            grid = grid.merge(agg_fill, on="cell_id", how="left")
            inside_by_fill = grid["fill_level"].notna()
            grid.loc[inside_by_fill, "inside_iso_raw"] = True
            grid["iso_level_raw"] = np.fmax(
                grid["iso_level_raw"].astype(float).fillna(-1),
                pd.to_numeric(grid["fill_level"], errors="coerce").fillna(-1),
            )
            grid["iso_level_raw"].replace(-1, np.nan, inplace=True)
        else:
            grid["fill_level"] = np.nan

        # 4) ПРАВИЛО «3 СТОРОНЫ»: соседи по общему ребру с достаточной длиной пересечения границ
        neighbors = self._edge_neighbors(grid)
        inside_map = dict(zip(grid["cell_id"].values, grid["inside_iso_raw"].values))
        promote = []
        for rid, neighs in neighbors.items():
            if not inside_map.get(rid, False):
                if sum(inside_map.get(nb, False) for nb in neighs) >= 3:
                    promote.append(rid)

        grid["inside_iso_closed"] = grid.apply(
            lambda r: bool(r["inside_iso_raw"] or (r["cell_id"] in promote)),
            axis=1
        )

        # 5) УРОВЕНЬ ДЛЯ ПРОМОУТНУТЫХ КЛЕТОК — мода уровней соседей raw-True
        level_map = dict(zip(grid["cell_id"].values, grid["iso_level_raw"].values))

        def _neighbor_level_mode(rid: int) -> float:
            vals = [
                level_map.get(nb)
                for nb in neighbors.get(rid, [])
                if inside_map.get(nb, False) and pd.notna(level_map.get(nb))
            ]
            if not vals:
                return np.nan
            # мода
            return int(pd.Series(vals).value_counts().index[0])

        grid["iso_level"] = grid["iso_level_raw"]
        need_fill = grid["inside_iso_closed"] & (~grid["inside_iso_raw"])
        grid.loc[need_fill, "iso_level"] = grid.loc[need_fill, "cell_id"].apply(_neighbor_level_mode)

        # Итоговый набор колонок
        cols_out = [
            "cell_id", "geometry",
            "iso_pids",
            "inside_iso_raw", "inside_iso_closed",
            "iso_level_raw", "iso_level",
            "fill_level",
        ]
        grid_out = gpd.GeoDataFrame(grid[cols_out].copy(), geometry="geometry", crs=grid.crs)

        if output_crs is not None:
            grid_out = grid_out.to_crs(output_crs)

        if self.verbose:
            print(
                f"OK | cells={len(grid_out)}, raw True={int(grid_out.inside_iso_raw.sum())}, "
                f"closed True={int(grid_out.inside_iso_closed.sum())}"
            )
        return grid_out

    # ---------------------------
    # ВНУТРЕННИЕ ХЕЛПЕРЫ
    # ---------------------------
    def make_grid_for_blocks(
        self,
        blocks_gdf: gpd.GeoDataFrame,
        cell_size_m: float = 15.0,
        midlines: Optional[gpd.GeoDataFrame | gpd.GeoSeries | List] = None,
        block_id_col: Optional[str] = None,
        offset_m: float = 20.0,
        output_crs: Optional[str | int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Построить регулярную сетку по каждому кварталу:
            - Сначала накрываем bbox квартала квадратной сеткой (шаг = cell_size_m).
            - Затем обрезаем клетки по «внутренней» области, ограниченной midline.
            Практически: clip_area = block.buffer(-offset_m)  (совпадает с вашей логикой midline = boundary(buffer(-offset_m))).
            Если отрицательный буфер даёт пустоту (узкий квартал) — fallback: clip_area = block.buffer(-1e-6)
            Если и это пусто — clip_area = сам квартал.

        Параметры:
            blocks_gdf : GeoDataFrame с Polygon кварталов. Может иметь столбец идентификатора (block_id_col).
            cell_size_m: размер стороны клетки в метрах (по умолчанию 15).
            midlines   : (необязательно) GeoSeries/GeoDataFrame/список LineString/MultiLineString.
                        Нужны лишь для выравнивания CRS и будущих проверок; clip идёт по offset_m.
            block_id_col: имя столбца кварталов, который писать в результат как block_id; если None — берём индекс.
            offset_m   : величина «сдвига внутрь квартала» для определения области клиппинга (по умолчанию 20 м).
            output_crs : если задан, результат будет перепроецирован в эту CRS.

        Возвращает:
            GeoDataFrame с колонками:
            - block_id : идентификатор квартала
            - cell_id  : локальный ID клетки внутри квартала (0..N_i-1)
            - geometry : Polygon/MultiPolygon (ячейка, обрезанная по clip_area)
        """
        from shapely.geometry import box as _box

        if blocks_gdf is None or len(blocks_gdf) == 0:
            return gpd.GeoDataFrame({"block_id": [], "cell_id": []}, geometry=[], crs=self.fallback_epsg)

        # --- CRS и репроекция ---
        blocks = blocks_gdf.copy()
        blocks = self._ensure_grid_crs(blocks)
        if midlines is not None and self.auto_reproject:
            try:
                if isinstance(midlines, (gpd.GeoDataFrame, gpd.GeoSeries)) and midlines.crs != blocks.crs:
                    midlines = midlines.to_crs(blocks.crs)
            except Exception:
                pass  # midlines могут прийти как список «сырой» геометрии

        # Подготовим идентификатор квартала
        if block_id_col and block_id_col in blocks.columns:
            block_ids = list(blocks[block_id_col].values)
        else:
            block_ids = list(range(len(blocks)))

        out_rows = []
        out_geoms = []

        # Основной цикл по кварталам
        for i, (bid, geom) in enumerate(zip(block_ids, blocks.geometry.values)):
            if geom is None or geom.is_empty:
                continue

            # 1) bbox сетка
            minx, miny, maxx, maxy = geom.bounds
            if maxx - minx <= 0 or maxy - miny <= 0:
                continue

            # нормированные границы по шагу, чтобы сетка "ложилась" аккуратно
            start_x = np.floor(minx / cell_size_m) * cell_size_m
            start_y = np.floor(miny / cell_size_m) * cell_size_m
            end_x   = np.ceil(maxx / cell_size_m) * cell_size_m
            end_y   = np.ceil(maxy / cell_size_m) * cell_size_m

            # 2) область клиппинга по midline (через отрицательный буфер offset_m)
            clip_area = None
            try:
                if offset_m > 0:
                    clip_area = geom.buffer(-float(offset_m))
            except Exception:
                clip_area = None

            if clip_area is None or clip_area.is_empty:
                # узкий квартал → попробуем минимальный «внутрь», иначе оставим целый квартал
                try:
                    clip_area = geom.buffer(-1e-6)
                except Exception:
                    clip_area = None

            if clip_area is None or clip_area.is_empty:
                clip_area = geom

            # 3) итеративно создаём клетки и обрезаем их по clip_area
            cell_id = 0
            y = start_y
            while y < end_y - 1e-9:
                x = start_x
                while x < end_x - 1e-9:
                    cell = _box(x, y, x + cell_size_m, y + cell_size_m)
                    # Быстрый предикат: пересекается ли клетка с bbox clip_area?
                    if not cell.intersects(clip_area):
                        x += cell_size_m
                        continue
                    try:
                        clipped = cell.intersection(clip_area)
                        # починим валидность по необходимости
                        if not clipped.is_empty:
                            try:
                                clipped = clipped.buffer(0)
                            except Exception:
                                pass
                    except Exception:
                        clipped = cell.intersection(geom)  # запасной вариант

                    if clipped is not None and (not clipped.is_empty) and clipped.area > 1e-6:
                        out_rows.append({"block_id": bid, "cell_id": cell_id})
                        out_geoms.append(clipped)
                        cell_id += 1

                    x += cell_size_m
                y += cell_size_m

        grid = gpd.GeoDataFrame(out_rows, geometry=out_geoms, crs=blocks.crs)

        if output_crs is not None:
            grid = grid.to_crs(output_crs)

        if self.verbose:
            n_blocks = len(set(grid["block_id"])) if len(grid) else 0
            print(f"GRID | blocks={n_blocks}, cells={len(grid)}, cell_size={cell_size_m} m, offset={offset_m} m")

        return grid

    def _sjoin(self, left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, predicate: str, how: str = "left", **kwargs) -> gpd.GeoDataFrame:
        """Совместимость gpd.sjoin для разных версий (predicate/op)."""
        try:
            return gpd.sjoin(left, right, predicate=predicate, how=how, **kwargs)
        except TypeError:
            return gpd.sjoin(left, right, op=predicate, how=how, **kwargs)

    def _ensure_grid_crs(self, grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if grid.crs is None:
            warnings.warn(f"CRS сетки отсутствует, назначаю EPSG:{self.fallback_epsg}")
            return grid.set_crs(epsg=self.fallback_epsg, allow_override=True)
        return grid

    def _align_isolines_to_grid_crs(self, iso: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
        if not self.auto_reproject:
            return iso
        if iso.crs != target_crs:
            if self.verbose:
                print(f"Reproject isolines: {iso.crs} → {target_crs}")
            return iso.to_crs(target_crs)
        return iso

    def _isolines_to_polys(self, iso_gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
        """Линии → полосы через buffer; полигоны оставляем как есть. Возвращает GeoDataFrame с iso_pid, geometry."""
        iso_gdf = iso_gdf.copy()
        line_mask = iso_gdf.geom_type.isin(["LineString", "MultiLineString"])
        poly_mask = iso_gdf.geom_type.isin(["Polygon", "MultiPolygon"])

        parts = []
        if line_mask.any():
            lines = iso_gdf.loc[line_mask].copy()
            lines["geometry"] = lines.geometry.buffer(buffer_m)
            parts.append(lines)
        if poly_mask.any():
            parts.append(iso_gdf.loc[poly_mask].copy())

        out = pd.concat(parts, ignore_index=True) if parts else iso_gdf.copy()
        if not parts:  # если не нашли ни линий, ни полигонов (экзотика) — всё же буфернем
            out["geometry"] = out.geometry.buffer(buffer_m)

        out = gpd.GeoDataFrame(out, geometry="geometry", crs=iso_gdf.crs).reset_index(drop=True)
        out["iso_pid"] = np.arange(len(out))
        return out[["iso_pid", "geometry"]]

    def _attach_nesting_level(self, polys: gpd.GeoDataFrame, id_col: str, out_level_col: str) -> gpd.GeoDataFrame:
        """Вычислить уровень вложенности по принципу: кол-во полигонов, внутри которых лежит representative_point, минус 1."""
        pts = polys[[id_col, "geometry"]].copy()
        pts["geometry"] = pts.geometry.representative_point()
        pairs = _sjoin(pts, polys[[id_col, "geometry"]], predicate="within", how="left", lsuffix="pt", rsuffix="poly")
        cnt = pairs.groupby(f"{id_col}_pt", dropna=False).size() - 1
        levels = (
            cnt.reset_index()
            .rename(columns={f"{id_col}_pt": id_col, 0: out_level_col})
            .astype({id_col: "Int64"})
        )
        out = polys.merge(levels, on=id_col, how="left")
        out[out_level_col] = out[out_level_col].fillna(0).astype(int)
        return out

    def _rings_to_fill_polys(self, iso_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Построить «внутренние» полигоны колец из замкнутых линий/полигонов:
          - Polygon/MultiPolygon → внешние контуры как есть.
          - LineString/MultiLineString → только замкнутые (is_ring=True) в Polygon.
        Возвращает GeoDataFrame с fill_id, geometry (может содержать вложенности/пересечения).
        """
        geoms: List[Polygon] = []
        for geom in iso_gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            gtype = geom.geom_type
            if gtype == "Polygon":
                geoms.append(Polygon(geom.exterior))
            elif gtype == "MultiPolygon":
                for p in geom.geoms:
                    geoms.append(Polygon(p.exterior))
            elif gtype == "LineString":
                if getattr(geom, "is_ring", False):
                    geoms.append(Polygon(geom.coords))
            elif gtype == "MultiLineString":
                for ls in geom.geoms:
                    if getattr(ls, "is_ring", False):
                        geoms.append(Polygon(ls.coords))
            # прочее игнорируем

        if not geoms:
            return gpd.GeoDataFrame({"fill_id": [], "geometry": []}, geometry="geometry", crs=iso_gdf.crs)

        # Чиним валидность по необходимости
        geoms = [g.buffer(0) if isinstance(g, (Polygon, MultiPolygon)) else g for g in geoms]
        fill = gpd.GeoDataFrame({"geometry": geoms}, geometry="geometry", crs=iso_gdf.crs)
        fill = fill[~fill.geometry.is_empty & fill.geometry.notna()].copy().reset_index(drop=True)
        fill["fill_id"] = np.arange(len(fill))
        return fill[["fill_id", "geometry"]]

    def _edge_neighbors(self, grid: gpd.GeoDataFrame) -> Dict[int, List[int]]:
        """
        Соседство по общему РЕБРУ с контролем минимальной длины общего ребра:
        длина пересечения границ >= edge_share_frac * характерная_длина_ребра,
        где характерная_длина_ребра ~= sqrt(area клетки).
        """
        pairs = _sjoin(
            grid[["cell_id", "geometry"]],
            grid[["cell_id", "geometry"]],
            predicate="touches",
            how="left",
            lsuffix="a",
            rsuffix="b"
        )
        pairs = pairs[(pairs["cell_id_a"] != pairs["cell_id_b"]) & pairs["cell_id_b"].notna()].copy()
        pairs["cell_id_b"] = pairs["cell_id_b"].astype(int)

        geom_list = list(grid.geometry.values)
        pos = {rid: i for i, rid in enumerate(grid["cell_id"].values)}
        edge_len_est = np.sqrt(np.maximum(grid.geometry.area.values, 1e-9))
        thr_len = self.edge_share_frac * edge_len_est

        def _is_edge_neighbor(a: int, b: int) -> bool:
            ia, ib = pos[a], pos[b]
            inter = geom_list[ia].boundary.intersection(geom_list[ib].boundary)
            length = getattr(inter, "length", 0.0)
            return length >= min(thr_len[ia], thr_len[ib])

        pairs["edge_ok"] = pairs.apply(lambda r: _is_edge_neighbor(int(r["cell_id_a"]), int(r["cell_id_b"])), axis=1)
        pairs = pairs[pairs["edge_ok"]]

        neighbors: Dict[int, List[int]] = {rid: [] for rid in grid["cell_id"].values}
        for ra, rb in pairs[["cell_id_a", "cell_id_b"]].itertuples(index=False):
            neighbors[int(ra)].append(int(rb))
        return neighbors
