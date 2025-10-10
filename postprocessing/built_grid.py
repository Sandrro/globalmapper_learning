from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon


def _sjoin(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, predicate: str, how: str = "left", **kwargs) -> gpd.GeoDataFrame:
    """Совместимость gpd.sjoin для разных версий (predicate/op)."""
    try:
        return gpd.sjoin(left, right, predicate=predicate, how=how, **kwargs)
    except TypeError:
        return gpd.sjoin(left, right, op=predicate, how=how, **kwargs)


@dataclass
class IsolineCellTagger:
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
        hit = _sjoin(cells, iso_polys[["iso_pid", "iso_level", "geometry"]], predicate="intersects", how="left")
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
            hit_fill = _sjoin(cells_pts, iso_fill[["fill_id", "fill_level", "geometry"]], predicate="within", how="left")
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
