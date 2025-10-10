from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.prepared import prep
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours


@dataclass
class DensityIsolines:
    """
    Построение изолиний плотности по точкам внутри полигонов-кварталов
    без утечек за границы, с edge-correction и управляемыми весами.

    Главный метод:
        build(zones_gdf, points_gdf, zone_id_col=None, output_crs=None) -> GeoDataFrame

    Примечания:
    - Ожидается, что зоны и точки в одной СК. Если auto_reproject=True и указан target_epsg,
      они будут приведены автоматически.
    - Влияние 'e' можно отключить use_e_weights=False или не задавать weight_column.
    """
    grid_size_m: float = 15.0
    bandwidth_m: float = 10.0
    level_quantiles: List[float] = field(default_factory=lambda: [0.60, 0.75, 0.85, 0.92, 0.97])
    min_line_len_m: float = 30.0
    max_levels: int = 12

    # Управление весами
    use_e_weights: bool = True
    weight_column: Optional[str] = None  # например "e"; None => авто-поиск ["e","e_i","E","e_pred","existence","existence_score"]

    # CRS/репроекция
    target_epsg: Optional[int] = 32636
    auto_reproject: bool = True

    # Логи
    verbose: bool = True

    # -------------------------
    # Публичный метод
    # -------------------------
    def build(
        self,
        zones_gdf: gpd.GeoDataFrame,
        points_gdf: gpd.GeoDataFrame,
        zone_id_col: Optional[str] = None,
        output_crs: Optional[str | int] = None,
    ) -> gpd.GeoDataFrame:
        """
        Построить изолинии для всех кварталов.
        :param zones_gdf: полигоны кварталов
        :param points_gdf: точки-центроиды с опциональной колонкой весов
        :param zone_id_col: имя колонки ID зоны; если None — авто-поиск ("id","zone_id","ZONE_ID","zone","name")
        :param output_crs: итоговая СК результата; если None — как у zones_gdf
        """
        zgdf, pgdf = self._prepare_crs(zones_gdf, points_gdf)

        # Определяем колонку ID зоны
        zid_col = zone_id_col or next((c for c in ["id", "zone_id", "ZONE_ID", "zone", "name"] if c in zgdf.columns), None)

        # Точки и веса
        px = pgdf.geometry.x.to_numpy()
        py = pgdf.geometry.y.to_numpy()
        pw = self._extract_weights(pgdf, use_e=self.use_e_weights, weight_col=self.weight_column)

        out_records: List[dict] = []

        for idx, row in zgdf.iterrows():
            poly = row.geometry
            if poly is None or poly.is_empty:
                continue
            zid = row.get(zid_col, idx) if zid_col else idx

            density, xedges, yedges, inside_mask = self._density_grid_for_block(poly, px, py, pw)

            vals_in = density[inside_mask]
            finite_pos = vals_in[np.isfinite(vals_in) & (vals_in > 0)]

            if self.verbose:
                vmin = float(np.nanmin(vals_in)) if vals_in.size else float("nan")
                vmax = float(np.nanmax(vals_in)) if vals_in.size else float("nan")
                print(f"[zone {zid}] cells_in={int(inside_mask.sum())}, pos_cells={int(finite_pos.size)}, "
                      f"density[min,max]=({vmin:.4g}, {vmax:.4g})")

            levels = self._build_levels_from_quantiles(finite_pos)
            if not levels:
                # бэкап — центр диапазона
                vmn, vmx = (float(np.nanmin(vals_in)), float(np.nanmax(vals_in))) if vals_in.size else (np.nan, np.nan)
                if np.isfinite(vmn) and np.isfinite(vmx) and vmn < vmx:
                    levels = [vmn + 0.5 * (vmx - vmn)]
                else:
                    if self.verbose:
                        print("  [skip] нет валидного диапазона плотности для изолиний")
                    continue

            lines = self._contours_for_levels(density, xedges, yedges, levels)
            if (not lines) and finite_pos.size:
                a, b = np.quantile(finite_pos, 0.05), np.quantile(finite_pos, 0.95)
                if a < b:
                    tight_levels = list(np.linspace(a + 1e-9, b - 1e-9, min(5, self.max_levels)))
                    lines = self._contours_for_levels(density, xedges, yedges, tight_levels)

            for geom, lvl in lines:
                inter = geom.intersection(poly)
                if inter.is_empty:
                    continue
                if inter.geom_type == "LineString":
                    geoms = [inter]
                elif inter.geom_type == "MultiLineString":
                    geoms = list(inter.geoms)
                else:
                    continue
                for g in geoms:
                    if g.length >= self.min_line_len_m:
                        out_records.append({
                            "zone_id": zid,
                            "level": float(lvl),
                            "grid_m": float(self.grid_size_m),
                            "bandwidth_m": float(self.bandwidth_m),
                            "use_e": bool(self.use_e_weights),
                            "geometry": g
                        })

        if out_records:
            gdf_out = gpd.GeoDataFrame(out_records, geometry="geometry", crs=zgdf.crs)
        else:
            gdf_out = gpd.GeoDataFrame(
                {
                    "zone_id": pd.Series(dtype="int64"),
                    "level": pd.Series(dtype="float64"),
                    "grid_m": pd.Series(dtype="float64"),
                    "bandwidth_m": pd.Series(dtype="float64"),
                    "use_e": pd.Series(dtype="bool"),
                    "geometry": gpd.GeoSeries(dtype="geometry"),
                },
                geometry="geometry",
                crs=zgdf.crs,
            )

        if output_crs is not None:
            gdf_out = gdf_out.to_crs(output_crs)

        if self.verbose:
            print(f"Готово: изолиний={len(gdf_out)} | GRID={self.grid_size_m} м, BANDWIDTH={self.bandwidth_m} м, "
                  f"уровни={self.level_quantiles}, USE_E_WEIGHTS={self.use_e_weights}")

        return gdf_out

    # -------------------------
    # Вспомогательные методы
    # -------------------------
    def _prepare_crs(
        self,
        zones_gdf: gpd.GeoDataFrame,
        points_gdf: gpd.GeoDataFrame
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """
        Привести зоны и точки к единой целевой СК, если задан target_epsg и включён auto_reproject.
        Иначе — оставить как есть, но убедиться, что обе имеют CRS.
        """
        zgdf = zones_gdf
        pgdf = points_gdf

        if self.auto_reproject and self.target_epsg is not None:
            if zgdf.crs is None:
                zgdf = zgdf.set_crs(epsg=self.target_epsg, allow_override=True)
            elif zgdf.crs.to_epsg() != self.target_epsg:
                zgdf = zgdf.to_crs(epsg=self.target_epsg)

            if pgdf.crs is None:
                pgdf = pgdf.set_crs(epsg=self.target_epsg, allow_override=True)
            elif pgdf.crs.to_epsg() != self.target_epsg:
                pgdf = pgdf.to_crs(epsg=self.target_epsg)
        else:
            # если один без CRS — назначим CRS другого при возможности
            if zgdf.crs is None and pgdf.crs is not None:
                zgdf = zgdf.set_crs(pgdf.crs, allow_override=True)
            if pgdf.crs is None and zgdf.crs is not None:
                pgdf = pgdf.set_crs(zgdf.crs, allow_override=True)

        if self.verbose:
            print(f"CRS: zones={zgdf.crs}, points={pgdf.crs}")
            print(f"Counts: zones={len(zgdf)}, points={len(pgdf)}")

        return zgdf, pgdf

    def _extract_weights(
        self,
        gdf_pts: gpd.GeoDataFrame,
        use_e: bool = True,
        weight_col: Optional[str] = None
    ) -> np.ndarray:
        """
        Вернуть веса точек.
        - Если use_e=False -> все единицы.
        - Иначе берём колонку 'weight_col' или авто-поиск среди ["e","e_i","E","e_pred","existence","existence_score"].
        Нормируем в [1,2].
        """
        if not use_e:
            return np.ones(len(gdf_pts), dtype=float)

        col = weight_col
        if col is None:
            for c in ["e", "e_i", "E", "e_pred", "existence", "existence_score"]:
                if c in gdf_pts.columns:
                    col = c
                    break
        if col is None:
            return np.ones(len(gdf_pts), dtype=float)

        w = pd.to_numeric(gdf_pts[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if w.size:
            w_min, w_max = np.nanmin(w), np.nanmax(w)
            if w_max > w_min:
                w = (w - w_min) / (w_max - w_min)
            w = np.clip(w, 0.0, 1.0)
            w = 1.0 + w  # 1..2
        else:
            w = np.ones(len(gdf_pts), dtype=float)
        return w

    def _cell_centers_mask(self, poly, xedges, yedges) -> np.ndarray:
        xs = (xedges[:-1] + xedges[1:]) * 0.5
        ys = (yedges[:-1] + yedges[1:]) * 0.5
        XX, YY = np.meshgrid(xs, ys)  # H x W
        centers = np.column_stack([XX.ravel(), YY.ravel()])
        # covers — чтобы учитывать граничные клетки
        m = np.fromiter((poly.covers(Point(xy)) for xy in centers), dtype=bool).reshape(len(ys), len(xs))
        return m

    def _density_grid_for_block(
        self,
        poly,
        px: np.ndarray,
        py: np.ndarray,
        pw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Построение растра плотности с edge-correction внутри одного полигона.
        Возвращает: (density HxW, xedges, yedges, inside_mask)
        """
        minx, miny, maxx, maxy = poly.bounds
        W = max(3, int(math.ceil((maxx - minx) / self.grid_size_m)))
        H = max(3, int(math.ceil((maxy - miny) / self.grid_size_m)))

        xedges = np.linspace(minx, maxx, W + 1)
        yedges = np.linspace(miny, maxy, H + 1)

        inside_mask = self._cell_centers_mask(poly, xedges, yedges)

        pr = prep(poly)
        in_bbox = (px >= minx) & (px <= maxx) & (py >= miny) & (py <= maxy)
        if np.any(in_bbox):
            sel = np.fromiter((pr.covers(Point(x, y)) for x, y in zip(px[in_bbox], py[in_bbox])), dtype=bool)
            px_in = px[in_bbox][sel]
            py_in = py[in_bbox][sel]
            pw_in = pw[in_bbox][sel]
        else:
            px_in = py_in = pw_in = np.array([], dtype=float)

        hist, _, _ = np.histogram2d(
            px_in, py_in,
            bins=[W, H],
            range=[[minx, maxx], [miny, maxy]],
            weights=pw_in
        )
        counts = hist.T  # H x W

        sigma_cells = max(0.5, float(self.bandwidth_m / self.grid_size_m))
        smooth_counts = gaussian_filter(counts, sigma=sigma_cells, mode="nearest")
        smooth_mask = gaussian_filter(inside_mask.astype(float), sigma=sigma_cells, mode="nearest")
        smooth_mask = np.maximum(smooth_mask, 1e-9)
        smooth_corr = smooth_counts / smooth_mask

        cell_area = self.grid_size_m * self.grid_size_m
        density = smooth_corr / max(cell_area, 1.0)

        vals_in = density[inside_mask]
        if vals_in.size:
            min_in = float(vals_in.min())
            density = np.where(inside_mask, density, min_in - 1e-9)
        else:
            density = np.where(inside_mask, density, -np.inf)

        return density, xedges, yedges, inside_mask

    def _build_levels_from_quantiles(self, vals: np.ndarray) -> List[float]:
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return []
        vmin, vmax = float(vals.min()), float(vals.max())
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            return []

        lvl_list: List[float] = []
        for q in self.level_quantiles:
            q = float(np.clip(q, 0.0, 1.0))
            lvl = float(np.quantile(vals, q))
            if vmin < lvl < vmax and (not lvl_list or abs(lvl - lvl_list[-1]) > 1e-12):
                lvl_list.append(lvl)

        if len(lvl_list) == 0:
            # равномерная сетка по диапазону
            lvl_list = list(np.linspace(vmin + 1e-9, vmax - 1e-9, 5))

        if len(lvl_list) > self.max_levels:
            step = max(1, len(lvl_list) // self.max_levels)
            lvl_list = lvl_list[::step][:self.max_levels]

        return lvl_list

    def _contours_for_levels(
        self,
        density: np.ndarray,
        xedges: np.ndarray,
        yedges: np.ndarray,
        levels: List[float]
    ) -> List[Tuple[LineString, float]]:
        lines: List[Tuple[LineString, float]] = []
        dx = (xedges[1] - xedges[0])
        dy = (yedges[1] - yedges[0])

        def ij_to_xy(ii, jj):
            x = xedges[0] + jj * dx
            y = yedges[0] + ii * dy
            return x, y

        for lvl in levels:
            try:
                paths = find_contours(density, level=lvl)
            except Exception:
                continue
            for path in paths:
                if len(path) < 2:
                    continue
                coords = [ij_to_xy(i, j) for (i, j) in path]
                geom = LineString(coords)
                if geom.is_valid and geom.length >= self.min_line_len_m:
                    lines.append((geom, lvl))
        return lines
