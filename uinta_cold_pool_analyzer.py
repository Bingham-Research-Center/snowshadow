import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from herbie import Herbie
import metpy.calc as mpcalc
from metpy.interpolate import cross_section
from metpy.calc import cross_section_components
from metpy.units import units


class UintaColdPoolAnalyzer:
    """
    A class for analyzing and visualizing cold pool structures in the Uinta Basin
    using HRRR model data and MetPy's cross-section capabilities.
    """

    def __init__(self, date_time=None, model="hrrr", product="sfc", fxx=0):
        """
        Initialize the UintaColdPoolAnalyzer with specified parameters.
        """
        self.date_time = date_time
        self.model = model
        self.product = product
        self.fxx = fxx
        self.herbie = None
        self.dataset = None
        self.cross_section_data = None
        self.cross_section_path = None
        self.fig = None
        self.ax = None
        self.output_dir = "output"

        # Default coordinates for Uinta Basin (can be adjusted later)
        self.start_point = (40.0, -111.0)  # Western edge of basin (lat, lon)
        self.end_point = (40.0, -109.0)      # Eastern edge of basin (lat, lon)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self, date_time=None, variables=None, custom_coordinates=None):
        """
        Load HRRR data using Herbie for the specified time and variables.
        """
        if date_time:
            self.date_time = date_time

        if variables is None:
            variables = ["TMP:2 m", "UGRD:10 m", "VGRD:10 m", "HGT:surface", "PRES:surface"]

        if custom_coordinates:
            if isinstance(custom_coordinates, tuple) and len(custom_coordinates) == 2:
                self.start_point, self.end_point = custom_coordinates
            else:
                print("Warning: custom_coordinates must be a tuple of two points ((lat1, lon1), (lat2, lon2))")

        print(f"Loading HRRR data for {self.date_time}...")

        try:
            self.herbie = Herbie(self.date_time, model=self.model, product=self.product, fxx=self.fxx)
            datasets = []
            for var in variables:
                try:
                    print(f"Fetching {var}...")
                    ds = self.herbie.xarray(var)
                    datasets.append(ds)
                except Exception as e:
                    print(f"Error fetching {var}: {e}")

            if datasets:
                self.dataset = xr.merge(datasets, compat='override')
                print(f"Successfully loaded data with variables: {', '.join(variables)}")
                self._prepare_dataset()
                return self.dataset
            else:
                print("No data was loaded.")
                return None

        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prepare_dataset(self):
        """
        Process the dataset to ensure compatibility with MetPy's cross-section functions.
        """
        if self.dataset is None:
            print("No dataset loaded. Please call load_data() first.")
            return

        # If both 'time' and 'valid_time' are present, drop 'time' to avoid ambiguity
        if 'time' in self.dataset and 'valid_time' in self.dataset:
            self.dataset = self.dataset.drop_vars('time')

        self.dataset = self.dataset.metpy.parse_cf()
        self.dataset = self.dataset.metpy.assign_y_x()

        print("Dataset prepared for cross-section analysis")

    def preprocess_data(self, vertical_levels=None, interpolation_steps=200):
        """
        Preprocess data for cross-sectional analysis.
        """
        if self.dataset is None:
            print("No dataset loaded. Please call load_data() first.")
            return None

        print("Preprocessing data for cold pool analysis...")

        try:
            # Check if we have 3D data with pressure levels or surface data
            is_3d = len(self.dataset.dims) > 2 and 'isobaric' in self.dataset.dims

            if not is_3d:
                print("Working with 2D surface data. No vertical cross-section will be generated.")
                return self.dataset

            if vertical_levels and 'isobaric' in self.dataset.dims:
                self.dataset = self.dataset.sel(isobaric=vertical_levels, method='nearest')

            print(f"Generating cross-section from {self.start_point} to {self.end_point}...")
            self.cross_section_data = cross_section(
                self.dataset,
                self.start_point,
                self.end_point,
                steps=interpolation_steps,
                interp_type='linear'
            )

            self.cross_section_path = (self.start_point, self.end_point)

            if 't2m' in self.cross_section_data and 'isobaric' in self.cross_section_data:
                pressure = self.cross_section_data['isobaric']
                temperature = self.cross_section_data['t2m']
                self.cross_section_data['theta'] = mpcalc.potential_temperature(
                    pressure, temperature
                )
                print("Calculated potential temperature")

            if 'u10' in self.cross_section_data and 'v10' in self.cross_section_data:
                u_wind = self.cross_section_data['u10']
                v_wind = self.cross_section_data['v10']
                t_wind, n_wind = cross_section_components(u_wind, v_wind)
                self.cross_section_data['t_wind'] = t_wind
                self.cross_section_data['n_wind'] = n_wind
                print("Calculated cross-section wind components")

            print("Data preprocessing complete")
            return self.cross_section_data

        except Exception as e:
            print(f"Error preprocessing data: {e}")
            import traceback
            traceback.print_exc()
            return None

    def generate_cross_section(self, variables=None, levels=20, figsize=(15, 10),
                               cmaps=None, save=True, show=True):
        """
        Generate and visualize a cross-section plot of the cold pool structure.
        """
        if self.cross_section_data is None:
            print("No cross-section data. Please call preprocess_data() first.")
            return None, None

        if variables is None:
            variables = []
            if 'theta' in self.cross_section_data:
                variables.append('theta')
            if 't_wind' in self.cross_section_data and 'n_wind' in self.cross_section_data:
                variables.extend(['t_wind', 'n_wind'])
            if 't2m' in self.cross_section_data and not variables:
                variables.append('t2m')

        if not variables:
            print("No valid variables found for plotting.")
            return None, None

        if cmaps is None:
            cmaps = {
                'theta': 'viridis',
                't2m': 'RdBu_r',
                'n_wind': 'coolwarm',
                't_wind': 'PuOr'
            }

        self.fig, self.ax = plt.subplots(figsize=figsize)

        if 'isobaric' in self.cross_section_data.dims:
            p_min = self.cross_section_data.isobaric.min().values
            p_max = self.cross_section_data.isobaric.max().values
            self.ax.set_ylim(p_max, p_min)

        contours = {}
        for var in variables:
            if var not in self.cross_section_data:
                print(f"Variable {var} not found in dataset, skipping")
                continue

            if var == 'theta':
                contours[var] = self.ax.contour(
                    self.cross_section_data['lon'],
                    self.cross_section_data['isobaric'],
                    self.cross_section_data[var],
                    levels=levels,
                    colors='k',
                    linewidths=1.0,
                    alpha=0.7
                )
                self.ax.clabel(contours[var], inline=True, fontsize=8, fmt='%1.0f')

            elif var in ['t2m', 'theta']:
                contours[var] = self.ax.contourf(
                    self.cross_section_data['lon'],
                    self.cross_section_data['isobaric'],
                    self.cross_section_data[var],
                    levels=levels,
                    cmap=cmaps.get(var, 'RdBu_r')
                )
                plt.colorbar(contours[var], ax=self.ax, shrink=0.8,
                             label=f"{var} (K)" if var == 'theta' else f"{var} (°C)")
            elif var in ['t_wind', 'n_wind']:
                continue

        if 't_wind' in self.cross_section_data and 'n_wind' in self.cross_section_data:
            skip_x = max(1, len(self.cross_section_data['lon']) // 20)
            skip_y = max(1, len(self.cross_section_data['isobaric']) // 10)
            self.ax.barbs(
                self.cross_section_data['lon'][::skip_x],
                self.cross_section_data['isobaric'][::skip_y],
                self.cross_section_data['t_wind'][::skip_y, ::skip_x],
                self.cross_section_data['n_wind'][::skip_y, ::skip_x],
                length=5
            )

        if 'orog' in self.cross_section_data:
            terrain_y = self.cross_section_data['isobaric'].max().values
            self.ax.fill_between(
                self.cross_section_data['lon'],
                terrain_y,
                self.cross_section_data['orog'],
                facecolor='lightgray',
                zorder=10
            )

        start_lat, start_lon = self.start_point
        end_lat, end_lon = self.end_point

        self.ax.set_xlabel('Longitude (°)')
        self.ax.set_ylabel('Pressure (hPa)')

        valid_time = self.dataset.valid_time.values if hasattr(self.dataset, 'valid_time') else self.date_time
        title = (f"Uinta Basin Cold Pool Cross-Section ({start_lon:.2f}°, {start_lat:.2f}° to "
                 f"{end_lon:.2f}°, {end_lat:.2f}°)\n{valid_time}")
        self.ax.set_title(title)

        self.ax.grid(linestyle=':', alpha=0.5)

        if save:
            self._save_figure()

        if show:
            plt.tight_layout()
            plt.show()

        return self.fig, self.ax

    def visualize(self, map_extent=None, figsize=(15, 12), save=True, show=True):
        """
        Create a map view of the Uinta Basin with the cross-section line.
        """
        if self.dataset is None:
            print("No dataset loaded. Please call load_data() first.")
            return None, None

        if map_extent is None:
            map_extent = [-111.0, -109.0, 39.5, 41.0]

        proj = ccrs.PlateCarree()
        self.fig, self.ax = plt.subplots(figsize=figsize, subplot_kw={'projection': proj})
        self.ax.set_extent(map_extent, crs=proj)

        self.ax.coastlines('50m', linewidth=0.8)
        self.ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.8, edgecolor='black')
        self.ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=0.8)

        try:
            counties = cfeature.NaturalEarthFeature(
                'cultural', 'admin_2_counties', '10m',
                facecolor='none', edgecolor='gray', linewidth=0.5)
            self.ax.add_feature(counties)
        except Exception as e:
            print(f"County boundaries feature not available: {e}")

        self.ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.7, edgecolor='blue', alpha=0.7)
        self.ax.add_feature(cfeature.LAKES.with_scale('10m'), edgecolor='blue', facecolor='lightblue', alpha=0.5)

        try:
            import cartopy.io.img_tiles as cimgt
            terrain_tiles = cimgt.Stamen('terrain-background')
            self.ax.add_image(terrain_tiles, 8, alpha=0.3)
        except Exception as e:
            print(f"Could not add topography: {e}")

        if 't2m' in self.dataset and 'x' in self.dataset and 'y' in self.dataset:
            try:
                temp = self.dataset['t2m'].metpy.convert_units('degC')
                if hasattr(self.dataset.metpy, 'cartopy_crs'):
                    transform = self.dataset.metpy.cartopy_crs
                else:
                    transform = proj

                contour = self.ax.contourf(
                    self.dataset['x'],
                    self.dataset['y'],
                    temp.squeeze(),
                    levels=20,
                    cmap='RdBu_r',
                    transform=transform
                )
                plt.colorbar(contour, ax=self.ax, shrink=0.8, label='2m Temperature (°C)')
            except Exception as e:
                print(f"Error plotting temperature: {e}")
                import traceback
                traceback.print_exc()

        cities = {
            'Vernal': (-109.528, 40.455),
            'Duchesne': (-110.399, 40.163),
            'Roosevelt': (-109.989, 40.299),
            'Price': (-110.811, 39.600)
        }

        for city, (lon, lat) in cities.items():
            west, east, south, north = map_extent
            if west <= lon <= east and south <= lat <= north:
                self.ax.plot(lon, lat, 'ko', transform=proj, markersize=5)
                self.ax.text(
                    lon + 0.03, lat, city, transform=proj,
                    fontsize=10, horizontalalignment='left', fontweight='bold',
                    path_effects=[patheffects.withStroke(linewidth=2, foreground='white')]
                )

        if self.cross_section_path:
            start_lat, start_lon = self.start_point
            end_lat, end_lon = self.end_point

            self.ax.plot(
                [start_lon, end_lon],
                [start_lat, end_lat],
                'r-',
                linewidth=2,
                transform=proj,
                label='Cross-section'
            )
            self.ax.plot(start_lon, start_lat, 'ro', transform=proj, markersize=6)
            self.ax.plot(end_lon, end_lat, 'ro', transform=proj, markersize=6)

            self.ax.text(
                start_lon, start_lat, 'Start',
                transform=proj,
                fontsize=10,
                horizontalalignment='right',
                verticalalignment='bottom',
                path_effects=[patheffects.withStroke(linewidth=2, foreground='white')]
            )
            self.ax.text(
                end_lon, end_lat, 'End',
                transform=proj,
                fontsize=10,
                horizontalalignment='left',
                verticalalignment='bottom',
                path_effects=[patheffects.withStroke(linewidth=2, foreground='white')]
            )

        gl = self.ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')
        gl.top_labels = False
        gl.right_labels = False

        valid_time = self.dataset.valid_time.values if hasattr(self.dataset, 'valid_time') else self.date_time
        self.ax.set_title(f"Uinta Basin Overview: {valid_time}", fontsize=16)

        if self.cross_section_path:
            self.ax.legend(loc='lower right')

        if save:
            self._save_figure(name="uinta_basin_overview")

        if show:
            plt.tight_layout()
            plt.show()

        return self.fig, self.ax

    def _save_figure(self, name=None, dpi=300, formats=None):
        """
        Save the current figure to files.
        """
        if self.fig is None:
            print("No figure to save.")
            return

        if name is None:
            name = "uinta_basin_cross_section"

        if formats is None:
            formats = ['png']

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        for fmt in formats:
            filename = f"{self.output_dir}/{name}_{timestamp}.{fmt}"
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Figure saved to {filename}")

    def export_data(self, filename=None, format='csv'):
        """
        Export the cross-section data to a file.
        """
        if self.cross_section_data is None:
            print("No cross-section data to export. Please call preprocess_data() first.")
            return None

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if filename is None:
            if hasattr(self.dataset, 'valid_time'):
                date_str = pd.Timestamp(self.dataset.valid_time.values).strftime("%Y%m%d_%H%M")
            else:
                date_str = pd.Timestamp.now().strftime("%Y%m%d_%H%M")
            filename = f"{self.output_dir}/uinta_cross_section_{date_str}.{format}"

        try:
            if format.lower() == 'csv':
                df = self.cross_section_data.to_dataframe()
                df.to_csv(filename)
                print(f"Data exported to {filename}")
            elif format.lower() == 'nc':
                self.cross_section_data.to_netcdf(filename)
                print(f"Data exported to {filename}")
            elif format.lower() == 'json':
                df = self.cross_section_data.to_dataframe()
                df.to_json(filename)
                print(f"Data exported to {filename}")
            else:
                print(f"Unsupported format: {format}. Use 'csv', 'nc', or 'json'")
                return None

            return filename

        except Exception as e:
            print(f"Error exporting data: {e}")
            return None


def analyze_uinta_basin():
    """
    Example of how to use the UintaColdPoolAnalyzer class.
    """
    date_time = "2025-02-26 18:00"
    analyzer = UintaColdPoolAnalyzer(date_time)
    analyzer.load_data()

    # Define cross-section path across the Uinta Basin
    analyzer.start_point = (40.299, -109.989)  # Roosevelt, UT
    analyzer.end_point = (40.455, -109.528)      # Vernal, UT

    analyzer.preprocess_data(interpolation_steps=300)

    # Calculate cropped extent from the dataset (assuming 'x' and 'y' exist)
    extent_lon_min = analyzer.dataset['x'].min()
    extent_lon_max = analyzer.dataset['x'].max()
    extent_lat_min = analyzer.dataset['y'].min()
    extent_lat_max = analyzer.dataset['y'].max()

    # Convert to Python scalars if necessary
    extent_lon_min = extent_lon_min.item() if hasattr(extent_lon_min, 'item') else extent_lon_min
    extent_lon_max = extent_lon_max.item() if hasattr(extent_lon_max, 'item') else extent_lon_max
    extent_lat_min = extent_lat_min.item() if hasattr(extent_lat_min, 'item') else extent_lat_min
    extent_lat_max = extent_lat_max.item() if hasattr(extent_lat_max, 'item') else extent_lat_max

    print(f"Cropped extent: Lon [{extent_lon_min:.2f} to {extent_lon_max:.2f}], "
          f"Lat [{extent_lat_min:.2f} to {extent_lat_max:.2f}]")

    analyzer.generate_cross_section(variables=['theta', 't_wind', 'n_wind'], figsize=(14, 8), show=True)
    analyzer.visualize(map_extent=[-111.0, -109.0, 39.5, 41.0], figsize=(14, 10), show=True)
    analyzer.export_data(format='csv')


def main():
    """
    Main function to run the analysis.
    """
    analyze_uinta_basin()


if __name__ == "__main__":
    main()
