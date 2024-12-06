import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import rasterio
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind
import pickle

class EnhancedUintaSnowAnalysis:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.setup_directories()
        self.station_names = {
            'MMTU1': 'Mosby Mountain',
            'TCKU1': 'Trout Creek',
            'KGCU1': "King's Cabin",
            'UBHSP': 'Horsepool',
            'UBMYT': 'Myton',
            'UBMTH': 'Mountain Home'
        }
        self.high_stations = ['MMTU1', 'TCKU1', 'KGCU1']
        self.basin_stations = ['UBHSP', 'UBMYT', 'UBMTH']
        self.colors = {
            'high': '#2196F3',
            'basin': '#FF9800',
            'terrain': plt.cm.terrain
        }
        self.winter_months = [11, 12, 1, 2, 3, 4]
        self.load_data()

    def setup_directories(self):
        self.figures_dir = self.base_dir / 'figures' / 'snow_shadow_analysis'
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        print("Loading data...")
        try:
            metadata_path = self.base_dir / 'combined' / 'metadata_combined.pkl'
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            stations_data = []
            for stid, data in metadata.items():
                station_info = {
                    'stid': stid,
                    'name': data.get('NAME', ''),
                    'latitude': data.get('latitude', None),
                    'longitude': data.get('longitude', None),
                    'elevation': data.get('ELEVATION', None),
                    'record_start': pd.to_datetime(data.get('RECORD_START')),
                    'record_end': pd.to_datetime(data.get('RECORD_END'))
                }
                stations_data.append(station_info)

            self.meta_df = pd.DataFrame(stations_data)

            weather_path = self.base_dir / 'combined' / 'weather_data_combined.parquet'
            if not weather_path.exists():
                raise FileNotFoundError(f"Weather data file not found at {weather_path}")

            print("Loading weather data...")
            self.weather_data = pd.read_parquet(weather_path)
            self.process_data()

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def process_data(self):
        print("Processing data...")
        self.weather_data['snow_depth'] = self.weather_data['snow_depth'] / 10.0
        self.weather_data['snow_water_equiv'] = self.weather_data['snow_water_equiv'] / 10.0
        self.weather_data.index = pd.to_datetime(self.weather_data.index)

        # Remove duplicate labels in the index
        self.weather_data = self.weather_data[~self.weather_data.index.duplicated(keep='first')]

        self.weather_data['month'] = self.weather_data.index.month
        self.weather_data['year'] = self.weather_data.index.year
        self.weather_data['water_year'] = np.where(
            self.weather_data['month'] >= 10,
            self.weather_data['year'] + 1,
            self.weather_data['year']
        )
        self.weather_data.loc[self.weather_data['snow_depth'] > 300, 'snow_depth'] = np.nan
        self.weather_data.loc[self.weather_data['snow_water_equiv'] > 100, 'snow_water_equiv'] = np.nan

    # Other methods remain unchanged

    def create_terrain_map(self):
        print("Creating terrain map...")
        try:
            tif_path = self.base_dir / 'tif' / 'uinta_merged.tif'
            if not tif_path.exists():
                raise FileNotFoundError(f"Terrain data file not found at {tif_path}")

            with rasterio.open(tif_path) as src:
                elevation = src.read(1)
                extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

            fig = plt.figure(figsize=(20, 15))
            gs = GridSpec(2, 2, height_ratios=[2, 1], width_ratios=[2, 1])
            ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
            terrain_plot = ax_map.imshow(elevation, extent=extent, cmap=self.colors['terrain'], transform=ccrs.PlateCarree())
            ax_map.add_feature(cfeature.STATES, linewidth=0.5)
            ax_map.add_feature(cfeature.RIVERS, alpha=0.5)
            ax_map.gridlines(draw_labels=True)

            high_stations = self.meta_df[self.meta_df['stid'].isin(self.high_stations)]
            basin_stations = self.meta_df[self.meta_df['stid'].isin(self.basin_stations)]
            ax_map.scatter(high_stations['longitude'], high_stations['latitude'], c=self.colors['high'], s=100, label='High Elevation Stations', edgecolor='white', linewidth=1, transform=ccrs.PlateCarree())
            ax_map.scatter(basin_stations['longitude'], basin_stations['latitude'], c=self.colors['basin'], s=100, label='Basin Stations', edgecolor='white', linewidth=1, transform=ccrs.PlateCarree())

            for _, station in pd.concat([high_stations, basin_stations]).iterrows():
                snow_stats = self.calculate_station_stats(station['stid'])
                label = (f"{self.station_names[station['stid']]}\n"
                         f"{station['elevation']:.0f} ft\n"
                         f"Avg Snow: {snow_stats['mean_snow']:.1f} cm")
                ax_map.annotate(label, (station['longitude'], station['latitude']), xytext=(5, 5), textcoords='offset points', fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.7), transform=ccrs.PlateCarree())

            ax_map.plot([-110.75, -110.75], [39.5, 41], 'r--', label='Wasatch Range', linewidth=2, transform=ccrs.PlateCarree())
            ax_map.plot([-111, -109], [40.5, 40.5], 'g--', label='Uinta Range', linewidth=2, transform=ccrs.PlateCarree())
            ax_map.set_title('Uinta Basin Study Area\nStation Locations and Terrain', pad=20)
            plt.colorbar(terrain_plot, ax=ax_map, label='Elevation (m)')
            ax_map.legend(loc='upper right')

            ax_snow = fig.add_subplot(gs[1, 0])
            self.plot_snow_depth_analysis(ax_snow)
            ax_profile = fig.add_subplot(gs[1, 1])
            self.plot_elevation_profile(ax_profile)

            plt.tight_layout()
            plt.savefig(self.figures_dir / 'terrain_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error creating terrain map: {str(e)}")
            raise

    def plot_snow_depth_analysis(self, ax):
        selected_stations = self.high_stations + self.basin_stations
        for stid in selected_stations:
            station_data = self.weather_data[self.weather_data['stid'] == stid]
            station_data = station_data[station_data['month'].isin(self.winter_months)]
            monthly_avg = station_data.resample('M')['snow_depth'].mean()
            color = self.colors['high'] if stid in self.high_stations else self.colors['basin']
            ax.plot(monthly_avg.index, monthly_avg.values, label=self.station_names[stid], color=color, alpha=0.7)
        ax.set_title('Winter Snow Depth Time Series')
        ax.set_xlabel('Date')
        ax.set_ylabel('Snow Depth (cm)')
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)

    def plot_elevation_profile(self, ax):
        winter_data = self.weather_data[self.weather_data['month'].isin(self.winter_months)]
        station_stats = []
        for stid in self.high_stations + self.basin_stations:
            station_data = winter_data[winter_data['stid'] == stid]
            station_info = self.meta_df[self.meta_df['stid'] == stid].iloc[0]
            stats = {
                'stid': stid,
                'elevation': station_info['elevation'],
                'mean_snow': station_data['snow_depth'].mean(),
                'station_type': 'high' if stid in self.high_stations else 'basin'
            }
            station_stats.append(stats)
        stats_df = pd.DataFrame(station_stats)
        for stype, color in [('high', self.colors['high']), ('basin', self.colors['basin'])]:
            data = stats_df[stats_df['station_type'] == stype]
            ax.scatter(data['elevation'], data['mean_snow'], c=color, s=100, label=f"{'High Elevation' if stype == 'high' else 'Basin'} Stations")
        z = np.polyfit(stats_df['elevation'], stats_df['mean_snow'], 1)
        p = np.poly1d(z)
        ax.plot(stats_df['elevation'], p(stats_df['elevation']), '--k', alpha=0.5)
        ax.set_title('Elevation vs Average Snow Depth')
        ax.set_xlabel('Elevation (ft)')
        ax.set_ylabel('Average Snow Depth (cm)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def calculate_station_stats(self, stid):
        station_data = self.weather_data[(self.weather_data['stid'] == stid) & (self.weather_data['month'].isin(self.winter_months))]
        stats = {
            'mean_snow': station_data['snow_depth'].mean(),
            'max_snow': station_data['snow_depth'].max(),
            'snow_days': (station_data['snow_depth'] > 0).sum(),
            'total_records': len(station_data),
            'data_completeness': station_data['snow_depth'].notna().mean() * 100
        }
        return stats

    def create_statistical_analysis(self):
        print("Performing statistical analysis...")
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_monthly_distributions(ax1)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_annual_trends(ax2)
        ax3 = fig.add_subplot(gs[1, :])
        self.plot_statistical_summary(ax3)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_monthly_distributions(self, ax):
        winter_data = self.weather_data[self.weather_data['month'].isin(self.winter_months)].copy()
        winter_data['station_type'] = np.where(winter_data['stid'].isin(self.high_stations), 'High Elevation', 'Basin')
        sns.boxplot(data=winter_data, x='month', y='snow_depth', hue='station_type', palette=[self.colors['high'], self.colors['basin']], ax=ax)
        ax.set_title('Monthly Snow Depth Distributions')
        ax.set_xlabel('Month')
        ax.set_ylabel('Snow Depth (cm)')

    def plot_annual_trends(self, ax):
        annual_data = (self.weather_data[self.weather_data['month'].isin(self.winter_months)]
                       .groupby(['water_year', 'stid'])['snow_depth']
                       .mean()
                       .reset_index())
        for stid in self.high_stations + self.basin_stations:
            station_data = annual_data[annual_data['stid'] == stid]
            color = self.colors['high'] if stid in self.high_stations else self.colors['basin']
            ax.plot(station_data['water_year'], station_data['snow_depth'], 'o-', color=color, label=self.station_names[stid])
        ax.set_title('Annual Winter Snow Depth Trends')
        ax.set_xlabel('Water Year')
        ax.set_ylabel('Average Snow Depth (cm)')
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)

    def plot_statistical_summary(self, ax):
        winter_data = self.weather_data[self.weather_data['month'].isin(self.winter_months)]
        high_snow = winter_data[winter_data['stid'].isin(self.high_stations)]['snow_depth']
        basin_snow = winter_data[winter_data['stid'].isin(self.basin_stations)]['snow_depth']
        t_stat, p_value = ttest_ind(high_snow.dropna(), basin_snow.dropna())
        pooled_std = np.sqrt((high_snow.std() ** 2 + basin_snow.std() ** 2) / 2)
        cohens_d = (high_snow.mean() - basin_snow.mean()) / pooled_std
        summary_text = (
            "Statistical Analysis of Snow Shadow Effect\n"
            "=========================================\n\n"
            f"High Elevation Stations (n={len(self.high_stations)}):\n"
            f"  Mean Snow Depth: {high_snow.mean():.1f} cm\n"
            f"  Standard Deviation: {high_snow.std():.1f} cm\n\n"
            f"Basin Stations (n={len(self.basin_stations)}):\n"
            f"  Mean Snow Depth: {basin_snow.mean():.1f} cm\n"
            f"  Standard Deviation: {basin_snow.std():.1f} cm\n\n"
            f"Statistical Tests:\n"
            f"  t-statistic: {t_stat:.2f}\n"
            f"  p-value: {p_value:.2e}\n"
            f"  Cohen's d: {cohens_d:.2f}\n\n"
            f"Snow Depth Difference: {high_snow.mean() - basin_snow.mean():.1f} cm"
        )
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, verticalalignment='top', fontfamily='monospace', fontsize=10)
        ax.axis('off')

    def generate_report(self):
        print("Generating analysis report...")
        try:
            report_path = self.figures_dir / 'snow_shadow_analysis_report.txt'
            with open(report_path, 'w') as f:
                f.write("Uinta Basin Snow Shadow Effect Analysis\n")
                f.write("======================================\n\n")
                f.write("Station Information:\n")
                f.write("-------------------\n")
                for stid in self.high_stations + self.basin_stations:
                    stats = self.calculate_station_stats(stid)
                    station_info = self.meta_df[self.meta_df['stid'] == stid].iloc[0]
                    f.write(f"\n{self.station_names[stid]}:\n")
                    f.write(f"  Elevation: {station_info['elevation']:.0f} ft\n")
                    f.write(f"  Mean Snow Depth: {stats['mean_snow']:.1f} cm\n")
                    f.write(f"  Maximum Snow Depth: {stats['max_snow']:.1f} cm\n")
                    f.write(f"  Snow Days: {stats['snow_days']}\n")
                    f.write(f"  Data Completeness: {stats['data_completeness']:.1f}%\n")
                winter_data = self.weather_data[self.weather_data['month'].isin(self.winter_months)]
                high_snow = winter_data[winter_data['stid'].isin(self.high_stations)]['snow_depth']
                basin_snow = winter_data[winter_data['stid'].isin(self.basin_stations)]['snow_depth']
                t_stat, p_value = ttest_ind(high_snow.dropna(), basin_snow.dropna())
                f.write("\nStatistical Analysis:\n")
                f.write("--------------------\n")
                f.write(f"t-statistic: {t_stat:.2f}\n")
                f.write(f"p-value: {p_value:.2e}\n")
                f.write(f"Mean difference: {high_snow.mean() - basin_snow.mean():.1f} cm\n")
                f.write("\nInterpretation:\n")
                f.write("--------------\n")
                if p_value < 0.05:
                    f.write("There is a statistically significant difference in snow depth between high elevation and basin stations.\n")
                else:
                    f.write("The difference in snow depth between high elevation and basin stations is not statistically significant.\n")
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise

    def run_complete_analysis(self):
        print("\nStarting Uinta Basin Snow Shadow Analysis...")
        try:
            self.create_terrain_map()
            print("Terrain analysis complete.")
            self.create_statistical_analysis()
            print("Statistical analysis complete.")
            self.generate_report()
            print("Report generation complete.")
            print(f"\nAll analysis outputs have been saved to: {self.figures_dir}")
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        analysis = EnhancedUintaSnowAnalysis()
        analysis.run_complete_analysis()
    except Exception as e:
        print(f"Fatal error: {str(e)}")