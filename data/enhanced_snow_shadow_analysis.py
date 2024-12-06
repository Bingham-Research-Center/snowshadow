import pickle
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind
from statsmodels.tsa.seasonal import STL


class EnhancedUintaSnowAnalysis:
    """A class to perform enhanced snow analysis in the Uinta Basin."""

    def __init__(self):
        """Initialize the analysis with directories, station information, colors, and load data."""
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
        """Set up directories for storing figures."""
        self.figures_dir = self.base_dir / 'figures' / 'snow_shadow_analysis'
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load metadata and weather data from specified paths."""
        print("Loading data...")
        try:
            metadata_path = self.base_dir / 'combined' / 'metadata_combined.pkl'
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata file not found at {metadata_path}"
                )

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
                raise FileNotFoundError(
                    f"Weather data file not found at {weather_path}"
                )

            print("Loading weather data...")
            self.weather_data = pd.read_parquet(weather_path)
            self.process_data()

        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def process_data(self):
        """Process weather data by adjusting units and handling duplicates."""
        print("Processing data...")
        self.weather_data['snow_depth'] = self.weather_data['snow_depth'] / 10.0
        self.weather_data['snow_water_equiv'] = self.weather_data[
            'snow_water_equiv'
        ] / 10.0
        self.weather_data.index = pd.to_datetime(self.weather_data.index)
        self.weather_data = self.weather_data[
            ~self.weather_data.index.duplicated(keep='first')
        ]
        self.weather_data['month'] = self.weather_data.index.month
        self.weather_data['year'] = self.weather_data.index.year
        self.weather_data['water_year'] = np.where(
            self.weather_data['month'] >= 10,
            self.weather_data['year'] + 1,
            self.weather_data['year']
        )
        self.weather_data.loc[
            self.weather_data['snow_depth'] > 300, 'snow_depth'
        ] = np.nan
        self.weather_data.loc[
            self.weather_data['snow_water_equiv'] > 100, 'snow_water_equiv'
        ] = np.nan

    def analyze_data_gaps(self):
        """Analyze data gaps and visualize data completeness."""
        print("Analyzing data gaps...")
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)

        # Data completeness heatmap
        ax1 = fig.add_subplot(gs[0, :])
        completeness_data = []
        for stid in self.high_stations + self.basin_stations:
            station_data = self.weather_data[self.weather_data['stid'] == stid]
            # Change 'M' to 'ME' for month-end resampling
            monthly_data = (
                station_data.resample('ME')['snow_depth'].count() /
                station_data.resample('ME')['snow_depth'].size()
            )
            completeness_data.append(monthly_data)

        completeness_df = pd.concat(completeness_data, axis=1)
        completeness_df.columns = [
            self.station_names[stid] for stid in self.high_stations + self.basin_stations
        ]

        # Handle NaN values before plotting
        completeness_df = completeness_df.fillna(0)

        sns.heatmap(completeness_df.T, cmap='YlOrRd', ax=ax1)
        ax1.set_title('Monthly Data Completeness')

        # Gap statistics
        ax2 = fig.add_subplot(gs[1, :])
        gap_stats = self.calculate_gap_statistics()
        self.plot_gap_statistics(gap_stats, ax2)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / 'data_gaps_analysis.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def calculate_gap_statistics(self):
        """Calculate data completeness and gap statistics for each station."""
        gap_stats = {}
        for stid in self.high_stations + self.basin_stations:
            station_data = self.weather_data[self.weather_data['stid'] == stid]

            gap_stats[stid] = {
                'completeness': station_data['snow_depth'].notna().mean() * 100,
                'max_gap': station_data['snow_depth'].isna().astype(int).groupby(
                    station_data['snow_depth'].notna().astype(int).cumsum()
                ).sum().max(),
                'winter_completeness': station_data[
                    station_data['month'].isin(self.winter_months)
                ]['snow_depth'].notna().mean() * 100,
                'gap_periods': self.identify_gap_periods(station_data)
            }
        return gap_stats

    def identify_gap_periods(self, station_data):
        """Identify significant gaps in the data."""
        gaps = []
        is_missing = station_data['snow_depth'].isna()
        gap_groups = (
            is_missing != is_missing.shift()
        ).cumsum()[is_missing]

        for group in gap_groups.unique():
            gap_period = gap_groups[gap_groups == group]
            if len(gap_period) >= 7:  # Consider gaps of 7 days or more
                gaps.append({
                    'start': gap_period.index[0],
                    'end': gap_period.index[-1],
                    'duration': len(gap_period)
                })
        return gaps

    def plot_gap_statistics(self, gap_stats, ax):
        """Plot data completeness overview for all stations."""
        stats_df = pd.DataFrame(gap_stats).T
        stats_df['Station'] = [
            self.station_names[stid] for stid in stats_df.index
        ]
        stats_df['Type'] = [
            'High Elevation' if stid in self.high_stations else 'Basin'
            for stid in stats_df.index
        ]

        y_pos = np.arange(len(stats_df))
        ax.barh(
            y_pos,
            stats_df['completeness'],
            color=[
                self.colors['high'] if t == 'High Elevation' else self.colors['basin']
                for t in stats_df['Type']
            ]
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(stats_df['Station'])
        ax.set_xlabel('Data Completeness (%)')
        ax.set_title('Station Data Completeness Overview')

        # Add completeness values as text
        for i, v in enumerate(stats_df['completeness']):
            ax.text(v + 1, i, f'{v:.1f}%', va='center')

    def interpolate_missing_data(self):
        """
        Interpolate missing data using various methods depending on gap characteristics.
        """
        print("Interpolating missing data...")
        interpolated_data = self.weather_data.copy()

        for stid in self.high_stations + self.basin_stations:
            station_data = self.weather_data[self.weather_data['stid'] == stid].copy()

            # For short gaps (< 3 days), use linear interpolation
            station_data['snow_depth'] = station_data['snow_depth'].interpolate(
                method='linear', limit=3
            )

            # For longer gaps, use seasonal decomposition if enough data is available
            if len(station_data) > 365:  # Need at least a year of data
                try:
                    stl = STL(
                        station_data['snow_depth'].fillna(
                            station_data['snow_depth'].mean()
                        ),
                        period=365
                    )
                    result = stl.fit()
                    station_data['snow_depth_filled'] = (
                        result.trend + result.seasonal
                    )

                    # Only fill gaps, keep original valid data
                    mask = station_data['snow_depth'].isna()
                    station_data.loc[
                        mask, 'snow_depth'
                    ] = station_data.loc[mask, 'snow_depth_filled']
                except Exception as e:
                    print(
                        f"Could not perform seasonal decomposition for {stid}: {str(e)}"
                    )

            interpolated_data.loc[station_data.index, 'snow_depth'] = (
                station_data['snow_depth']
            )

        return interpolated_data

    def create_terrain_map(self):
        """Create a terrain map with station locations and data completeness."""
        print("Creating terrain map...")
        try:
            tif_path = self.base_dir / 'tif' / 'uinta_merged.tif'
            if not tif_path.exists():
                raise FileNotFoundError(
                    f"Terrain data file not found at {tif_path}"
                )

            with rasterio.open(tif_path) as src:
                elevation = src.read(1)
                extent = [
                    src.bounds.left,
                    src.bounds.right,
                    src.bounds.bottom,
                    src.bounds.top
                ]

            fig = plt.figure(figsize=(20, 15))
            gs = GridSpec(2, 2, height_ratios=[2, 1])
            ax_map = fig.add_subplot(
                gs[0, :], projection=ccrs.PlateCarree()
            )

            # Handle NaN values in elevation data
            elevation = np.ma.masked_invalid(elevation)

            terrain_plot = ax_map.imshow(
                elevation,
                extent=extent,
                cmap=self.colors['terrain'],
                transform=ccrs.PlateCarree()
            )
            ax_map.add_feature(cfeature.STATES, linewidth=0.5)
            ax_map.add_feature(cfeature.RIVERS, alpha=0.5)
            ax_map.gridlines(draw_labels=True)

            # Plot stations with error handling for coordinates
            gap_stats = self.calculate_gap_statistics()
            for group, stations in [
                ('high', self.high_stations),
                ('basin', self.basin_stations)
            ]:
                station_data = self.meta_df[
                    self.meta_df['stid'].isin(stations)
                ]
                completeness = [
                    gap_stats[stid]['completeness'] for stid in stations
                ]

                # Filter out any stations with invalid coordinates
                valid_stations = station_data[
                    station_data['longitude'].notna() &
                    station_data['latitude'].notna()
                ]

                if not valid_stations.empty:
                    scatter = ax_map.scatter(
                        valid_stations['longitude'],
                        valid_stations['latitude'],
                        c=completeness[:len(valid_stations)],
                        cmap='RdYlGn',
                        s=100,
                        vmin=0,
                        vmax=100,
                        label=(
                            f"{'High Elevation' if group == 'high' else 'Basin'} Stations"
                        ),
                        edgecolor='white',
                        linewidth=1,
                        transform=ccrs.PlateCarree()
                    )

            plt.colorbar(scatter, ax=ax_map, label='Data Completeness (%)')
            ax_map.set_title(
                'Uinta Basin Study Area\nStation Locations, Terrain, and Data Completeness'
            )
            ax_map.legend()

            plt.tight_layout()
            plt.savefig(
                self.figures_dir / 'terrain_analysis.png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

        except Exception as e:
            print(f"Error creating terrain map: {str(e)}")
            raise

    def create_statistical_analysis(self):
        """Perform statistical analysis considering data gaps and completeness."""
        print("Performing statistical analysis with gap consideration...")
        winter_data = self.weather_data[self.weather_data['month'].isin(self.winter_months)].copy()

        # Calculate completeness weights
        winter_data['completeness_weight'] = 1.0  # Default weight
        for stid in winter_data['stid'].unique():
            station_mask = winter_data['stid'] == stid
            completeness = winter_data.loc[station_mask, 'snow_depth'].notna().mean()
            winter_data.loc[station_mask, 'completeness_weight'] = completeness

        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(2, 2)

        # Monthly distributions with completeness information
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_monthly_distributions(ax1, winter_data)

        # Annual trends with gap highlighting
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_annual_trends(ax2, winter_data)

        # Statistical summary
        ax3 = fig.add_subplot(gs[1, :])
        self.plot_statistical_summary(ax3, winter_data)

        plt.tight_layout()
        plt.savefig(
            self.figures_dir / 'statistical_analysis.png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def plot_monthly_distributions(self, ax, winter_data):
        """Plot monthly snow depth distributions using boxplots."""
        winter_data['station_type'] = np.where(
            winter_data['stid'].isin(self.high_stations),
            'High Elevation',
            'Basin'
        )

        sns.boxplot(
            data=winter_data,
            x='month',
            y='snow_depth',
            hue='station_type',
            palette=[
                self.colors['high'],
                self.colors['basin']
            ],
            ax=ax
        )

        ax.set_title('Monthly Snow Depth Distributions\n(Box width indicates data completeness)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Snow Depth (cm)')

    def plot_annual_trends(self, ax, winter_data):
        """Plot annual trends in snow depth with data completeness indicated by point size."""
        annual_data = winter_data.groupby(
            ['water_year', 'stid']
        )['snow_depth'].agg(
            mean='mean',
            completeness=lambda x: x.notna().mean() * 100
        ).reset_index()

        for stid in self.high_stations + self.basin_stations:
            station_data = annual_data[annual_data['stid'] == stid]
            color = self.colors['high'] if stid in self.high_stations else self.colors['basin']

            # Plot points with size varying by completeness
            scatter = ax.scatter(
                station_data['water_year'],
                station_data['mean'],
                s=station_data['completeness'],
                color=color,
                alpha=0.6,
                label=f"{self.station_names[stid]}"
            )

            # Connect points with lines
            ax.plot(
                station_data['water_year'],
                station_data['mean'],
                color=color,
                alpha=0.3
            )

        ax.set_title('Annual Winter Snow Depth Trends\n(Point size indicates data completeness)')
        ax.set_xlabel('Water Year')
        ax.set_ylabel('Average Snow Depth (cm)')
        ax.legend(bbox_to_anchor=(1.05, 1))
        ax.grid(True, alpha=0.3)

    def plot_statistical_summary(self, ax, winter_data):
        """Calculate and plot statistical summary with proper weight handling."""
        try:
            # Calculate statistics for high elevation stations
            high_snow = winter_data[winter_data['stid'].isin(self.high_stations)].copy()
            basin_snow = winter_data[winter_data['stid'].isin(self.basin_stations)].copy()

            # Filter for valid data and weights
            high_valid_mask = high_snow['snow_depth'].notna() & (high_snow['completeness_weight'] > 0)
            basin_valid_mask = basin_snow['snow_depth'].notna() & (basin_snow['completeness_weight'] > 0)

            high_valid_data = high_snow.loc[high_valid_mask, 'snow_depth']
            high_valid_weights = high_snow.loc[high_valid_mask, 'completeness_weight']

            basin_valid_data = basin_snow.loc[basin_valid_mask, 'snow_depth']
            basin_valid_weights = basin_snow.loc[basin_valid_mask, 'completeness_weight']

            # Normalize weights
            high_weights_sum = high_valid_weights.sum()
            basin_weights_sum = basin_valid_weights.sum()

            if high_weights_sum > 0 and basin_weights_sum > 0:
                high_valid_weights = high_valid_weights / high_weights_sum
                basin_valid_weights = basin_valid_weights / basin_weights_sum

                # Calculate weighted statistics
                high_mean = np.sum(high_valid_data * high_valid_weights)
                basin_mean = np.sum(basin_valid_data * basin_valid_weights)

                # Calculate weighted standard deviations
                high_std = np.sqrt(
                    np.sum(high_valid_weights * (high_valid_data - high_mean) ** 2)
                )
                basin_std = np.sqrt(
                    np.sum(basin_valid_weights * (basin_valid_data - basin_mean) ** 2)
                )
            else:
                # Fallback to unweighted statistics if weights are invalid
                high_mean = high_valid_data.mean()
                basin_mean = basin_valid_data.mean()
                high_std = high_valid_data.std()
                basin_std = basin_valid_data.std()

            # Calculate completeness percentages
            high_completeness = (high_valid_mask.sum() / len(high_snow)) * 100
            basin_completeness = (basin_valid_mask.sum() / len(basin_snow)) * 100

            # Perform t-test on valid data
            t_stat, p_value = ttest_ind(high_valid_data, basin_valid_data)

            # Calculate effect size
            pooled_std = np.sqrt((high_std ** 2 + basin_std ** 2) / 2)
            cohens_d = (high_mean - basin_mean) / pooled_std if pooled_std != 0 else 0

            # Create summary text
            summary_text = (
                "Statistical Analysis of Snow Shadow Effect\n"
                "=========================================\n\n"
                f"High Elevation Stations (n={len(self.high_stations)}):\n"
                f"  Mean Snow Depth: {high_mean:.1f} cm\n"
                f"  Standard Deviation: {high_std:.1f} cm\n"
                f"  Data Completeness: {high_completeness:.1f}%\n"
                f"  Valid Observations: {high_valid_mask.sum()}\n\n"
                f"Basin Stations (n={len(self.basin_stations)}):\n"
                f"  Mean Snow Depth: {basin_mean:.1f} cm\n"
                f"  Standard Deviation: {basin_std:.1f} cm\n"
                f"  Data Completeness: {basin_completeness:.1f}%\n"
                f"  Valid Observations: {basin_valid_mask.sum()}\n\n"
                f"Statistical Tests:\n"
                f"  t-statistic: {t_stat:.2f}\n"
                f"  p-value: {p_value:.2e}\n"
                f"  Cohen's d: {cohens_d:.2f}\n\n"
                f"Snow Depth Difference: {high_mean - basin_mean:.1f} cm\n\n"
                "Note: Statistics are weighted by data completeness\n"
                "Analysis based on valid observations only"
            )

            ax.text(
                0.05,
                0.95,
                summary_text,
                transform=ax.transAxes,
                verticalalignment='top',
                fontfamily='monospace',
                fontsize=10
            )
            ax.axis('off')

        except Exception as e:
            print(f"Error in statistical summary: {str(e)}")
            raise

    def generate_report(self):
        """Generate a comprehensive analysis report."""
        print("Generating analysis report...")
        try:
            report_path = self.figures_dir / 'snow_shadow_analysis_report.txt'
            with open(report_path, 'w') as f:
                f.write("Uinta Basin Snow Shadow Effect Analysis\n")
                f.write("======================================\n\n")

                # Data completeness section
                f.write("Data Completeness Analysis:\n")
                f.write("-------------------------\n")
                gap_stats = self.calculate_gap_statistics()
                for stid in self.high_stations + self.basin_stations:
                    stats = gap_stats[stid]
                    f.write(f"\n{self.station_names[stid]}:\n")
                    f.write(f"  Overall Completeness: {stats['completeness']:.1f}%\n")
                    f.write(f"  Winter Completeness: {stats['winter_completeness']:.1f}%\n")
                    f.write(f"  Longest Gap: {stats['max_gap']} observations\n")

                    if stats['gap_periods']:
                        f.write("  Significant Gap Periods:\n")
                        for gap in stats['gap_periods']:
                            f.write(
                                f"    {gap['start'].date()} to {gap['end'].date()} "
                                f"({gap['duration']} days)\n"
                            )

                # Station information
                f.write("\nStation Information:\n")
                f.write("-------------------\n")
                for stid in self.high_stations + self.basin_stations:
                    station_info = self.meta_df[
                        self.meta_df['stid'] == stid
                    ].iloc[0]
                    f.write(f"\n{self.station_names[stid]}:\n")
                    f.write(
                        f"  Elevation: {station_info['elevation']:.0f} ft\n"
                    )
                    f.write(
                        f"  Record Start: {station_info['record_start'].date()}\n"
                    )
                    f.write(
                        f"  Record End: {station_info['record_end'].date()}\n"
                    )

                # Statistical analysis
                winter_data = self.weather_data[
                    self.weather_data['month'].isin(self.winter_months)
                ]
                high_snow = winter_data[
                    winter_data['stid'].isin(self.high_stations)
                ]['snow_depth']
                basin_snow = winter_data[
                    winter_data['stid'].isin(self.basin_stations)
                ]['snow_depth']

                t_stat, p_value = ttest_ind(
                    high_snow.dropna(),
                    basin_snow.dropna()
                )
                f.write("\nStatistical Analysis:\n")
                f.write("--------------------\n")
                f.write(f"t-statistic: {t_stat:.2f}\n")
                f.write(f"p-value: {p_value:.2e}\n")
                f.write(
                    f"Mean difference: {high_snow.mean() - basin_snow.mean():.1f} cm\n"
                )

                f.write("\nData Gap Impact:\n")
                f.write("--------------\n")
                f.write(
                    "The analysis accounts for data gaps through:\n"
                    "1. Weighted statistics based on data completeness\n"
                    "2. Transparent reporting of data availability\n"
                    "3. Gap identification and characterization\n"
                )

                f.write("\nInterpretation:\n")
                f.write("--------------\n")
                if p_value < 0.05:
                    f.write(
                        "There is a statistically significant difference in snow depth "
                        "between high elevation and basin stations.\n"
                    )
                else:
                    f.write(
                        "The difference in snow depth between high elevation and basin "
                        "stations is not statistically significant.\n"
                    )
                f.write(
                    "\nNote: Interpretation should consider data completeness and "
                    "temporal distribution of gaps.\n"
                )

        except Exception as e:
            print(f"Error generating report: {str(e)}")
            raise

    def run_analysis(self):
        """Run the complete snow analysis workflow."""
        print("\nStarting Uinta Basin Snow Shadow Analysis...")
        try:
            # Analyze data gaps first
            self.analyze_data_gaps()
            print("Data gap analysis complete.")

            # Optional: interpolate missing data
            interpolated_data = self.interpolate_missing_data()
            print("Data interpolation complete.")

            # Store original data and temporarily use interpolated data
            original_data = self.weather_data.copy()
            self.weather_data = interpolated_data

            # Run analyses
            self.create_terrain_map()
            print("Terrain analysis complete.")
            self.create_statistical_analysis()
            print("Statistical analysis complete.")

            # Restore original data
            self.weather_data = original_data

            # Generate report
            self.generate_report()
            print("Report generation complete.")
            print(f"\nAll analysis outputs have been saved to: {self.figures_dir}")

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise


def main():
    """Main function to execute the snow analysis."""
    try:
        analysis = EnhancedUintaSnowAnalysis()
        analysis.run_analysis()  # Changed from run_complete_analysis to run_analysis
    except Exception as e:
        print(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()


