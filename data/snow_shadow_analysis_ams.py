# snow_shadow_analysis_ams.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
import rasterio
from rasterio.plot import show
import pickle
import matplotlib.colors as colors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def setup_directories():
    """Create necessary directories for output files"""
    base_dir = Path(__file__).parent
    figures_dir = base_dir / 'figures' / 'ams_presentation'
    figures_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, figures_dir


def load_data():
    """Load the metadata and weather data"""
    base_dir = Path(__file__).parent

    # Load metadata
    with open(base_dir / 'combined' / 'metadata_combined.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # Convert metadata to DataFrame
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

    meta_df = pd.DataFrame(stations_data)

    # Calculate record length
    meta_df['record_length'] = (meta_df['record_end'] - meta_df['record_start']).dt.total_seconds() / (
                365.25 * 24 * 3600)

    # Load weather data
    weather_data = pd.read_parquet(base_dir / 'combined' / 'weather_data_combined.parquet')

    return meta_df, weather_data


def create_comprehensive_analysis():
    # Setup directories and load data
    base_dir, figures_dir = setup_directories()
    meta_df, weather_data = load_data()

    def create_terrain_map():
        """Create professional terrain map with stations and context"""
        # Load the TIF file
        tif_path = base_dir / 'tif' / 'uinta_merged.tif'
        with rasterio.open(tif_path) as src:
            elevation = src.read(1)
            extent = [src.bounds.left, src.bounds.right,
                      src.bounds.bottom, src.bounds.top]

        # Create figure
        fig = plt.figure(figsize=(15, 12))

        # Main map
        ax = plt.axes()

        # Custom terrain colormap
        terrain_colors = plt.cm.terrain(np.linspace(0, 1, 256))
        terrain_colors[:25, 3] = np.linspace(0, 1, 25)  # Add transparency to lower elevations
        custom_terrain = colors.ListedColormap(terrain_colors)

        # Plot terrain
        terrain_plot = ax.imshow(elevation, extent=extent,
                                 cmap=custom_terrain,
                                 aspect='equal')

        # Plot stations with size based on record length
        scatter = ax.scatter(meta_df['longitude'], meta_df['latitude'],
                             c=meta_df['elevation'],
                             s=meta_df['record_length'] * 5,  # Scale dot size by record length
                             cmap='viridis',
                             edgecolor='white',
                             linewidth=0.5,
                             alpha=0.8)

        # Add mountain ranges with custom styling
        wasatch = ax.plot([-110.75, -110.75], [39.5, 41],
                          color='red', linestyle='--',
                          linewidth=2, label='Wasatch Range')
        uinta = ax.plot([-111, -109], [40.5, 40.5],
                        color='green', linestyle='--',
                        linewidth=2, label='Uinta Range')

        # Add labels for key stations
        key_stations = ['MMTU1', 'TCKU1', 'KGCU1', 'UBHSP', 'UBMYT', 'UBMTH']
        for idx, row in meta_df[meta_df['stid'].isin(key_stations)].iterrows():
            ax.annotate(row['stid'],
                        (row['longitude'], row['latitude']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='white',
                        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))

        # Customize axes
        ax.set_xlabel('Longitude (°W)', fontsize=12)
        ax.set_ylabel('Latitude (°N)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add colorbars
        terrain_cbar = plt.colorbar(terrain_plot, label='Terrain Elevation (m)', pad=0.01)
        station_cbar = plt.colorbar(scatter, label='Station Elevation (ft)', pad=0.05)

        # Add title
        plt.title('Uinta Basin Station Network\nTerrain and Weather Station Locations',
                  fontsize=16, pad=20)

        # Add legend with custom entries
        legend_elements = [
            plt.Line2D([0], [0], color='red', linestyle='--', label='Wasatch Range'),
            plt.Line2D([0], [0], color='green', linestyle='--', label='Uinta Range'),
            plt.scatter([0], [0], c='white', edgecolor='black',
                        label='Weather Stations\n(size indicates record length)')
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                  bbox_to_anchor=(1.15, 1), fontsize=10)

        # Add scale bar
        scale_bar_length = 0.5  # degrees longitude
        scale_bar = Rectangle((extent[0] + 0.1, extent[2] + 0.1),
                              scale_bar_length, 0.02,
                              facecolor='white', edgecolor='black')
        ax.add_patch(scale_bar)
        ax.text(extent[0] + 0.1 + scale_bar_length / 2, extent[2] + 0.15,
                f'{scale_bar_length:.1f}°', ha='center', va='bottom',
                color='white', fontsize=10)

        # Add simple inset map instead of cartopy version
        axins = inset_axes(ax, width="30%", height="30%",
                           loc='lower left', bbox_to_anchor=(0.05, 0.05, 1, 1),
                           bbox_transform=ax.transAxes)

        # Simplified US outline for context
        axins.set_xlim([-125, -65])
        axins.set_ylim([25, 50])
        axins.set_facecolor('lightgray')

        # Show study area as rectangle
        study_area = Rectangle(
            (extent[0], extent[2]),
            extent[1] - extent[0],
            extent[3] - extent[2],
            facecolor='red',
            alpha=0.5
        )
        axins.add_patch(study_area)
        axins.set_title('Study Area', fontsize=8)

        plt.savefig(figures_dir / 'uinta_basin_terrain_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_elevation_analysis():
        """Create elevation profile analysis"""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2, figure=fig)

        # East-West Profile
        ax1 = fig.add_subplot(gs[0, :])
        sns.scatterplot(data=meta_df, x='longitude', y='elevation',
                        size='record_length', sizes=(20, 200),
                        hue='elevation', palette='viridis',
                        ax=ax1)
        ax1.set_title('East-West Elevation Profile', fontsize=12)
        ax1.axvline(x=-110.75, color='red', linestyle='--',
                    label='Wasatch Range')
        ax1.set_xlabel('Longitude (°W)')
        ax1.set_ylabel('Elevation (ft)')

        # North-South Profile
        ax2 = fig.add_subplot(gs[1, 0])
        sns.scatterplot(data=meta_df, x='latitude', y='elevation',
                        size='record_length', sizes=(20, 200),
                        hue='elevation', palette='viridis',
                        ax=ax2)
        ax2.set_title('North-South Elevation Profile', fontsize=12)
        ax2.axhline(y=40.5, color='green', linestyle='--',
                    label='Uinta Range')
        ax2.set_xlabel('Latitude (°N)')
        ax2.set_ylabel('Elevation (ft)')

        # Add elevation distribution
        ax3 = fig.add_subplot(gs[1, 1])
        sns.histplot(data=meta_df, x='elevation', bins=20,
                     ax=ax3, color='skyblue')
        ax3.set_title('Station Elevation Distribution', fontsize=12)
        ax3.set_xlabel('Elevation (ft)')
        ax3.set_ylabel('Count')

        plt.tight_layout()
        plt.savefig(figures_dir / 'elevation_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_snow_patterns():
        """Analyze snow patterns across stations"""
        # Select our best station pairs
        high_stations = ['MMTU1', 'TCKU1', 'KGCU1']
        basin_stations = ['UBHSP', 'UBMYT', 'UBMTH']

        # Filter for winter months
        winter_data = weather_data[
            (weather_data['stid'].isin(high_stations + basin_stations)) &
            (weather_data.index.month.isin([11, 12, 1, 2, 3, 4]))
            ].copy()

        winter_data.index = pd.to_datetime(winter_data.index)

        # Calculate monthly statistics and keep the datetime index
        monthly_stats = (winter_data.groupby(['stid', pd.Grouper(freq='ME')])['snow_depth']
                         .agg(['mean', 'std'])
                         .reset_index())

        # Print DataFrame info for debugging
        print("\nMonthly stats columns:", monthly_stats.columns.tolist())
        print("\nFirst few rows of monthly_stats:")
        print(monthly_stats.head())

        # Statistical analysis
        high_snow = winter_data[winter_data['stid'].isin(high_stations)]['snow_depth']
        basin_snow = winter_data[winter_data['stid'].isin(basin_stations)]['snow_depth']

        t_stat, p_value = stats.ttest_ind(high_snow.dropna(), basin_snow.dropna())

        return monthly_stats, t_stat, p_value, high_stations + basin_stations

    def analyze_snow_patterns():
        """Analyze snow patterns across stations"""
        # Select our best station pairs
        high_stations = ['MMTU1', 'TCKU1', 'KGCU1']
        basin_stations = ['UBHSP', 'UBMYT', 'UBMTH']

        # Filter for winter months
        winter_data = weather_data[
            (weather_data['stid'].isin(high_stations + basin_stations)) &
            (weather_data.index.month.isin([11, 12, 1, 2, 3, 4]))
            ].copy()

        winter_data.index = pd.to_datetime(winter_data.index)

        # Convert snow depth from mm to cm
        winter_data['snow_depth'] = winter_data['snow_depth'] / 10.0

        # Add basic quality control
        winter_data.loc[winter_data['snow_depth'] > 300, 'snow_depth'] = np.nan  # Flag unrealistic values

        # Calculate monthly statistics
        monthly_stats = winter_data.groupby(['stid', pd.Grouper(freq='ME')])['snow_depth'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()

        # Add station metadata
        monthly_stats = monthly_stats.merge(
            meta_df[['stid', 'elevation']],
            on='stid',
            how='left'
        )

        # Statistical analysis
        high_snow = winter_data[winter_data['stid'].isin(high_stations)]['snow_depth']
        basin_snow = winter_data[winter_data['stid'].isin(basin_stations)]['snow_depth']

        t_stat, p_value = stats.ttest_ind(high_snow.dropna(), basin_snow.dropna())

        return monthly_stats, t_stat, p_value, high_stations + basin_stations

    def create_snow_analysis_plots(monthly_stats, selected_stations):
        """Create comprehensive snow analysis visualizations"""
        plt.style.use('seaborn-v0_8-whitegrid')  # Use a clean, professional style

        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 1, height_ratios=[1, 1.5, 0.1], hspace=0.3)

        # 1. Monthly snow depth patterns
        ax1 = fig.add_subplot(gs[0])
        sns.boxplot(data=monthly_stats, x='stid', y='mean',
                    ax=ax1, palette='coolwarm')

        # Add elevation text above each box
        for i, stid in enumerate(monthly_stats['stid'].unique()):
            elev = monthly_stats[monthly_stats['stid'] == stid]['elevation'].iloc[0]
            ax1.text(i, ax1.get_ylim()[1], f'{elev:.0f} ft',
                     ha='center', va='bottom', rotation=0, fontsize=8)

        ax1.set_title('Monthly Snow Depth Distribution by Station', pad=20, fontsize=12)
        ax1.set_xlabel('Station ID', fontsize=10)
        ax1.set_ylabel('Snow Depth (cm)', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)

        # 2. Time series of snow depth
        ax2 = fig.add_subplot(gs[1])

        # Create custom colormap for elevation-based coloring
        elevations = meta_df[meta_df['stid'].isin(selected_stations)]['elevation']
        norm = plt.Normalize(elevations.min(), elevations.max())

        for station in selected_stations:
            station_data = monthly_stats[monthly_stats['stid'] == station]
            station_elev = meta_df[meta_df['stid'] == station]['elevation'].iloc[0]

            color = plt.cm.viridis(norm(station_elev))

            # Plot line with error bands
            ax2.plot(station_data['date_time'], station_data['mean'],
                     label=f'{station} ({station_elev:.0f} ft)',
                     color=color, alpha=0.8, linewidth=1.5)

            # Add error bands
            ax2.fill_between(station_data['date_time'],
                             station_data['mean'] - station_data['std'],
                             station_data['mean'] + station_data['std'],
                             color=color, alpha=0.2)

        ax2.set_title('Snow Depth Time Series by Station', pad=20, fontsize=12)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Snow Depth (cm)', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)

        # Customize legend
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize=8, frameon=True, edgecolor='black')

        # Add colorbar for elevation
        ax3 = fig.add_subplot(gs[2])
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis),
                          cax=ax3, orientation='horizontal')
        cb.set_label('Station Elevation (ft)', fontsize=10)

        # Add a title for the entire figure
        fig.suptitle('Snow Depth Analysis in the Uinta Basin\nEvidence of Snow Shadow Effect',
                     fontsize=14, y=0.95)

        # Adjust layout
        plt.savefig(figures_dir / 'snow_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Execute analyses
    print("Starting comprehensive snow shadow analysis...")

    print("Creating terrain map...")
    create_terrain_map()

    print("Creating elevation analysis...")
    create_elevation_analysis()

    print("Analyzing snow patterns...")
    monthly_stats, t_stat, p_value, selected_stations = analyze_snow_patterns()

    print("Creating snow analysis plots...")
    create_snow_analysis_plots(monthly_stats, selected_stations)

    # Generate summary statistics
    summary_stats = {
        "elevation_difference": meta_df['elevation'].max() - meta_df['elevation'].min(),
        "t_statistic": t_stat,
        "p_value": p_value
    }

    # Create summary report
    report_path = figures_dir / 'analysis_summary.txt'
    with open(report_path, 'w') as f:
        f.write("Uinta Basin Snow Shadow Effect Analysis\n")
        f.write("======================================\n\n")
        f.write(f"Maximum Elevation Difference: {summary_stats['elevation_difference']:.0f} ft\n")
        f.write(f"Statistical Significance:\n")
        f.write(f"T-statistic: {summary_stats['t_statistic']:.3f}\n")
        f.write(f"P-value: {summary_stats['p_value']:.3e}\n")

    print(f"\nAll analysis outputs have been saved to: {figures_dir}")
    return summary_stats


if __name__ == "__main__":
    print("Beginning Uinta Basin Snow Shadow Analysis for AMS Presentation...")
    summary_stats = create_comprehensive_analysis()
    print("\nAnalysis complete!")