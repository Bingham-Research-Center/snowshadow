import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import os
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap


class PosterDataGapAnalysis:
    def __init__(self):
        self.base_dir = Path('/Users/a02428741/PycharmProjects/snowshadow')
        self.data_path = self.base_dir / 'data' / 'combined'
        # Set style for poster-quality figures
        plt.style.use('seaborn-v0_8-whitegrid')
        self.set_poster_style()
        self.load_data()
        self.setup_output_dir()

    def set_poster_style(self):
        """Set matplotlib parameters for poster-quality figures"""
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        plt.rcParams['figure.dpi'] = 300

    def setup_output_dir(self):
        self.output_dir = self.base_dir / 'figures' / 'poster_data_gaps'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        print("Loading data files...")
        self.weather_data = pd.read_parquet(self.data_path / 'weather_data_combined.parquet')
        self.weather_data.index = pd.to_datetime(self.weather_data.index)

        with open(self.data_path / 'metadata_combined.pkl', 'rb') as f:
            self.metadata = pickle.load(f)

    def create_data_availability_plot(self):
        """Create a clean, poster-worthy data availability visualization"""
        print("Creating data availability plot...")

        # Calculate station completeness
        stats_df = self.calculate_station_stats()
        good_stations = stats_df[stats_df['completeness'] > 1].index.tolist()

        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 1, height_ratios=[2, 1], hspace=0.3)

        # Custom colormap
        colors = ['#f7fbff', '#08519c']  # Light blue to dark blue
        custom_cmap = LinearSegmentedColormap.from_list('custom_blues', colors)

        # Top plot: Data availability heatmap
        ax1 = fig.add_subplot(gs[0])
        data_matrix = self.weather_data[self.weather_data['stid'].isin(good_stations)].pivot(
            columns='stid', values='snow_depth'
        ).notna().astype(int)

        # Resample to weekly for better visualization
        weekly_data = data_matrix.resample('W').mean()

        im = ax1.imshow(weekly_data.T, aspect='auto', cmap=custom_cmap,
                        extent=[0, len(weekly_data), 0, len(good_stations)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Data Availability', rotation=270, labelpad=15)

        # Customize axes
        ax1.set_title('Weekly Data Availability by Station (2005-2024)', pad=20)
        ax1.set_ylabel('Station ID')
        ax1.set_yticks(np.arange(len(good_stations)))
        ax1.set_yticklabels(good_stations)

        # X-axis with years
        years = pd.date_range(weekly_data.index[0], weekly_data.index[-1], freq='Y')
        ax1.set_xticks(np.linspace(0, len(weekly_data), len(years)))
        ax1.set_xticklabels([year.year for year in years], rotation=45)

        # Bottom plot: Total available stations over time
        ax2 = fig.add_subplot(gs[1])
        station_counts = weekly_data.sum(axis=1)
        ax2.plot(range(len(station_counts)), station_counts,
                 color='#08519c', linewidth=2)

        # Customize bottom plot
        ax2.set_title('Number of Stations Reporting Data Over Time')
        ax2.set_ylabel('Number of Active Stations')
        ax2.set_xticks(np.linspace(0, len(weekly_data), len(years)))
        ax2.set_xticklabels([year.year for year in years], rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.savefig(self.output_dir / 'data_availability.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_station_quality_plot(self):
        """Create a visualization of station data quality"""
        print("Creating station quality plot...")

        stats_df = self.calculate_station_stats()

        # Filter for stations with any data
        active_stations = stats_df[stats_df['completeness'] > 1].sort_values('completeness', ascending=True)

        # Create figure
        fig = plt.figure(figsize=(15, 10))
        ax = plt.gca()

        # Create horizontal bar chart
        bars = ax.barh(range(len(active_stations)),
                       active_stations['completeness'],
                       color=plt.cm.viridis(np.linspace(0, 1, len(active_stations))))

        # Customize plot
        ax.set_title('Station Data Completeness\n(Stations with >1% Data Coverage)', pad=20)
        ax.set_xlabel('Data Completeness (%)')
        ax.set_ylabel('Station ID')
        ax.set_yticks(range(len(active_stations)))
        ax.set_yticklabels(active_stations.index)

        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                    f'{width:.1f}%',
                    ha='left', va='center')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'station_quality.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def create_seasonal_analysis(self):
        """Create seasonal analysis visualization"""
        print("Creating seasonal analysis plot...")

        # Get stations with good data coverage
        stats_df = self.calculate_station_stats()
        good_stations = stats_df[stats_df['completeness'] > 75].index.tolist()

        seasonal_data = self.weather_data[
            self.weather_data['stid'].isin(good_stations)
        ].copy()

        seasonal_data['month'] = seasonal_data.index.month
        monthly_completeness = seasonal_data.groupby(['stid', 'month'])['snow_depth'].apply(
            lambda x: x.notna().mean() * 100
        ).reset_index()

        # Create figure
        fig = plt.figure(figsize=(15, 10))
        ax = plt.gca()

        # Create violin plot
        sns.violinplot(data=monthly_completeness, x='month', y='snow_depth',
                       color='#08519c', alpha=0.6, ax=ax)

        # Customize plot
        ax.set_title('Seasonal Patterns in Data Availability\n(Stations with >75% Overall Completeness)',
                     pad=20)
        ax.set_xlabel('Month')
        ax.set_ylabel('Data Completeness (%)')

        # Set month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_labels)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(self.output_dir / 'seasonal_patterns.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    def calculate_station_stats(self):
        """Calculate statistics for each station"""
        stats = []
        for stid in self.weather_data['stid'].unique():
            station_data = self.weather_data[self.weather_data['stid'] == stid]
            stats.append({
                'stid': stid,
                'completeness': (1 - station_data['snow_depth'].isna().mean()) * 100,
                'total_records': len(station_data),
                'missing_records': station_data['snow_depth'].isna().sum(),
                'start_date': station_data.index.min(),
                'end_date': station_data.index.max()
            })
        return pd.DataFrame(stats).set_index('stid')

    def run_analysis(self):
        """Run the complete poster-quality analysis"""
        print("\nGenerating poster-quality visualizations...")
        self.create_data_availability_plot()
        self.create_station_quality_plot()
        self.create_seasonal_analysis()
        print(f"\nAnalysis complete. Poster figures saved to: {self.output_dir}")


if __name__ == "__main__":
    analysis = PosterDataGapAnalysis()
    analysis.run_analysis()