import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from pathlib import Path
import os
from tqdm import tqdm


class UintaBasinDataGapAnalysis:
    def __init__(self):
        self.base_dir = Path('/Users/a02428741/PycharmProjects/snowshadow')
        self.metadata_path = self.base_dir / 'data' / 'metadata'
        self.parquet_path = self.base_dir / 'data' / 'parquet'
        self.setup_output_dir()
        self.years = range(2006, 2025)  # 2006-2024
        self.winter_months = [11, 12, 1, 2, 3]  # November through March
        self.load_data()

    def setup_output_dir(self):
        self.output_dir = self.base_dir / 'figures' / 'data_gaps'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        print("Loading data files...")
        self.weather_data = pd.DataFrame()
        self.metadata = {}
        self.station_names = {}  # Dictionary to store stid -> name mappings

        # Load each year's data
        for year in tqdm(self.years, desc="Loading yearly data"):
            try:
                # Load parquet file
                parquet_file = self.parquet_path / f'UB_obs_{year}.parquet'
                year_data = pd.read_parquet(parquet_file)

                # Load metadata
                metadata_file = self.metadata_path / f'UB_obs_{year}_meta.pkl'
                with open(metadata_file, 'rb') as f:
                    year_metadata = pickle.load(f)
                    self.metadata[year] = year_metadata

                    # Store station names from metadata DataFrame
                    for stid in year_metadata.columns:
                        if 'NAME' in year_metadata.index:
                            self.station_names[stid] = year_metadata.loc['NAME', stid]

                # Ensure datetime index
                if not isinstance(year_data.index, pd.DatetimeIndex):
                    year_data.index = pd.to_datetime(year_data.index)

                # Append to main dataframe
                self.weather_data = pd.concat([self.weather_data, year_data])

            except Exception as e:
                print(f"Error loading data for {year}: {str(e)}")

        print(f"\nLoaded data from {len(self.weather_data)} records")
        print(f"Found {len(self.station_names)} stations with names")

        # Add month for filtering
        self.weather_data['month'] = self.weather_data.index.month
        self.weather_data['winter_season'] = self.weather_data.apply(
            lambda x: f"{x.name.year - 1}-{x.name.year}" if x['month'] in [1, 2, 3]
            else f"{x.name.year}-{x.name.year + 1}" if x['month'] in [11, 12]
            else None, axis=1
        )

    def filter_winter_data(self, data):
        """Filter for winter months only"""
        return data[data['month'].isin(self.winter_months)]

    def create_winter_availability_plot(self):
        """Create visualization of winter data availability"""
        print("Creating winter data availability plot...")

        # Get winter data only
        winter_data = self.filter_winter_data(self.weather_data)

        # Calculate availability by station and season
        winter_data['station_name'] = winter_data['stid'].map(self.station_names)

        # Group by station and winter season
        grouped_data = (
            winter_data
            .groupby(['stid', 'winter_season'])
            .agg(
                total_records=('snow_depth', 'size'),
                valid_records=('snow_depth', 'count')
            )
            .reset_index()
        )

        # Calculate percentages
        grouped_data['availability'] = (
                grouped_data['valid_records'] / grouped_data['total_records'] * 100
        )

        # Calculate overall station stats
        station_stats = (
            grouped_data
            .groupby('stid')
            .agg(
                mean_availability=('availability', 'mean'),
                total_records=('total_records', 'sum')
            )
            .sort_values('mean_availability', ascending=False)
        )

        # Filter for stations with sufficient data
        good_stations = station_stats[station_stats['mean_availability'] > 1].index

        # Add station names
        grouped_data['station_name'] = grouped_data['stid'].map(self.station_names)

        # Create pivot table for heatmap
        heatmap_data = (
            grouped_data[grouped_data['stid'].isin(good_stations)]
            .pivot(index='stid', columns='winter_season', values='availability')
            .fillna(0)
        )

        # Sort by overall availability
        heatmap_data = heatmap_data.loc[station_stats.index]

        # Replace index with station names
        heatmap_data.index = heatmap_data.index.map(self.station_names)

        # Create the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale=[
                [0, '#f7fbff'],
                [1, '#08519c']
            ],
            hoverongaps=False,
            text=np.round(heatmap_data.values, 1),
            hovertemplate=(
                    'Station: %{y}<br>' +
                    'Winter Season: %{x}<br>' +
                    'Data Availability: %{z:.1f}%<br>' +
                    '<extra></extra>'
            ),
            colorbar=dict(
                title='Data Availability (%)',
                titleside='right'
            )
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Winter Snow Data Availability in Uinta Basin (2006-2024)',
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            xaxis_title='Winter Season',
            yaxis_title='Station Location',
            width=1200,
            height=max(800, len(heatmap_data) * 25),  # Increased height per station
            yaxis=dict(
                tickfont=dict(size=10),
                title_font=dict(size=14),
                automargin=True,
                tickangle=0  # Keep station names horizontal
            ),
            xaxis=dict(
                tickangle=-45,
                title_font=dict(size=14),
                tickfont=dict(size=12),
                automargin=True
            ),
            margin=dict(l=200, r=80, t=50, b=50),  # Increased left margin for station names
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(
                family='Arial',
                size=14
            )
        )

        # Save the plot
        fig.write_html(str(self.output_dir / 'winter_data_availability.html'))
        print(f"Plot saved to: {self.output_dir / 'winter_data_availability.html'}")

        # Print summary statistics
        print("\nSummary Statistics:")
        print(f"Total Stations Analyzed: {len(good_stations)}")
        print("\nTop 10 Stations by Data Availability:")
        top_stations = (
            station_stats[station_stats.index.isin(good_stations)]
            .head(10)
            .copy()
        )
        top_stations.index = top_stations.index.map(self.station_names)
        print(top_stations.round(1))

    def run_analysis(self):
        """Run the complete analysis focusing on winter snow data gaps"""
        print("\nAnalyzing winter snow data gaps in Uinta Basin...")
        self.create_winter_availability_plot()
        print(f"\nAnalysis complete. Figures saved to: {self.output_dir}")


if __name__ == "__main__":
    analysis = UintaBasinDataGapAnalysis()
    analysis.run_analysis()