import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def analyze_snow_stations():
    # Load metadata and combined weather data
    base_dir = Path(__file__).parent
    metadata_file = base_dir / 'combined' / 'metadata_combined.pkl'
    weather_file = base_dir / 'combined' / 'weather_data_combined.parquet'

    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    print("Loading weather data...")
    weather_data = pd.read_parquet(weather_file)

    # Create metadata DataFrame with classifications
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

    # Classification based on mountain ranges
    meta_df['wasatch_position'] = np.where(meta_df['longitude'] < -110.75, 'windward', 'leeward')
    meta_df['uinta_position'] = np.where(meta_df['latitude'] < 40.5, 'windward', 'leeward')

    # Analyze snow data completeness for each station with winter focus
    snow_stats = []

    for stid in meta_df['stid']:
        station_data = weather_data[weather_data['stid'] == stid].copy()
        station_data.index = pd.to_datetime(station_data.index)

        # Focus on winter months (Nov-Apr)
        winter_data = station_data[station_data.index.month.isin([11, 12, 1, 2, 3, 4])]

        # Calculate statistics for both all-year and winter-only
        stats_dict = {
            'stid': stid,
            'total_records': len(station_data),
            'winter_records': len(winter_data),
            'snow_depth_records': station_data['snow_depth'].notna().sum(),
            'snow_water_equiv_records': station_data['snow_water_equiv'].notna().sum(),
            'winter_snow_depth_records': winter_data['snow_depth'].notna().sum(),
            'winter_snow_water_equiv_records': winter_data['snow_water_equiv'].notna().sum(),
            'record_years': (station_data.index.max() - station_data.index.min()).days / 365.25,
            'snow_depth_pct': (station_data['snow_depth'].notna().sum() / len(station_data)) * 100 if len(
                station_data) > 0 else 0,
            'snow_water_equiv_pct': (station_data['snow_water_equiv'].notna().sum() / len(station_data)) * 100 if len(
                station_data) > 0 else 0,
            'winter_snow_depth_pct': (winter_data['snow_depth'].notna().sum() / len(winter_data)) * 100 if len(
                winter_data) > 0 else 0,
            'winter_snow_water_equiv_pct': (winter_data['snow_water_equiv'].notna().sum() / len(
                winter_data)) * 100 if len(winter_data) > 0 else 0
        }
        snow_stats.append(stats_dict)

    snow_stats_df = pd.DataFrame(snow_stats)

    # Merge with metadata
    analysis_df = pd.merge(meta_df, snow_stats_df, on='stid')

    # Filter for stations with significant snow data, using lower thresholds
    min_data_pct = 20  # Lowered threshold
    min_years = 2  # Minimum years of data

    snow_stations = analysis_df[
        (((analysis_df['winter_snow_depth_pct'] > min_data_pct) |
          (analysis_df['winter_snow_water_equiv_pct'] > min_data_pct)) &
         (analysis_df['record_years'] >= min_years))
    ].sort_values(['elevation', 'winter_snow_depth_pct', 'winter_snow_water_equiv_pct'],
                  ascending=[False, False, False])

    # Print results for each category
    categories = [
        ('leeward', 'windward', 'Basin Floor (Leeward Wasatch, Windward Uinta)'),
        ('leeward', 'leeward', 'High Uintas (Leeward both)')
    ]

    print("\nBest Stations for Snow Shadow Analysis:")
    for wasatch, uinta, description in categories:
        print(f"\n{description}:")
        stations = snow_stations[
            (snow_stations['wasatch_position'] == wasatch) &
            (snow_stations['uinta_position'] == uinta)
            ]

        print("\nStation Details (sorted by elevation):")
        elevation_sorted = stations.sort_values('elevation', ascending=False).head(10)
        print(elevation_sorted[[
            'stid', 'name', 'elevation',
            'winter_snow_depth_pct', 'winter_snow_water_equiv_pct',
            'record_years'
        ]].to_string())

        print("\nStation Details (sorted by data completeness):")
        completeness_sorted = stations.sort_values(['winter_snow_depth_pct', 'winter_snow_water_equiv_pct'],
                                                   ascending=[False, False]).head(10)
        print(completeness_sorted[[
            'stid', 'name', 'elevation',
            'winter_snow_depth_pct', 'winter_snow_water_equiv_pct',
            'record_years'
        ]].to_string())

        print(f"\nTotal qualifying stations: {len(stations)}")

    return snow_stations


# Run the analysis
snow_stations = analyze_snow_stations()