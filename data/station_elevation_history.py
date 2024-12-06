import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np


def analyze_station_history():
    base_dir = Path('/Users/a02428741/PycharmProjects/snowshadow')
    metadata_path = base_dir / 'data' / 'metadata'
    parquet_path = base_dir / 'data' / 'parquet'
    years = range(2006, 2025)

    print("Loading and analyzing parquet files...")

    # Store all station data records
    all_stations = {}  # {stid: {year: {'valid': count, 'total': count}}}

    for year in tqdm(years, desc="Processing yearly data"):
        try:
            df = pd.read_parquet(parquet_path / f'UB_obs_{year}.parquet')

            # Group by station and count records
            station_counts = df.groupby('stid').agg({
                'snow_depth': [
                    ('total', 'size'),
                    ('valid', lambda x: x.notna().sum())
                ]
            })
            station_counts.columns = station_counts.columns.droplevel()

            # Store counts for each station
            for stid, row in station_counts.iterrows():
                if stid not in all_stations:
                    all_stations[stid] = {}
                if row['valid'] > 0:  # Only store if we have valid measurements
                    all_stations[stid][year] = {
                        'total': row['total'],
                        'valid': row['valid']
                    }

        except FileNotFoundError:
            print(f"File not found for year {year}")
        except Exception as e:
            print(f"Error processing year {year}: {e}")

    # Load most recent metadata just for station names and locations
    print("\nLoading metadata for station information...")
    station_info = {}
    for year in sorted(years, reverse=True):
        try:
            with open(metadata_path / f'UB_obs_{year}_meta.pkl', 'rb') as f:
                metadata = pickle.load(f)
                print(f"Using metadata from {year} for station information")

                # Store station information
                for stid in metadata.columns:
                    if stid not in station_info:
                        station_info[stid] = {
                            'name': metadata.loc['NAME', stid],
                            'elevation': metadata.loc['ELEVATION', stid],
                            'latitude': metadata.loc['latitude', stid],
                            'longitude': metadata.loc['longitude', stid]
                        }
                break
        except Exception as e:
            continue

    # Compile station statistics
    station_data = []
    for stid, years_data in all_stations.items():
        if not years_data:  # Skip if no valid data
            continue

        # Get years with data
        years_with_data = sorted(years_data.keys())
        first_year = years_with_data[0]
        last_year = years_with_data[-1]
        total_period = last_year - first_year + 1

        # Calculate coverage percentages
        yearly_coverage = len(years_with_data) / total_period * 100

        # Calculate average data coverage within active years
        data_coverages = [
            years_data[year]['valid'] / years_data[year]['total'] * 100
            for year in years_with_data
        ]
        avg_data_coverage = np.mean(data_coverages)

        # Get missing years
        missing_years = sorted(set(range(first_year, last_year + 1)) - set(years_with_data))

        # Get station info
        info = station_info.get(stid, {'name': stid, 'elevation': None, 'latitude': None, 'longitude': None})

        station_data.append({
            'Station ID': stid,
            'Name': info['name'],
            'First Year': first_year,
            'Last Year': last_year,
            'Total Years': total_period,
            'Years with Data': len(years_with_data),
            'Year Coverage %': round(yearly_coverage, 1),
            'Data Coverage %': round(avg_data_coverage, 1),
            'Elevation (ft)': info['elevation'],
            'Latitude': info['latitude'],
            'Longitude': info['longitude'],
            'Missing Years': ', '.join(map(str, missing_years)) if missing_years else 'None',
            'Total Valid Measurements': sum(years_data[year]['valid'] for year in years_with_data)
        })

    # Convert to DataFrame and sort
    df = pd.DataFrame(station_data)

    # Print Station Distribution by Elevation
    print("\nStation Distribution by Elevation:")
    print("================================")

    # Create elevation bands (every 1000 feet)
    df['Elevation Band'] = pd.cut(
        df['Elevation (ft)'],
        bins=range(4000, 11000, 1000),
        labels=[f'{i}-{i + 1}k ft' for i in range(4, 10)],
        include_lowest=True
    )

    elevation_summary = df.groupby('Elevation Band').agg({
        'Station ID': 'count',
        'Elevation (ft)': ['min', 'max'],
        'Data Coverage %': 'mean',
        'Years with Data': 'mean'
    }).round(1)

    print("\nStations per Elevation Band:")
    print(elevation_summary)

    # Print long-term stations by elevation band
    print("\nLong-term Stations (8+ years) by Elevation Band:")
    print("============================================")
    long_term = df[df['Years with Data'] >= 8].sort_values('Elevation (ft)')

    for band in df['Elevation Band'].unique():
        print(f"\n{band}:")
        band_stations = long_term[long_term['Elevation Band'] == band]
        if not band_stations.empty:
            print(band_stations[['Name', 'Elevation (ft)', 'Years with Data',
                                 'Data Coverage %']].to_string())
        else:
            print("No long-term stations in this band")

    # List stations by elevation range
    print("\nAll Stations by Elevation Band:")
    print("============================")

    for band in sorted(df['Elevation Band'].unique()):
        band_stations = df[df['Elevation Band'] == band].sort_values('Elevation (ft)')
        print(f"\n{band} ({len(band_stations)} stations):")
        print(band_stations[['Name', 'Elevation (ft)', 'Years with Data',
                             'Data Coverage %']].to_string())

    return df


if __name__ == "__main__":
    df = analyze_station_history()