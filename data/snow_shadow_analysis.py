import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go
import pickle
import logging
import matplotlib.pyplot as plt
import seaborn as sns


class UintaBasinSnowAnalysis:
    def __init__(self, data_units='inches', max_snow_depth=500):
        """
        Initialize the analysis.

        Parameters:
        - data_units (str): Units of 'snow_depth' in the data ('inches', 'cm', 'mm').
        - max_snow_depth (float): Maximum plausible snow depth in inches.
        """
        self.base_dir = Path('/Users/a02428741/PycharmProjects/snowshadow')
        self.metadata_path = self.base_dir / 'data' / 'metadata'
        self.parquet_path = self.base_dir / 'data' / 'parquet'
        self.output_dir = self.base_dir / 'figures'
        self.output_dir.mkdir(exist_ok=True)
        self.years = range(2006, 2025)
        self.winter_months = [11, 12, 1, 2, 3]
        self.data_units = data_units
        self.max_snow_depth = max_snow_depth

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def load_data(self):
        """Load and process winter snow data"""
        logging.info("Loading and analyzing winter data...")

        # Store winter snow depth measurements by station
        self.station_data = {}

        for year in tqdm(self.years, desc="Processing yearly data"):
            try:
                parquet_file = self.parquet_path / f'UB_obs_{year}.parquet'
                if not parquet_file.exists():
                    logging.warning(f"Parquet file for year {year} does not exist. Skipping.")
                    continue

                df = pd.read_parquet(parquet_file)
                df.index = pd.to_datetime(df.index)

                # Convert snow_depth to inches if necessary
                if self.data_units == 'mm':
                    df['snow_depth'] = df['snow_depth'] / 25.4  # mm to inches
                    logging.info(f"Converted 'snow_depth' from mm to inches for year {year}.")
                elif self.data_units == 'cm':
                    df['snow_depth'] = df['snow_depth'] / 2.54  # cm to inches
                    logging.info(f"Converted 'snow_depth' from cm to inches for year {year}.")
                elif self.data_units == 'inches':
                    logging.info(f"'snow_depth' is already in inches for year {year}. No conversion applied.")
                else:
                    logging.error(f"Unknown data_units '{self.data_units}'. Expected 'inches', 'cm', or 'mm'.")
                    raise ValueError(f"Unknown data_units '{self.data_units}'.")

                # Data Cleaning Steps
                initial_count = len(df)
                # Remove negative snow_depth values
                negative_values = df['snow_depth'] < 0
                num_negatives = negative_values.sum()
                if num_negatives > 0:
                    df.loc[negative_values, 'snow_depth'] = np.nan
                    logging.info(f"Year {year}: Removed {num_negatives} negative snow_depth values.")

                # Cap snow_depth at max_snow_depth
                excessive_values = df['snow_depth'] > self.max_snow_depth
                num_excessive = excessive_values.sum()
                if num_excessive > 0:
                    df.loc[excessive_values, 'snow_depth'] = np.nan
                    logging.info(
                        f"Year {year}: Capped {num_excessive} snow_depth values exceeding {self.max_snow_depth} inches.")

                # Validate snow_depth values
                logging.info(f"\nYear {year} Snow Depth Stats After Cleaning:")
                print(df['snow_depth'].describe())
                print("Sample Snow Depth Values:")
                # Display both head and tail samples
                sample_non_nan_head = df['snow_depth'].dropna().head(5)
                sample_non_nan_tail = df['snow_depth'].dropna().tail(5)
                if not sample_non_nan_head.empty:
                    print("First few valid snow_depth values:")
                    print(sample_non_nan_head)
                else:
                    print("No valid snow_depth values available at the beginning.")

                if not sample_non_nan_tail.empty:
                    print("Last few valid snow_depth values:")
                    print(sample_non_nan_tail)
                else:
                    print("No valid snow_depth values available at the end.")

                # Optionally, display random samples
                sample_random = df['snow_depth'].dropna().sample(n=5, random_state=42)
                print("Random snow_depth samples:")
                print(sample_random)

                # Plot distribution
                self.plot_snow_depth_distribution(df, year)

                # Filter for winter months
                df['month'] = df.index.month
                winter_data = df[df['month'].isin(self.winter_months)]

                if winter_data.empty:
                    logging.warning(f"No winter data for year {year}. Skipping.")
                    continue

                # Group by station and calculate statistics using named aggregations
                stats = winter_data.groupby('stid').agg(
                    total=('snow_depth', 'size'),
                    valid=('snow_depth', lambda x: x.notna().sum()),
                    mean=('snow_depth', 'mean'),
                    mean_nonzero=('snow_depth', lambda x: x[x > 0].mean()),
                    max=('snow_depth', 'max')
                )

                # Store stats for stations with valid measurements
                for stid, row in stats.iterrows():
                    if row['valid'] > 0:  # Only store if we have valid measurements
                        if stid not in self.station_data:
                            self.station_data[stid] = {}
                        self.station_data[stid][year] = {
                            'total': row['total'],
                            'valid': row['valid'],
                            'mean': row['mean'],
                            'mean_nonzero': row['mean_nonzero'],
                            'max': row['max']
                        }

            except Exception as e:
                logging.error(f"Error processing year {year}: {e}")

        logging.info(f"\nProcessed data for {len(self.station_data)} stations")

        # Load station metadata
        logging.info("\nLoading station metadata...")
        self.metadata = None
        for year in sorted(self.years, reverse=True):
            try:
                meta_file = self.metadata_path / f'UB_obs_{year}_meta.pkl'
                if not meta_file.exists():
                    logging.warning(f"Metadata file for year {year} does not exist. Skipping.")
                    continue

                with open(meta_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logging.info(f"Using metadata from {year}")
                break  # Use the most recent available metadata
            except Exception as e:
                logging.error(f"Error loading metadata for year {year}: {e}")
                continue

        if self.metadata is None:
            logging.critical("No metadata files found. Cannot proceed without metadata.")
            raise FileNotFoundError("No metadata files found. Cannot proceed without metadata.")

    def plot_snow_depth_distribution(self, df, year):
        """Plot the distribution of snow_depth for a given year"""
        plt.figure(figsize=(10, 6))
        sns.histplot(df['snow_depth'], bins=100, kde=True, color='skyblue')
        plt.title(f'Snow Depth Distribution for Year {year}')
        plt.xlabel('Snow Depth (inches)')
        plt.ylabel('Frequency')
        plt.xlim(0, self.max_snow_depth * 1.1)  # Adjust x-axis based on max_snow_depth
        plt.tight_layout()
        plt.show()

    def analyze_stations(self):
        """Analyze stations focusing on snow shadow effects"""
        station_summary = []

        for stid, years_data in self.station_data.items():
            if not years_data:  # Skip if no data
                continue

            years_with_data = sorted(years_data.keys())
            first_year = years_with_data[0]
            last_year = years_with_data[-1]
            total_years = last_year - first_year + 1

            # Calculate total valid measurements and weighted means
            total_valid = sum(data['valid'] for data in years_data.values())
            if total_valid == 0:
                continue

            # Weighted average calculation using non-zero means
            valid_means = [data['mean_nonzero'] for data in years_data.values()
                           if data['mean_nonzero'] is not None and not np.isnan(data['mean_nonzero'])]

            if not valid_means:
                continue

            weighted_sum = sum(data['mean_nonzero'] * data['valid'] for data in years_data.values()
                               if data['mean_nonzero'] is not None and not np.isnan(data['mean_nonzero']))
            weighted_avg_snow_depth = weighted_sum / total_valid

            # Maximum snow depth remains the maximum of all yearly maxes
            max_snow_depth = max(data['max'] for data in years_data.values() if not np.isnan(data['max']))

            try:
                station_summary.append({
                    'Station ID': stid,
                    'Name': self.metadata.loc['NAME', stid],
                    'Elevation (ft)': self.metadata.loc['ELEVATION', stid],
                    'Latitude': self.metadata.loc['latitude', stid],
                    'Longitude': self.metadata.loc['longitude', stid],
                    'First Winter': f"{first_year}-{first_year + 1}",
                    'Last Winter': f"{last_year}-{last_year + 1}",
                    'Years with Data': len(years_with_data),
                    'Winter Coverage %': round(len(years_with_data) / total_years * 100, 1),
                    'Avg Snow Depth': round(weighted_avg_snow_depth, 1),
                    'Max Snow Depth': round(max_snow_depth, 1)
                })
            except KeyError as e:
                logging.warning(f"Missing metadata for station {stid}: {e}")
                continue

        df = pd.DataFrame(station_summary)
        logging.info(f"\nAnalyzed {len(df)} stations with valid snow depth data")
        return df

    def print_summary_statistics(self, df):
        """Print summary statistics"""
        logging.info("\nSnow Depth Patterns by Elevation:")
        logging.info("==============================")

        # Create elevation bands
        df['Elevation Band'] = pd.cut(
            df['Elevation (ft)'],
            bins=range(4000, 11000, 1000),
            labels=[f'{i}-{i + 1}k ft' for i in range(4, 10)],
            right=False  # Include the left bin edge, exclude the right
        )

        elevation_summary = df.groupby('Elevation Band', observed=True).agg(
            **{
                'Station ID Count': ('Station ID', 'count'),
                'Elevation Min (ft)': ('Elevation (ft)', 'min'),
                'Elevation Max (ft)': ('Elevation (ft)', 'max'),
                'Avg Snow Depth (in)': ('Avg Snow Depth', 'mean'),
                'Std Dev Snow Depth (in)': ('Avg Snow Depth', 'std'),
                'Mean Max Snow Depth (in)': ('Max Snow Depth', 'mean'),
                'Mean Years with Data': ('Years with Data', 'mean')
            }
        ).round(1)

        print("\nStations per Elevation Band:")
        print(elevation_summary)

        # Print long-term stations
        logging.info("\nLong-term Stations (8+ years) by Elevation Band:")
        logging.info("============================================")
        long_term = df[df['Years with Data'] >= 8].sort_values('Elevation (ft)')

        for band in sorted(df['Elevation Band'].dropna().unique()):
            print(f"\n{band}:")
            band_stations = long_term[long_term['Elevation Band'] == band]
            if not band_stations.empty:
                for _, station in band_stations.iterrows():
                    print(f"{station['Name']} ({station['Elevation (ft)']} ft):")
                    print(f"  Average Snow Depth: {station['Avg Snow Depth']:.1f} inches")
                    print(f"  Maximum Snow Depth: {station['Max Snow Depth']:.1f} inches")
                    print(f"  Years of Data: {station['Years with Data']}")
            else:
                print("No long-term stations in this band")

        # Print west-east patterns
        logging.info("\nWest to East Snow Depth Patterns:")
        logging.info("===============================")
        df['Longitude Band'] = pd.qcut(
            df['Longitude'],
            q=4,
            labels=['West', 'West-Central', 'East-Central', 'East']
        )

        longitude_summary = df.groupby('Longitude Band', observed=True).agg(
            **{
                'Station ID Count': ('Station ID', 'count'),
                'Avg Snow Depth (in)': ('Avg Snow Depth', 'mean'),
                'Std Dev Snow Depth (in)': ('Avg Snow Depth', 'std'),
                'Mean Elevation (ft)': ('Elevation (ft)', 'mean')
            }
        ).round(1)

        print("\nWest to East Snow Depth Patterns:")
        print(longitude_summary)
        return df

    def analyze_and_visualize(self):
        """Run complete analysis with visualization"""
        self.load_data()
        df = self.analyze_stations()
        df = self.print_summary_statistics(df)
        return df


if __name__ == "__main__":
    # Initialize analysis with the correct data units and maximum snow depth threshold
    # Change 'inches' to 'mm' or 'cm' if your data is in those units
    # Adjust max_snow_depth based on your domain knowledge
    analysis = UintaBasinSnowAnalysis(data_units='inches', max_snow_depth=500)
    df = analysis.analyze_and_visualize()

    # Optionally, print some statistics to verify
    print("\nSample Statistics After Processing:")
    print(df[['Avg Snow Depth', 'Max Snow Depth']].describe())
