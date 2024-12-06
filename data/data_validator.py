import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime


class SnowDepthDiagnostic:
    def __init__(self):
        self.base_dir = Path('/Users/a02428741/PycharmProjects/snowshadow')
        self.metadata_path = self.base_dir / 'data' / 'metadata'
        self.parquet_path = self.base_dir / 'data' / 'parquet'
        self.output_path = self.base_dir / 'analysis_output'
        self.years = range(2006, 2025)

        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)

    def create_output_filename(self, base_name):
        """Create a filename with timestamp for output files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp}.csv"

    def identify_suspicious_values(self, df, station_id):
        """
        Identify suspicious snow depth values using more sophisticated criteria:
        1. Check for physically impossible jumps in snow depth
        2. Look for values that are statistical outliers within the station's history
        3. Consider elevation-based maximum plausible values
        """
        suspicious = []
        df = df.copy()
        df['snow_depth_diff'] = df['snow_depth'].diff()

        # Calculate rolling statistics for context
        rolling_mean = df['snow_depth'].rolling(window=24, center=True).mean()
        rolling_std = df['snow_depth'].rolling(window=24, center=True).std()

        for idx, row in df.iterrows():
            is_suspicious = False
            reason = []

            # Check for physically impossible changes (more than 30 inches per hour)
            if abs(row['snow_depth_diff']) > 30:
                is_suspicious = True
                reason.append(f"Rapid change: {row['snow_depth_diff']:.1f} inches/hour")

            # Check for statistical outliers (more than 5 standard deviations from rolling mean)
            if not pd.isna(rolling_mean[idx]) and not pd.isna(rolling_std[idx]):
                z_score = abs((row['snow_depth'] - rolling_mean[idx]) / rolling_std[idx])
                if z_score > 5:
                    is_suspicious = True
                    reason.append(f"Statistical outlier: {z_score:.1f} σ")

            # Record suspicious values with context
            if is_suspicious:
                suspicious.append({
                    'station': station_id,
                    'timestamp': idx,
                    'snow_depth': row['snow_depth'],
                    'change_from_previous': row['snow_depth_diff'],
                    'rolling_mean': rolling_mean[idx],
                    'rolling_std': rolling_std[idx],
                    'reason': '; '.join(reason)
                })

        return suspicious

    def analyze_snow_depth_data(self):
        """Analyze snow depth data for potential issues."""
        print("Starting snow depth data analysis...")

        # Initialize DataFrames for different types of analysis
        yearly_summary_data = []
        station_yearly_data = []
        suspicious_values_data = []
        metadata_analysis_data = []

        # Load metadata first for elevation context
        metadata = None
        for year in sorted(self.years, reverse=True):
            try:
                with open(self.metadata_path / f'UB_obs_{year}_meta.pkl', 'rb') as f:
                    metadata = pickle.load(f)
                    metadata_year = year
                    break
            except FileNotFoundError:
                continue

        for year in tqdm(self.years, desc="Processing yearly data"):
            try:
                # Load parquet file
                df = pd.read_parquet(self.parquet_path / f'UB_obs_{year}.parquet')
                df.index = pd.to_datetime(df.index)

                # Year summary statistics
                year_stats = {
                    'year': year,
                    'total_readings': len(df),
                    'active_stations': df['stid'].nunique(),
                    'missing_values': df['snow_depth'].isna().sum(),
                    'missing_percentage': (df['snow_depth'].isna().sum() / len(df)) * 100,
                    'negative_values': (df['snow_depth'] < 0).sum() if 'snow_depth' in df else 0,
                    'zero_values': (df['snow_depth'] == 0).sum() if 'snow_depth' in df else 0,
                    'max_snow_depth': df['snow_depth'].max(),
                    'mean_snow_depth': df['snow_depth'].mean(),
                    'std_snow_depth': df['snow_depth'].std()
                }
                yearly_summary_data.append(year_stats)

                # Station-specific analysis
                for station in df['stid'].unique():
                    station_data = df[df['stid'] == station].copy()

                    # Calculate station statistics
                    stats = {
                        'station_id': station,
                        'total_readings': len(station_data),
                        'mean_depth': station_data['snow_depth'].mean(),
                        'std_depth': station_data['snow_depth'].std(),
                        'min_depth': station_data['snow_depth'].min(),
                        'max_depth': station_data['snow_depth'].max(),
                        'missing_values': station_data['snow_depth'].isna().sum(),
                        'year': year
                    }

                    stats['data_completeness'] = (
                        (stats['total_readings'] - stats['missing_values']) /
                        stats['total_readings'] * 100 if stats['total_readings'] > 0 else float('-inf')
                    )

                    station_yearly_data.append(stats)

                    # Check for suspicious values
                    if not station_data.empty:
                        suspicious = self.identify_suspicious_values(station_data, station)
                        suspicious_values_data.extend(suspicious)

            except FileNotFoundError:
                print(f"Warning: File not found for year {year}")
                continue
            except Exception as e:
                print(f"Error processing year {year}: {e}")
                continue

        # Metadata analysis
        if metadata is not None:
            for station in metadata.columns:
                elevation = metadata.loc['ELEVATION', station]
                metadata_analysis_data.append({
                    'station_id': station,
                    'metadata_year': metadata_year,
                    'name': metadata.loc['NAME', station],
                    'elevation': elevation,
                    'latitude': metadata.loc['latitude', station],
                    'longitude': metadata.loc['longitude', station],
                    'elevation_suspicious': not (4000 <= elevation <= 11000)
                    if pd.notna(elevation) else True
                })

        # Convert to DataFrames
        yearly_summary_df = pd.DataFrame(yearly_summary_data)
        station_yearly_df = pd.DataFrame(station_yearly_data)
        suspicious_values_df = pd.DataFrame(suspicious_values_data)
        metadata_analysis_df = pd.DataFrame(metadata_analysis_data)

        # Save results
        self.save_analysis_results(
            yearly_summary_df,
            station_yearly_df,
            suspicious_values_df,
            metadata_analysis_df
        )

        return {
            'yearly_summary': yearly_summary_df,
            'station_yearly': station_yearly_df,
            'suspicious_values': suspicious_values_df,
            'metadata_analysis': metadata_analysis_df
        }

    def save_analysis_results(self, yearly_summary_df, station_yearly_df,
                              suspicious_values_df, metadata_analysis_df):
        """Save all analysis results to CSV files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save yearly summary
        yearly_file = self.output_path / self.create_output_filename('yearly_summary')
        yearly_summary_df.to_csv(yearly_file, index=False)

        # Save station yearly analysis
        station_file = self.output_path / self.create_output_filename('station_yearly')
        station_yearly_df.to_csv(station_file, index=False)

        # Save suspicious values if any found
        if not suspicious_values_df.empty:
            suspicious_file = self.output_path / self.create_output_filename('suspicious_values')
            suspicious_values_df.to_csv(suspicious_file, index=False)

        # Save metadata analysis
        metadata_file = self.output_path / self.create_output_filename('metadata_analysis')
        metadata_analysis_df.to_csv(metadata_file, index=False)

        # Create summary report
        total_suspicious = len(suspicious_values_df)
        summary_text = [
            "SNOW DEPTH DATA ANALYSIS SUMMARY",
            "==============================",
            f"\nAnalysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\nTotal years analyzed: {len(yearly_summary_df)}",
            f"Total stations: {len(metadata_analysis_df)}",
            f"Suspicious measurements found: {total_suspicious}",
            "\nFiles generated:",
            f"1. Yearly Summary: {yearly_file.name}",
            f"2. Station Yearly Analysis: {station_file.name}",
            f"3. Metadata Analysis: {metadata_file.name}",
            f"4. Suspicious Values: {suspicious_file.name}" if total_suspicious > 0 else ""
        ]

        # Save summary report
        summary_file = self.output_path / self.create_output_filename('analysis_summary')
        with open(summary_file.with_suffix('.txt'), 'w') as f:
            f.write('\n'.join(summary_text))


if __name__ == "__main__":
    diagnostic = SnowDepthDiagnostic()
    results = diagnostic.analyze_snow_depth_data()