import pandas as pd
import pickle
import numpy as np
from pathlib import Path


def create_metadata_df():
    # Load metadata
    base_dir = Path(__file__).parent
    metadata_file = base_dir / 'combined' / 'metadata_combined.pkl'

    with open(metadata_file, 'rb') as f:
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
            'record_start': data.get('RECORD_START', None),
            'record_end': data.get('RECORD_END', None)
        }
        stations_data.append(station_info)

    df = pd.DataFrame(stations_data)

    # Adjust classification boundaries
    # Wasatch Range (moved east to -110.8°)
    df['wasatch_position'] = np.where(df['longitude'] < -110.8,
                                      'windward', 'leeward')

    # Uinta Range (adjusted to 40.5°)
    df['uinta_position'] = np.where(df['latitude'] < 40.5,
                                    'windward', 'leeward')

    # Calculate record length in years
    df['record_length'] = (pd.to_datetime(df['record_end']) -
                           pd.to_datetime(df['record_start'])).dt.total_seconds() / (365.25 * 24 * 3600)

    return df


# Create and display the metadata DataFrame
meta_df = create_metadata_df()

# Display all stations with their classifications
print("\nComplete Station List:")
print(meta_df[
          ['stid', 'name', 'latitude', 'longitude', 'wasatch_position', 'uinta_position', 'elevation', 'record_length']]
      .sort_values(['wasatch_position', 'uinta_position', 'record_length', 'elevation'],
                   ascending=[True, True, False, True]))

# Display summary statistics
print("\nStation Classifications:")
print("\nWasatch Range Distribution:")
print(meta_df['wasatch_position'].value_counts())
print("\nUinta Range Distribution:")
print(meta_df['uinta_position'].value_counts())

# Create a scatterplot to visualize station locations
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
for wasatch in ['windward', 'leeward']:
    for uinta in ['windward', 'leeward']:
        mask = (meta_df['wasatch_position'] == wasatch) & (meta_df['uinta_position'] == uinta)
        plt.scatter(meta_df[mask]['longitude'],
                    meta_df[mask]['latitude'],
                    label=f'{wasatch}-{uinta}',
                    alpha=0.6)

plt.axvline(x=-110.8, color='r', linestyle='--', label='Wasatch Divide')
plt.axhline(y=40.5, color='g', linestyle='--', label='Uinta Divide')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Station Locations and Classifications')
plt.legend()
plt.grid(True)
plt.savefig('station_classifications.png')
plt.close()