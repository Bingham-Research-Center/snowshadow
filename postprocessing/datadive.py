"""Wrapper for post-processing, data analysis, and visualization of observation data.
"""
# IMPORTS

import pandas as pd
import numpy as np

class DataDive:
    def __init__(self, df, metadata, save_new_df=None, process_df=True):
        """
        Initialize the DataDive object with a dataframe.

        Parameters:
        - df: pd.DataFrame
        """
        if process_df:
            self.df = self.fixing_dataframe_columns(df)
        else:
            self.df = df
        # Save memory by getting rid of the original copy
        del df

        if save_new_df:
            self.df.to_hdf(save_new_df, key='df_obs', mode='w')
        self.metadata = metadata

    def fixing_dataframe_columns(self,df):
        # This will only work if we don't change the vrbl list herein! Not elegant.
        # This should be done specifically for each experiment due to idiosyncracies of the data
        vrbl_names = [
            'air_temp_set_1',
            'altimeter_set_1',
            'dew_point_temperature_set_1',
            'elevation',
            'latitude',
            'longitude',
            'ozone_concentration_set_1',
            'pressure_set_1',
            'region',
            'relative_humidity_set_1',
            'sea_level_pressure_set_1',
            'snow_depth_set_1',
            'snow_water_equiv_set_1',
            # This is just the shallowest sensors to get a sense of "stored energy"
            'soil_temp_set_1',
            # Future work can include multiple levels
            # 'soil_temp_set_2',
            # 'soil_temp_set_3',
            # 'soil_temp_set_4',
            # 'soil_temp_set_5',
            'solar_radiation_set_1',
            'stid',
            'wind_direction_set_1',
            'wind_gust_set_1',
            'wind_speed_set_1',
        ]
        # Filter vrbl_names to include only those that exist in df.columns
        existing_columns = [col for col in vrbl_names if col in df.columns]
        pass

        # Create a new dataframe with only the existing, specified columns
        df = df[existing_columns]

        # Now we need to rename the columns to match the vrbls list
        # For each column, the string "_set_1" needs to be removed where it appears
        df = df.rename(columns=lambda x: x.replace('_set_1', ''))

        # Replace all inf and -inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        return df


    def create_representative_obs(self):
        """Combines data from multiple stations to create a representative observation.

        TODO: argument should include regions and which stations to use - needs data analysis to choose
        """
        # Placeholder for implementation
        pass

    def compute_basic_statistics(self):
        """
        Computes basic statistics (mean, median, std) for each variable/product.
        """
        # Placeholder for implementation

        # Nice seaborn style "Joy Division" joy-plot
        pass

    def aggregate_time_periods(self, period='M'):
        """
        Aggregates data over specified time periods.

        Parameters:
        - period: str, the period over which to aggregate (e.g., 'D', 'M', 'Y')
        """
        # Placeholder for implementation
        # How to account for UTC
        pass

# Example usage
# df_obs = pd.read_hdf('path/to/df_obs.h5')
# df_metadata = pd.read_hdf('path/to/df_metadata_bkup.h5')
# data_dive = DataDive(df_obs)
