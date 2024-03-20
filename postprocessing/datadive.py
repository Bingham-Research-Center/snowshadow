import pandas as pd


class DataDive:
    def __init__(self, df, metadata):
        """
        Initialize the DataDive object with a dataframe.

        Parameters:
        - df: pd.DataFrame
        """
        self.df = df
        self.metadata = metadata

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
