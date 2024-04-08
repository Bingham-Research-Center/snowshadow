"""Functions for filtering / processing observation and other data.
"""

import os

from scipy.signal import medfilt
import numpy as np

### FUNCTION ###

def replace_max_values(df, stid, vrbl, max_value=None):
    """Replace the maximum values in a column with NaNs.

    Args:
        df (pd.DataFrame): DataFrame of observation data
        stid (str): Station ID
        vrbl (str): Variable name
        max_value (float): Maximum value to replace.
            This is automatically computed if None.

    Returns:
        pd.DataFrame: DataFrame with maximum values replaced by NaNs then linearly interpolated
    """
    df_filtered = df[df["stid"] == stid].copy()
    if max_value is None:
        max_value = df_filtered[vrbl].max()

    # Replace the maximum values with NaN
    df_filtered.loc[df_filtered[vrbl] == max_value, vrbl] = np.nan

    # Interpolate the missing values linearly
    df_filtered.interpolate(method='linear')

    return df_filtered


def apply_median_filter(df, stid, vrbl, kernel_size):
    """Apply a median filter to a column in a DataFrame.

    This is needed because of sudden large outliers in the data. The kernel size should increase with the
    reporting frequency of the data and/or its noisiness. Long run - make kernel a function of reporting freq, etc.

    Args:
        df (pd.DataFrame): DataFrame of observation data
        stid (str): Station ID
        vrbl (str): Variable name
        kernel_size (int): Size of the kernel for the median filter

    TODO: explain the median filter better

    """
    df_filtered = df[df["stid"] == stid].copy()

    # Apply a median filter with a kernel size of your choice
    # The kernel size determines how many neighboring points are considered (choose an odd number)
    # Have to use 32 bit or higher for this
    df_filtered[vrbl] = medfilt(df_filtered[vrbl].astype('float32'), kernel_size=kernel_size)
    return df_filtered

def filter_data(df, stid, vrbl, kernel_size, filt_method="max_median"):
    """Filter data using a specified method.

    Usage:
        The only method implemented so far is "max_median", which
            1. replaces the maximum values (erroneous data) in a column with NaNs
            2. Interpolates linearly to replace the NaNs
            3. Applies a median filter to the column to remove outliers

    (Other methods here)

    Args:
        df (pd.DataFrame): DataFrame of observation data
        stid (str): Station ID
        vrbl (str): Variable name
        kernel_size (int): Size of the kernel for the median filter
        filt_method (str): Method to use for filtering the data

    Returns:
        pd.DataFrame: DataFrame with the filtered data
    """
    if filt_method == "max_median":
        filtered_df = apply_median_filter(replace_max_values(df, stid, vrbl), stid, vrbl, kernel_size=kernel_size)
    else:
        raise NotImplementedError(f"Filter method {filt_method} not implemented.")
    return filtered_df




