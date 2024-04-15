"""Functions for filtering / processing observation and other data.

TODO:
* There are numerous copy operations in these functions. Need to watch for memory and copying unnecessarily.
"""

import os

import pandas as pd
from scipy.signal import medfilt
import numpy as np

from utils.lookups import kernel_lookup

### FUNCTION ###

def replace_max_values(df, stid, vrbl, max_value=None):
    """Replace the maximum values in a column with NaNs.

    TODO: is this always needed? E.g., COOP stations won't need the max removing

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

    # Interpolate the missing values linearly - TODO fix warning
    df_filtered = df_filtered.infer_objects(copy=False)
    df_filtered = df_filtered.interpolate(method='linear')
    pass
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
    filt_vname = f"{vrbl}_filtered"

    # Subset copy of this stid only (might already be done!)
    df_filtered = df[df["stid"] == stid].copy()

    # The kernel size determines how many neighboring points are considered (choose an odd number)
    # Have to use 32 bit or higher for this
    df_filtered[filt_vname] = medfilt(df_filtered[vrbl].astype('float32'), kernel_size=kernel_size)

    # Return the column to 16 bit float to save space
    df_filtered[vrbl] = df_filtered[vrbl].astype('float16')
    return df_filtered


def filter_data(df, stid, vrbl, kernel_size, filt_method="max_median") -> pd.DataFrame:
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
    # TODO - change to only filter for non-COOP stations
    if filt_method == "max_median":
        if stid.startswith("COOP"):
            filtered_df = df
            print("No max filtering for COOP station.")
        else:
            filtered_df = replace_max_values(df, stid, vrbl)
            print("Max filtering applied.")
        filtered_df = apply_median_filter(filtered_df, stid, vrbl, kernel_size=kernel_size)
    else:
        raise NotImplementedError(f"Filter method {filt_method} not implemented.")
    return filtered_df


def filter_all_stations(df, vrbl, filt_method="max_median",default_kernel_size=1):
    """For each stid, filter the column "vrbl" in the DataFrame "df".

    This function ensures that a new column, named "{vrbl}_filtered", is added to the
    DataFrame containing filtered data based on the specified filtering method and
    dynamic kernel size.

    Args:
        df (pd.DataFrame): DataFrame of observation data with DateTimeIndex.
        vrbl (str): Variable name to filter.
        filt_method (str): Method to use for filtering the data.

    Returns:
        pd.DataFrame: Updated DataFrame with the filtered data in "{vrbl}_filtered" column.
    """
    # We need the following function to split data by stid, sort by time, apply filter, and then recombine
    # Copying, sorting, subsetting by stid, etc, should be done only once.


    filt_vname = f"{vrbl}_filtered"
    if filt_vname in df.columns:
        raise ValueError(f"Column {filt_vname} already exists in DataFrame.")

    df_filtered_slices = []
    for stid in df["stid"].unique():
        stid_df = df[df["stid"] == stid].copy()
        stid_df.sort_index(inplace=True)

        # Setting kernel size for filtering
        try:
            ks = kernel_lookup[stid]
        except KeyError:
            ks = default_kernel_size
            print(f"Kernel size not found for {stid} ==> using default ({ks=})")

        stid_df_filtered = filter_data(stid_df, stid, vrbl, kernel_size=ks, filt_method=filt_method)
        stid_df_filtered = stid_df_filtered[[vrbl, filt_vname]]
        stid_df_filtered['stid'] = stid
        df_filtered_slices.append(stid_df_filtered)

    # Concatenate all filtered slices together
    df_filtered_concat = pd.concat(df_filtered_slices)

    # df.reset_index(inplace=True)
    # df_filtered_concat.reset_index(inplace=True)

    # df_final = pd.merge(df, df_filtered_concat, on=['datetime', 'stid'], how='left')
    # df_final.set_index('datetime', inplace=True)

    df_final = df_filtered_concat
    return df_final



