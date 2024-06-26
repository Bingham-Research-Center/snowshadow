"""Acquire observation data from synopticPy or disk

More detail here.

John Lawson and Michael Davies, USU BRC, March 2024
"""

import os
import datetime

import numpy as np
import pandas as pd

import synoptic.services as ss
from utils.utils import vrbls, try_create, save_pickle, load_pickle, region_lookup, reduce_precision

### FUNCTIONS ###


def get_observation_data(vrbls: (list, tuple), data_root, start_date, end_date, regions,
                            radius: str = "UCL21,50", recent=3*60*60, force_do=False, redo_pp=False):
    """Get observation data from synopticPy

    Subtasks:
    * Get metadata
    * Get observation data and their variables (vrbls)

    Args:
        vrbls (list, tuple): list of variables to get
        radius (str): radius from the station ID of interest (distance in ??? TODO)
        recent (int): time in seconds to get the most recent data

    Returns:
        Two pandas DataFrames: metadata and observation data

    TODO: change radius to a dictionary of region:radius
    """
    # Two files - observation and metadata

    metadata_fname = "df_metadata.h5"
    if redo_pp:
        # Load raw obs
        data_fname = "df_obs.h5"
    else:
        # Load post-processed obs
        # TODO - not be hard-coded. Better way to do filtering and subsetting of raw data before this
        data_fname = "df_obs_pp_2023.h5"

    df_obs_fpath = os.path.join(data_root, data_fname)
    df_meta_fpath = os.path.join(data_root, metadata_fname)

    if not (os.path.exists(df_obs_fpath) and os.path.exists(df_meta_fpath)) or force_do:
        df_obs, df_meta = concatenate_regions(vrbls, regions, start_date, end_date, recent=recent)
        df_obs.to_hdf(df_obs_fpath, key='df_obs', mode='w')
        save_pickle(df_meta, df_obs_fpath.replace("obs", "metadata"))
    else:
        df_meta = load_pickle(df_meta_fpath)
        df_obs = pd.read_hdf(df_obs_fpath, key='df_obs')
        # df_meta = pd.read_hdf(df_meta_fpath, key='df_meta')
    return df_obs, df_meta


def concatenate_regions(vrbls, regions, start_date, end_date, recent=3 * 60 * 60):
    df_obs_list = []
    df_meta_list = []
    for region in regions:
        radius = region_lookup(region)
        df_obs, df_meta = download_obs_data(vrbls, radius, recent, start_date, end_date)

        # Assign 'region' column to the observation dataframe
        df_obs = df_obs.assign(region=region)

        df_obs_list.append(df_obs)
        df_meta_list.append(df_meta)

    # Concatenate all observation dataframes row-wise
    df_obs_combined = pd.concat(df_obs_list, axis=0)
    pass
    # Concatenate all metadata dataframes row-wise
    # If concatenating column-wise was intentional and each df_meta represents a unique set of columns,
    # consider verifying this logic aligns with your data structure and needs.

    # TODO: check if this is the right way to concatenate metadata : axis 1 or 0
    df_meta_combined = pd.concat(df_meta_list, axis=1)  # Changed from axis=1 to axis=0 for row-wise concatenation

    return df_obs_combined, df_meta_combined

def concatenate_metadata_dfs(region_list, recent=3*60*60):
    df_meta_list = []
    for region in region_list:
        radius = region_lookup(region)
        df_meta = ss.stations_metadata(radius=radius, recent=recent)
        df_meta_list.append(df_meta)
    df_meta_combined = pd.concat(df_meta_list, axis=1)  # Changed from axis=1 to axis=0 for row-wise concatenation
    return df_meta_combined

def download_obs_data(vrbls, radius, recent, start_date, end_date):
    df_list = list()
    df_meta = ss.stations_metadata(radius=radius, recent=recent)
    stids = get_stids_from_metadata(df_meta)

    for stid in stids:
        try:
            stid_df = ss.stations_timeseries(stid=stid,start=start_date,end=end_date,
                                                vars=vrbls, verbose=True,
                                                rename_set_1=False, rename_value_1=False)
        except AssertionError:
            print("Skipping",stid)
            continue
        else:
            print("Data found for",stid)
        finally:
            print("Done with",stid)

        # Pull out lat, lon, elevation from metadata
        # TODO: consider whether this is wasteful on ROM/RAM - keep in df_meta?
        stid_lat = df_meta[stid].loc["latitude"]
        stid_lon = df_meta[stid].loc["longitude"]
        # TODO: use the pint package to enforce units
        elev = df_meta[stid].loc["ELEVATION"]*0.304

        stid_df = stid_df.assign(stid=stid, elevation=elev, latitude=stid_lat,longitude=stid_lon)
        df_list.append(stid_df)
        del stid_df

    df_obs = pd.concat(df_list, axis=0, ignore_index=False)
    df_obs = reduce_precision(df_obs, new_dtype=np.float16)
    pass

    return df_obs, df_meta


def get_stids_from_metadata(metadata,):
    """Get station IDs from metadata

    Args:
        metadata (pd.DataFrame): metadata of stations

    Returns:
        list: station IDs

    """
    return list(metadata.columns.unique())
