"""Acquire observation data from synopticPy or disk

More detail here.

John Lawson and Michael Davies, USU BRC, March 2024
"""

import os
import datetime

import numpy as np
import pandas as pd

import synoptic.services as ss
from utils.utils import vrbls, try_create, save_pickle, load_pickle

### FUNCTIONS ###

def region_lookup(region: str):
    """Look up radius string for a region.
    """
    region_dict = {
        "basin": "UCL21,50",
        # TODO: decide on stid and radius for these new regions
        # "wasatch": "KSLC,25",
        # "uinta": "UCL21,50",
        }
    return region_dict[region]

def get_observation_data(vrbls: (list, tuple), data_root, radius: str = "UCL21,50", recent=3*60*60):
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
    # TODO: make this dynamic because area of interest may change

    data_fname = "basin_obs.h5"
    metadata_fname = "basin_ob_metadata.h5"

    df_obs_fpath = os.path.join(data_root, data_fname)
    df_meta_fpath = os.path.join(data_root, metadata_fname)

    if os.path.exists(df_obs_fpath) and os.path.exists(df_meta_fpath):
        df_obs = pd.read_hdf(df_obs_fpath, key='df_obs')
        # df_meta = pd.read_hdf(df_meta_fpath, key='df_meta')
        df_meta = load_pickle(df_meta_fpath)
    else:
        df_meta, df_obs = download_obs_data(vrbls, radius, recent, df_obs_fpath)
        return df_meta, df_obs

def download_obs_data(vrbls, radius, recent, df_fpath):
    # TODO: dates in arguments above
    df_list = list()
    df_meta = ss.stations_metadata(radius=radius, recent=recent)
    stids = get_stids_from_metadata(df_meta)

    for stid in stids[:4]:
        try:
            # TODO: not hard-code the start and end dates
            stid_df = ss.stations_timeseries(stid=stid,start=datetime.datetime(2022,11,17,0,0,0),
                                            end=datetime.datetime(2023,11,17,0,0,0),
                                            vrbls=vrbls, verbose=True)
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

    df_obs = pd.concat(df_list, axis=0, ignore_index=False)

    print("Original file size:",np.sum(df_obs.memory_usage())/1E6,"MB")
    # Find columns using float64
    col64 = [df_obs.columns[i] for i in range(len(list(df_obs.columns))) if (df_obs.dtypes.iloc[i] == np.float64)]
    change_dict = {c:np.float32 for c in col64}
    df_obs = df_obs.astype(change_dict)
    print("New (float32) file size:",np.sum(df_obs.memory_usage())/1E6,"MB")

    # Save to file - hdf5 (.h5)
    df_obs.to_hdf(df_fpath, key='df_obs', mode='w')
    # df_meta.to_hdf(df_fpath, key='df_meta', mode='w')
    save_pickle(df_meta,df_fpath.replace("obs","meta"))

    return df_meta, df_obs

def get_stids_from_metadata(metadata,):
    """Get station IDs from metadata

    Args:
        metadata (pd.DataFrame): metadata of stations

    Returns:
        list: station IDs

    """
    return list(metadata.columns.unique())
