import os
import datetime

import numpy as np
import pandas as pd

import synoptic.services as ss
from utils.utils import vars

def get_observation_data(vars, radius: str = "UCL21,50", recent=3*60*60):
    """Get observation data from synopticPy

    Subtasks:
    * Get metadata
    * Get observation data and their variables (vars)

    Args:
        radius (str): radius from the station ID of interest (distance in ??? TODO)
        recent (int): time in seconds to get the most recent data

    Returns:
        pd.DataFrame: observation data

    """
    df_list = list()
    df_meta = ss.stations_metadata(radius=radius, recent=recent)
    stids = get_stids_from_metadata(df_meta)

    for stid in stids[:4]:
        try:
            stid_df = ss.stations_timeseries(stid=stid,start=datetime.datetime(2022,11,17,0,0,0),
                                            end=datetime.datetime(2023,11,17,0,0,0),
                                            vars=vars, verbose=True)
        except AssertionError:
            print("Skipping",stid)
            continue
        else:
            print("Data found for",stid)
        finally:
            print("Done with",stid)

        # At this point we have the data for the station


        # Pull out lat, lon, elevation from metadata
        # TODO: consider whether this is wasteful on ROM/RAM - keep in df_meta?
        stid_lat = df_meta[stid].loc["latitude"]
        stid_lon = df_meta[stid].loc["longitude"]
        # TODO: use the pint package to enforce units
        elev = df_meta[stid].loc["ELEVATION"]*0.304

        stid_df = stid_df.assign(stid=stid, elevation=elev, latitude=stid_lat,longitude=stid_lon)

        df_list.append(stid_df)

        # Debugging and sanity check
        # pd.to_datetime(stid_df.index.strftime('%Y-%m-%dT%H:%M:%SZ'))

    # Now combine all stations into one df
    # df = 0

    df = pd.concat(df_list, axis=0, ignore_index=False)
    pass

    print("Original file size:",np.sum(df.memory_usage())/1E6,"MB")
    # Find columns using float64
    col64 = [df.columns[i] for i in range(len(list(df.columns))) if (df.dtypes.iloc[i] == np.float64)]
    change_dict = {c:np.float32 for c in col64}
    df = df.astype(change_dict)
    print("New (float32) file size:",np.sum(df.memory_usage())/1E6,"MB")
    # df.to_hdf(obs_fpath, key='df', mode='w')

    return df

def get_stids_from_metadata(metadata,):
    """Get station IDs from metadata

    Args:
        metadata (pd.DataFrame): metadata of stations

    Returns:
        list: station IDs

    """
    return list(metadata.columns.unique())

if __name__ == "__main__":
    start_date = datetime.datetime(2023,11,1,0,0,0)
    end_date = datetime.datetime(2024,1,1,0,0,0)

    df = get_observation_data(vars)
    pass
