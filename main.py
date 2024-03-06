"""Create visualisations of results of comparing snowfall in the Uinta Basin versus Wasatch Front"""

import os

from download.download import get_observation_data

if __name__ == "__main__":
    # Load data
    data_root = "./data"
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    obs_fpath = os.path.join(".", data_root, "basin_obs.h5")
    metadata_fpath = os.path.join(".", data_root, "basin_ob_metadata.h5")

    # Get observation data
    # TODO - fix the warning in synopticPy
    df_meta = get_observation_data()
    stids = list(df_meta.columns.unique())
    pass


    # Plot
    # import from own code in plotting folder
