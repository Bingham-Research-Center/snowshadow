"""Create visualisations of results of comparing snowfall in the Uinta Basin versus Wasatch Front"""

import os

from utils.utils import try_create, vrbls

from download.download import get_observation_data

if __name__ == "__main__":
    # Load data
    data_root = "./data"
    try_create(data_root)

    # TODO: define regions with stid and radius, then add to dictionary
    # regions = ["basin", "wasatch"]

    # Get observation data
    # TODO - fix the warning in synopticPy
    df_meta, df_obs = get_observation_data(vrbls, data_root)
    stids = list(df_meta.columns.unique())
    pass


    # Plot
    # import from own code in plotting folder

    # Save

    # Data analysis