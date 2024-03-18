"""Create visualisations of results of comparing snowfall in the Uinta Basin versus Wasatch Front"""

import os
import datetime

from utils.utils import try_create, vrbls, region_lookup

from download.download import get_observation_data
from plotting.birdseye import Birdseye



if __name__ == "__main__":
    ### SETTINGS ###
    plot_regions = False
    force_do = True

    # Load data
    data_root = "./data"
    try_create(data_root)

    # Visualisations
    figure_root = "./figures"
    try_create(figure_root)

    # TODO: might need to chop out wyoming from obs
    # start_date = datetime.datetime(2012, 11, 1, 0, 0, 0)
    start_date = datetime.datetime(2024, 2, 1, 0, 0, 0)
    end_date = datetime.datetime(2024, 3, 1, 0, 0, 0)

    regions = ["uinta_basin", "uinta_mtns", "nslv", "sslv"]

    if plot_regions:
        # TODO: Create a map of Utah and the four regions (radii) labelled with some key towns
        fname = "regions.png"
        region_map = Birdseye(os.path.join(figure_root, fname), figsize=(8,8))
        region_map.plot_regions(regions)
        region_map.save()
        region_map.fig.show()

    # Get observation data
    # TODO - fix the warning in synopticPy
    df_meta, df_obs = get_observation_data(vrbls, data_root, start_date, end_date, regions, force_do=force_do)
    stids = list(df_meta.columns.unique())
    pass

    # TODO: start playing with the data once we download all four regions
    pass