"""Create visualisations of results of comparing snowfall in the Uinta Basin versus Wasatch Front"""

import os
import datetime

from utils.utils import try_create, vrbls

from download.download import get_observation_data
from plotting.birdseye import Birdseye

if __name__ == "__main__":
    # Load data
    data_root = "./data"
    try_create(data_root)

    # Visualisations
    figure_root = "./figures"
    try_create(figure_root)

    start_date = datetime.datetime(2023, 11, 1, 0, 0, 0)
    end_date = datetime.datetime(2024, 3, 1, 0, 0, 0)

    # TODO: define regions with stid and radius, then add to dictionary
    regions = ["uinta_basin", "uinta_mtns", "nslv", "sslv"]

    # TODO: Create a map of Utah and the four regions (radii) labelled with some key towns
    fname = "regions.png"
    region_map = Birdseye(os.path.join(figure_root, fname), figsize=(8,8))
    region_map.plot_regions(regions)
    region_map.save()
    region_map.fig.show()

    # Get observation data
    # TODO - fix the warning in synopticPy
    df_meta, df_obs = get_observation_data(vrbls, data_root, start_date, end_date)
    stids = list(df_meta.columns.unique())
    pass

    # TODO: start playing with the data once we download all four regions
    pass