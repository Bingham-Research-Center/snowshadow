"""Create visualisations of results of comparing snowfall in the Uinta Basin versus Wasatch Front

TODO:
* An argparse interface to allow for different settings from command line
* A logger to record the progress of the program
* GUI or web interface to allow for user to interact with code

"""

import os
import datetime

from utils.utils import try_create, vrbls, region_lookup

from download.download import get_observation_data
from plotting.birdseye import Birdseye
from postprocessing.datadive import DataDive

if __name__ == "__main__":
    ### SETTINGS ###
    # Plotting useful maps/images
    plot_regions = False
    plot_stations = False

    force_do = False
    redo_pp = False

    # Load data
    data_root = "./data"
    try_create(data_root)

    # Visualisations
    figure_root = "./figures"
    try_create(figure_root)

    # TODO: might need to chop out wyoming from obs
    # start_date = datetime.datetime(2012, 11, 1, 0, 0, 0)
    # If this date range changes, the data will need to be re-downloaded and the .h5 files overwritten
    start_date = datetime.datetime(2022, 11, 1, 0, 0, 0)
    end_date = datetime.datetime(2024, 3, 25, 0, 0, 0)

    # Choose regions loaded - set in utils
    regions = ["uinta_basin", "uinta_mtns", "nslv", "cslv", "sslv"]

    if plot_regions:
        # TODO: Create a map of Utah and the four regions (radii) labelled with some key towns
        fname = "regions.png"
        region_map = Birdseye(os.path.join(figure_root, fname), figsize=(8,8))
        region_map.plot_regions(regions)
        region_map.save()
        region_map.fig.show()

    # Get observation data
    # TODO - fix the warning in synopticPy
    df_obs, df_meta = get_observation_data(vrbls, data_root, start_date, end_date, regions, force_do=force_do,
                                                redo_pp=redo_pp)
    # TODO: make sure only vrbls are in dataframe, not just ALL the variables
    print("Observation data has been loaded.")

    # TODO: ensure h5 has no pickled-object columns
    # e.g., convert stid strings to integers and create .txt file with the mapping

    if plot_stations:
        all_station = Birdseye(os.path.join(figure_root, "all_stations.png"), figsize=(8,8))
        all_station.plot_all_stations(df_obs, label_names=False)
        # Do save -> show to do tight_layout - can also do manually from all_station.fig.tight_layout()
        # close_after keeps it open for displaying on PyCharm - can turn off for production
        all_station.save(close_after=False)
        # Can remove for production:
        all_station.fig.show()

    # about 45 mins to download five regions in serial on wifi at home

    # Toggling the next two lines turns on/off saving the processed dataframe to disc.
    # This can be loaded as the "real thing" later.
    if redo_pp:
        dd = DataDive(df_obs, df_meta, process_df=True, save_new_df='./data/df_obs_pp.h5')
    else:
        dd = DataDive(df_obs, df_meta, process_df=False)
    pass
