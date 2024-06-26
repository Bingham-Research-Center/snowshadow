import pickle
import os

import numpy as np

vrbls = [
        "wind_speed", "wind_direction", "air_temp", "dew_point_temperature",
        "pressure", "snow_depth", "solar_radiation",
        "relative_humidity", "wind_gust", "altimeter", "soil_temp",
        "sea_level_pressure", "snow_accum",
        # "road_temp",
        # "cloud_layer_1_code", "cloud_layer_2_code",
        # "cloud_layer_3_code", "cloud_low_symbol",
        # "cloud_mid_symbol", "cloud_high_symbol",
        # "sonic_wind_direction", "peak_wind_speed",
        # "ceiling",
        # "sonic_wind_speed",
        "soil_temp_ir",
        "snow_smoothed", "snow_accum_manual", "snow_water_equiv",
        # "precipitable_water_vapor", "net_radiation_sw",
        # "sonic_air_temp", "sonic_vertical_vel",
        # "vertical_heat_flux", "outgoing_radiation_sw",
        # "PM_25_concentration",
        "ozone_concentration",
        # "derived_aerosol_boundary_layer_depth",
        # "NOx_concentration", "PM_10_concentration",
        # "visibility_code", "cloud_layer_1", "cloud_layer_2",
        # "cloud_layer_3",
        # "wet_bulb_temperature"
        ]

def save_pickle(data,fpath: str):
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(fpath: str):
    with open(fpath, 'rb') as f:
        return pickle.load(f)


def try_create(fpath):
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        print ("Creating directory at", fpath)
    else:
        print("Directory already exists at", fpath)
    return

def region_lookup(region: str):
    """Look up radius string for a region.

    Format: "stid,radius"
    # TODO - miles? km? nautical miles? statute miles?
    * Radius is in miles
    * stid is the station ID

    Regions: ["uinta_basin", "uinta_mtns", "nslv", "sslv"]
    * uinta_basin samples the E Uinta Basin
    * uinta_mtns samples the Uinta Mountains and W Basin
    * nslv samples the Northern Salt Lake Valley and Wasatch
    * cslv samples the Central Salt Lake Valley and Wasatch
    * sslv samples the Southern Salt Lake Valley and Wasatch

    """
    region_dict = {
        "uinta_basin": "UCL21,50",  # Pelican Lake Agrimet PELU
        # "uinta_mtns": "RCKU1,50",  # Rock Creek
        # "uinta_mtns": "FPLU1,40",  # Five Points Lake (Snotel)
        "uinta_mtns": "MTHU1,45",  # Moon Lake (HADS)
        "nslv": "KOGD,25",  # Ogden-Hinckley Airport (ASOS/AWOS)
        "cslv": "WBB,25",  # SLC WBB (U of U)
        "sslv": "FG015,25",  # Lincoln Point (FGNet)
        }
    return region_dict[region]

wasatch_towns = {
    "Salt Lake City": (40.7587, -111.8762),
    "Ogden": (41.2230, -111.9738),
    "Provo": (40.2338, -111.6585)
}

uinta_towns = {
    "Roosevelt": (40.2994, -109.9888),
    "Vernal": (40.4555, -109.5287),
    "Ouray": (40.0944, -109.6202)
}

stid_latlons = {
    "UCL21": (40.1742, -109.6666),  # Pelican Lake Agrimet PELU
    # "FPLU1": (40.7179, -110.4672),  # Five Points Lake (Snotel)
    # "RCKU1": (40.54875,-110.69292),  # Rock Creek (Snotel)
    "MTHU1": (40.56167, -110.49222),  # Moon Lake (HADS)
    "KOGD": (41.1941, -112.0168),  # Ogden-Hinckley Airport (ASOS/AWOS)
    "FG015": (40.1342, -111.8190),  # Lincoln Point (FGNet)
    "WBB": (40.7662, -111.8476),  # SLC WBB (U of U)
}

def reduce_precision(df_obs, new_dtype=np.float16):
    """Reduce the precision of the float64 columns to requested new_dtype."""
    print("Original file size:", np.sum(df_obs.memory_usage()) / 1E6, "MB")

    if new_dtype == np.float32:
        col_precise = df_obs.select_dtypes(include=[np.float64]).columns
    elif new_dtype == np.float16:
        col_precise = df_obs.select_dtypes(include=[np.float64, np.float32]).columns
    else:
        raise ValueError("new_dtype must be np.float32 or np.float16")

    df_obs[col_precise] = df_obs[col_precise].astype(new_dtype)

    print("New file size:", np.sum(df_obs.memory_usage()) / 1E6, "MB")
    return df_obs
