import pickle

vars = ["wind_speed", "wind_direction", "air_temp", "dew_point_temperature",
        "pressure", "snow_depth", "solar_radiation",
        "relative_humidity", "wind_gust", "altimeter", "soil_temp",
        "sea_level_pressure", "snow_accum", "road_temp",
        "cloud_layer_1_code", "cloud_layer_2_code",
        "cloud_layer_3_code", "cloud_low_symbol",
        "cloud_mid_symbol", "cloud_high_symbol",
        "sonic_wind_direction", "peak_wind_speed",
        "ceiling", "sonic_wind_speed", "soil_temp_ir",
        "snow_smoothed", "snow_accum_manual", "snow_water_equiv",
        "precipitable_water_vapor", "net_radiation_sw",
        "sonic_air_temp", "sonic_vertical_vel",
        "vertical_heat_flux", "outgoing_radiation_sw",
        "PM_25_concentration", "ozone_concentration",
        "derived_aerosol_boundary_layer_depth",
        "NOx_concentration", "PM_10_concentration",
        "visibility_code", "cloud_layer_1", "cloud_layer_2",
        "cloud_layer_3", "wet_bulb_temperature"
        ]

def save_pickle(data,fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        return pickle.load(f)

