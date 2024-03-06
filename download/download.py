import synoptic.services as ss

import os

def get_observation_data(radius="UCL21,50", recent=3*60*60):
    df_meta = ss.stations_metadata(radius=radius, recent=recent)
    # Save file

    return df_meta
