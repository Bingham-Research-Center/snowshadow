import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import metpy
from metpy.plots import USCOUNTIES

from utils.utils import region_lookup, stid_latlons

class Birdseye:
    """Top-down latitude--longitude cross-section plots.

    TODO:
    * Create figure showing the regions of interest (e.g., Uinta Basin, SLV) on map of Utah
    """
    def __init__(self, fpath, fig=None, ax=None, ncols=1, nrows=1, figsize=(8, 8), dpi=300):
        """Set up the figure and axes for the birdseye plot.
        """
        self.fpath = fpath
        if fig is None:
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols, subplot_kw={'projection': ccrs.PlateCarree()},
                                    figsize=figsize, dpi=dpi)
        self.fig = fig
        self.ax = ax

    def plot_regions(self,regions):
        """Plot map of Utah with each region labelled.

        Need a dictionary of region name (key) to string with station names and the radius (value)

        """
        # Set extent of basemap figure to northern 2/3 of Utah
        self.ax.set_extent([-113, -108.5, 38.5, 42.2], ccrs.PlateCarree())
        self.add_features_to_basemap()

        for region in regions:
            stid, radius = region_lookup(region).split(",")
            radius_km = float(radius) * 1.60934  # Assuming the radius was in miles

            stid_lat, stid_lon = stid_latlons[stid]

            # Approximate conversion from km to degrees (latitude)
            # This is a very rough approximation, assuming 1 degree ≈ 111 km
            radius_deg = radius_km / 111

            # Plot a half-transparent blue circle around the station
            self.ax.add_patch(plt.Circle((stid_lon, stid_lat), radius_deg,
                                         color='b', alpha=0.3, transform=ccrs.Geodetic()))
            self.ax.text(stid_lon, stid_lat, region, transform=ccrs.PlateCarree())

        return

    def add_features_to_basemap(self):
        """Add features to the basemap from Cartopy, e.g., borders, rivers, countries, etc.
        """
        self.ax.coastlines()
        self.ax.add_feature(cfeature.BORDERS, edgecolor="darkgray", linewidth=1)
        self.ax.add_feature(cfeature.RIVERS)
        self.ax.add_feature(cfeature.LAKES, edgecolor='b')
        self.ax.add_feature(cfeature.STATES, linestyle=':')
        self.ax.add_feature(cfeature.OCEAN, color='lightblue')

        # US counties
        self.ax.add_feature(USCOUNTIES.with_scale('5m'), edgecolor='gray', zorder=2, linestyle=':', linewidth=0.5)
        return

    def save(self, tight_layout=True):
        """Save the figure to file.
        """
        if tight_layout:
            self.fig.tight_layout()
        self.fig.savefig(self.fpath)
        return