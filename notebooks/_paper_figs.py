"""Examples for one-shot functions to generate data and images for the paper
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from postprocessing.filter_funcs import filter_data

def plot_two_years(stids, df_2023, df_2024, vrbl, kernel_lookup=None):
    """Compare two years of data

    Better way of doing this -
    * Generalise the years of data frame - lists or subsets?
    * Remove hard-coded figure size, labels, etc
    * Should be in a top-level or personal directory outside of repo.
    """
    def get_kernel(stid, kernel_lookup):
        return kernel_lookup[stid] if kernel_lookup is not None else 1

    fig, axs = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    # Plot for df_2023
    for stid in stids:
        kernel_size = get_kernel(stid, kernel_lookup)
        subset = filter_data(df_2023[df_2023["stid"] == stid], stid, "snow_depth", kernel_size)
        axs[0].plot(subset.index, subset[vrbl], label=stid)

    # Plot for df_2024
    for stid in stids:
        kernel_size = get_kernel(stid, kernel_lookup)
        subset = filter_data(df_2024[df_2024["stid"] == stid], stid, "snow_depth", kernel_size)
        axs[1].plot(subset.index, subset[vrbl], label=stid)

    # Set titles, labels, and customize plots
    axs[0].set_title(f"{vrbl}: Nov 2022 - Apr 2023")
    axs[0].set_xlabel("Date")
    # axs[0].set_ylabel("Snow Depth (mm)")
    # axs[0].set_ylim(0, 750)

    axs[1].set_title(f"{vrbl}: Nov 2023 - Mar 2024")
    axs[1].set_xlabel("Date")
    # axs[1].set_ylim(0, 750)

    # Add legend to the first plot
    axs[0].legend()

    plt.tight_layout()
    plt.show()

def plot_before_after(df, stid, vrbl, filtered_df):
    # Plotting before-and-after effect of the replacement function
    plt.figure(figsize=(12, 6))

    # Original data
    plt.plot(df[df["stid"] == stid].index, df[df["stid"] == stid][vrbl], label='Original', alpha=0.7)

    # Filtered data
    plt.plot(filtered_df.index, filtered_df[vrbl], label='After Removing Max Value', color='red')

    plt.title(f'Snow Depth for {stid} Before and After Removing Maximum Value')
    plt.xlabel('Date')
    plt.ylabel('Snow Depth (mm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_filtered_data(filtered_dfs, top_stations_percentage, region, stid_list=None):
    """Plot the filtered snow depth time series for the top 5 STIDs in the specified region on a single plot.

    Args:
        filtered_dfs (list): A list of DataFrames, each containing the filtered data for a specific STID.
        top_stations_percentage (pd.DataFrame): A DataFrame containing the top STIDs for each region.
        region (str): The region to plot the data for.
    """
    # Create a color palette with 5 different colors
    colors = plt.cm.get_cmap('hsv', 7)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, filtered_df in enumerate(filtered_dfs):
        if stid_list is None:
            stid = top_stations_percentage[top_stations_percentage["region"] == region].iloc[i]["stid"]
        else:
            stid = stid_list[i]
        filtered_df.plot(y="snow_depth", ax=ax, color=colors(i), label=f"STID: {stid}")

    # Set labels and title
    ax.set_ylabel("Snow Depth (mm)")
    ax.set_xlabel("Date")
    plt.title(f"Filtered Snow Depth Time Series for Top 5 STIDs in {region}")

    # Add a legend
    plt.legend()

    plt.tight_layout()
    plt.show()
    return

def filter_and_plot_data(df, top_stations_percentage, region, kernel_size, stid_list=None):
    """Filter snow_depth values for the stids for a specific region and plot the filtered data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        top_stations_percentage (pd.DataFrame): A DataFrame containing the top STIDs for each region.
        region (str): The region to filter and plot the data for.
    """
    if stid_list is None:
        stid_list = top_stations_percentage[top_stations_percentage["region"] == region]["stid"].values
    filtered_dfs = []
    for stid in stid_list:
        filtered_df = filter_data(df, stid, "snow_depth", kernel_size)
        filtered_dfs.append(filtered_df)

    plot_filtered_data(filtered_dfs, top_stations_percentage, region, stid_list)
    return

def compare_region_timeseries(df, top_stations_percentage, regions_to_plot):
    """Plot the filtered snow depth time series for the specified regions on a single plot.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        top_stations_percentage (pd.DataFrame): A DataFrame containing the top STIDs for each region.
        regions_to_plot (list): The regions to plot the data for.
    """
    fig, axes = plt.subplots(1, len(regions_to_plot), figsize=(20, 8))  # One column for each region

    for i, region in enumerate(regions_to_plot):
        stids_region = top_stations_percentage[top_stations_percentage["region"] == region]["stid"].values
        elevations = df[df["stid"].isin(stids_region)]['elevation'].unique()

        # TODO - might be better to manually assign "low/foothills" v "slope/peak" depending on UT region
        quartiles = np.quantile(elevations, [0.25, 0.75])

        legend_info = []  # To store handles, labels, and elevations

        for stid in stids_region:
            df_subset = df[df["stid"] == stid]
            if not df_subset.empty:
                elevation = df_subset['elevation'].iloc[0]
                linestyle = '-' if elevation <= quartiles[0] else ':'
                line, = axes[i].plot(df_subset.index, df_subset["filtered_snow_depth"], linestyle=linestyle, label=f"{stid} ({elevation}m)")
                # Append handle, label, and elevation
                legend_info.append((line, f"{stid} ({int(elevation)}m)", elevation))

        # Sort legend_info by elevation
        legend_info.sort(key=lambda x: x[2])
        # Unpack sorted legend_info to create the legend
        handles, labels, _ = zip(*legend_info)
        axes[i].legend(handles, labels)

        axes[i].set_title(f"Filtered Snow Depth: {region.replace('_', ' ').title()}")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Snow Depth (mm)")

    plt.tight_layout()
    plt.show()

def plot_elevation_stations(df, top_stations_percentage, elevation_categories, regions, elevation_type):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    def get_stids_region(region):
        return top_stations_percentage[top_stations_percentage["region"] == region]["stid"].values

    def get_combined_stids_region(region):
        assert len(region) > 1
        stids = []
        for r in region:
            stids.extend(get_stids_region(r))
        return stids

    for i, region in enumerate(regions):
        if isinstance(region, tuple):
            stids_region = get_combined_stids_region(region)
            region_str = "uinta_merge"
        else:
            stids_region = get_stids_region(region)
            region_str = region.replace('_', ' ').title()
        stids_filtered = [stid for stid in stids_region if elevation_categories[stid] == elevation_type]

        for stid in stids_filtered:
            df_stid = df[df["stid"] == stid]
            if not df_stid.empty:
                altitude = df_stid['elevation'].iloc[0]
                linestyle = '--' if df_stid['region'].iloc[0] == 'uinta_mtns' else '-'
                df_stid.plot(y="filtered_snow_depth", ax=axes[i//2, i%2], label=f"{stid} ({altitude}m)", linestyle=linestyle)

        # TODO - find a human-readable way to loop over axes without floor divide etc
        axes[i//2, i%2].set_title(f"{elevation_type.capitalize()}-Elevation Stations in {region_str}")
        axes[i//2, i%2].set_xlabel("Date")
        axes[i//2, i%2].set_ylabel("Snow Depth (mm)")

        # Create the legend for each subplot
        axes[i//2, i%2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8, framealpha=0.5)

        # Change the font size for the x-axis ticks
        axes[i//2, i%2].tick_params(axis='x', labelsize=8)

    # Adjust the layout to give more space for the legends
    plt.subplots_adjust(bottom=0.2)

    # Double the typical number of x-axis ticks for the date
    for ax in axes.flat:
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))

    plt.tight_layout()
    plt.show()