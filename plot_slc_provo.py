from weather_plotter import WeatherPlotter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np


def plot_slc_provo_valley():
    """
    Creates a weather visualization for the Salt Lake City and Provo Valley region
    for February 26, 2023 using the WeatherPlotter class.

    This region includes the Wasatch Front - the densely populated area running north-south
    along the western edge of the Wasatch Mountains.
    """
    # Create a WeatherPlotter instance for February 26, 2023
    # Using 18:00 UTC (noon local time in Utah, which is UTC-7 in winter)
    date_time = "2025-02-26 18:00"
    plotter = WeatherPlotter(date_time)

    # Fetch the 2m temperature data
    print("Fetching temperature data via Herbie...")
    plotter.fetch_temperature_data()

    # Create a plot with a specific figure size
    fig, ax = plotter.create_plot(figsize=(14, 12))

    # Set the map extent to focus on the Salt Lake City and Provo Valley region
    # These coordinates define the [west, east, south, north] boundaries
    # This covers from Salt Lake City to Provo and includes the Wasatch Mountains
    ax.set_extent([-112.5, -111.0, 39.9, 41.0], crs=ccrs.PlateCarree())

    # Add standard geographic features
    print("Adding geographic features...")
    plotter.add_geographic_features(ax, coastlines_scale='50m', states_scale='50m')

    # Add county boundaries for better regional context
    try:
        counties = cfeature.NaturalEarthFeature(
            'cultural', 'admin_2_counties', '10m',
            facecolor='none', edgecolor='gray', linewidth=0.5)
        ax.add_feature(counties)
        print("Added county boundaries")
    except Exception as e:
        print(f"County boundaries feature not available: {e}")

    # Add water features for the Great Salt Lake and Utah Lake
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.7, edgecolor='blue', alpha=0.7)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), edgecolor='blue', facecolor='lightblue', alpha=0.5)

    # Try to add shaded relief for topography to show the Wasatch Mountains
    try:
        import cartopy.io.img_tiles as cimgt
        terrain_tiles = cimgt.Stamen('terrain-background')
        ax.add_image(terrain_tiles, 8, alpha=0.3)
        print("Added topographic background")
    except Exception as e:
        print(f"Could not add topography: {e}")

    # Plot temperature with a colormap suitable for temperature visualization
    print("Plotting temperature data...")
    contour = plotter.plot_temperature(ax, cmap='RdBu_r', levels=20)

    # Add major cities in the region for reference points
    cities = {
        'Salt Lake City': (-111.89, 40.76),
        'Provo': (-111.66, 40.23),
        'Ogden': (-111.97, 41.22),
        'Lehi': (-111.85, 40.39),
        'Orem': (-111.69, 40.30),
        'West Valley City': (-112.00, 40.69),
        'Sandy': (-111.88, 40.57),
        'Draper': (-111.86, 40.52),
        'Herriman': (-112.03, 40.51),
        'Park City': (-111.50, 40.65)
    }

    print("Adding city markers...")
    for city, (lon, lat) in cities.items():
        # Check if the city coordinates are within our map extent
        west, east, south, north = ax.get_extent(ccrs.PlateCarree())
        if west <= lon <= east and south <= lat <= north:
            ax.plot(lon, lat, 'ko', transform=ccrs.PlateCarree(), markersize=5)
            # Adjust text position based on city location to avoid overlaps
            if city == 'Park City':  # East of SLC
                ax.text(lon + 0.03, lat, city, transform=ccrs.PlateCarree(),
                        fontsize=9, horizontalalignment='left', fontweight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])
            elif city in ['Salt Lake City', 'West Valley City']:  # Northern cities
                ax.text(lon, lat - 0.03, city, transform=ccrs.PlateCarree(),
                        fontsize=9, horizontalalignment='center', verticalalignment='top', fontweight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])
            elif city in ['Provo', 'Orem']:  # Southern cities
                ax.text(lon, lat + 0.03, city, transform=ccrs.PlateCarree(),
                        fontsize=9, horizontalalignment='center', verticalalignment='bottom', fontweight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])
            else:
                ax.text(lon + 0.03, lat, city, transform=ccrs.PlateCarree(),
                        fontsize=9, horizontalalignment='left', fontweight='bold',
                        path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])

    # Highlight the Wasatch Front corridor
    try:
        # Simple line to highlight the Wasatch Front
        wasatch_lons = np.array([-111.89, -111.84, -111.77, -111.69, -111.66])
        wasatch_lats = np.array([40.76, 40.57, 40.39, 40.30, 40.23])
        ax.plot(wasatch_lons, wasatch_lats, 'k--', linewidth=1.5, alpha=0.6,
                transform=ccrs.PlateCarree(), label='Wasatch Front')
        print("Added Wasatch Front reference line")
    except Exception as e:
        print(f"Could not add Wasatch Front reference: {e}")

    # Add a custom title specifically for the region
    ax.set_title(f"Salt Lake City & Provo Valley 2m Temperature: {plotter.dataset.valid_time.values} UTC",
                 fontsize=16)

    # Add a grid for reference
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')

    # Add a legend if we have the Wasatch Front line
    if 'wasatch_lons' in locals():
        ax.legend(loc='lower right')

    # Save the figure
    output_file = "slc_provo_temperature_20230226.png"
    print(f"Saving visualization to {output_file}...")
    plotter.save(output_file, dpi=300)

    # Show the plot
    print("Displaying visualization...")
    plotter.show()

    return fig, ax


if __name__ == "__main__":
    print("Starting Salt Lake City and Provo Valley weather visualization...")
    fig, ax = plot_slc_provo_valley()
    print("Visualization complete.")