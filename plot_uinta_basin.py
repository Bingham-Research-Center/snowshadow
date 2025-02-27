# Import the WeatherPlotter class
from weather_plotter import WeatherPlotter  # Assumes the class is saved in weather_plotter.py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_uinta_basin():
    """
    Creates a weather visualization for the Uinta Basin, Utah area
    for February 26, 2023 using the WeatherPlotter class.
    """
    # Create a WeatherPlotter instance for February 26, 2023
    # Using 18:00 UTC (noon local time in Utah, which is UTC-7 in winter)
    date_time = "2025-02-26 18:00"
    plotter = WeatherPlotter(date_time)

    # Fetch the 2m temperature data
    print("Fetching temperature data via Herbie...")
    plotter.fetch_temperature_data()

    # Create a plot with a specific figure size
    fig, ax = plotter.create_plot(figsize=(14, 10))

    # Set the map extent to focus on the Uinta Basin, Utah
    # These coordinates define the [west, east, south, north] boundaries
    ax.set_extent([-111.0, -109.0, 39.5, 41.0], crs=ccrs.PlateCarree())

    # Add standard geographic features
    print("Adding geographic features...")
    plotter.add_geographic_features(ax, coastlines_scale='50m', states_scale='50m')

    # Try to add county boundaries for better regional context
    try:
        counties = cfeature.NaturalEarthFeature(
            'cultural', 'admin_2_counties', '10m',
            facecolor='none', edgecolor='gray', linewidth=0.5)
        ax.add_feature(counties)
    except Exception as e:
        print(f"County boundaries feature not available: {e}")

    # Add water features for better geographical context
    ax.add_feature(cfeature.RIVERS.with_scale('10m'), linewidth=0.7, edgecolor='blue', alpha=0.7)
    ax.add_feature(cfeature.LAKES.with_scale('10m'), edgecolor='blue', facecolor='lightblue', alpha=0.5)

    # Plot temperature with a colormap suitable for temperature visualization
    # RdBu_r is a good choice as it shows cold (blue) to hot (red)
    print("Plotting temperature data...")
    plotter.plot_temperature(ax, cmap='RdBu_r', levels=20)

    # Add major cities in the Uinta Basin region for reference points
    cities = {
        'Vernal': (-109.528, 40.455),
        'Duchesne': (-110.399, 40.163),
        'Roosevelt': (-109.989, 40.299),
        'Price': (-110.811, 39.600)
    }

    print("Adding city markers...")
    for city, (lon, lat) in cities.items():
        ax.plot(lon, lat, 'ko', transform=ccrs.PlateCarree(), markersize=5)
        ax.text(lon + 0.03, lat, city, transform=ccrs.PlateCarree(),
                fontsize=10, horizontalalignment='left', fontweight='bold',
                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='white')])

    # Add a custom title specifically for the Uinta Basin
    ax.set_title(f"Uinta Basin 2m Temperature: {plotter.dataset.valid_time.values} UTC", fontsize=16)

    # Add a grid for reference
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle=':')

    # Save the figure
    output_file = "uinta_basin_temperature_20230226.png"
    print(f"Saving visualization to {output_file}...")
    plotter.save(output_file, dpi=300)

    # Show the plot
    print("Displaying visualization...")
    plotter.show()

    return fig, ax


if __name__ == "__main__":
    print("Starting Uinta Basin weather visualization...")
    fig, ax = plot_uinta_basin()
    print("Visualization complete.")