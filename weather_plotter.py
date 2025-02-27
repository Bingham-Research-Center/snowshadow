from herbie import Herbie
import metpy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


class WeatherPlotter:
    """
    A class for creating weather data visualizations using Herbie, MetPy, and Cartopy.

    This class provides methods to fetch weather data, process datasets,
    and create publication-quality meteorological maps with proper projections
    and geographic features.
    """

    def __init__(self, date_time, model="hrrr", product="sfc", fxx=0):
        """
        Initialize the WeatherPlotter with specified parameters.

        Parameters:
        -----------
        date_time : str
            Date and time string in format "YYYY-MM-DD HH:MM"
        model : str, optional
            Weather model to use (default is "hrrr")
        product : str, optional
            Product type (default is "sfc")
        fxx : int, optional
            Forecast hour (default is 0 for analysis)
        """
        # Initialize properties
        self.date_time = date_time
        self.model = model
        self.product = product
        self.fxx = fxx
        self.herbie = None
        self.dataset = None
        self.cartopy_proj = None

        # Initialize Herbie
        self._initialize_herbie()

    def _initialize_herbie(self):
        """
        Initialize the Herbie object with class parameters.
        Private method used internally.
        """
        self.herbie = Herbie(self.date_time,
                             model=self.model,
                             product=self.product,
                             fxx=self.fxx)

    def fetch_temperature_data(self, variable="TMP:2 m"):
        """
        Fetch temperature data using Herbie and process it.

        Parameters:
        -----------
        variable : str, optional
            Temperature variable to fetch (default is "TMP:2 m")

        Returns:
        --------
        xarray.Dataset
            Processed dataset
        """
        # Extract the xarray dataset
        ds = self.herbie.xarray(variable)

        # If both 'time' and 'valid_time' are present, drop the ambiguous 'time'
        if 'time' in ds and 'valid_time' in ds:
            ds = ds.drop_vars('time')

        # Parse CF metadata and assign proper 1D x/y coordinates
        ds = ds.metpy.parse_cf().metpy.assign_y_x()

        # Store dataset and projection
        self.dataset = ds
        self.cartopy_proj = ds.metpy_crs.item().to_cartopy()

        return ds

    def create_plot(self, figsize=(12, 10)):
        """
        Create a new matplotlib figure and axes with the appropriate projection.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches (default is (12, 10))

        Returns:
        --------
        tuple
            (fig, ax) matplotlib figure and axes objects
        """
        if self.cartopy_proj is None:
            raise ValueError("Dataset not loaded. Call fetch_temperature_data() first.")

        # Create the plot with a single Axes
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={'projection': self.cartopy_proj})

        return fig, ax

    def add_geographic_features(self, ax, coastlines_scale='50m', states_scale='50m'):
        """
        Add geographic features to the map.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to add features to
        coastlines_scale : str, optional
            Scale for coastlines (default is '50m')
        states_scale : str, optional
            Scale for state boundaries (default is '50m')
        """
        # Add geographic context
        ax.coastlines(coastlines_scale, linewidth=0.8)
        ax.add_feature(cfeature.STATES.with_scale(states_scale),
                       linewidth=0.5, edgecolor='black')

    def plot_temperature(self, ax, units='degC', cmap='plasma', levels=30, colorbar_shrink=0.6):
        """
        Plot temperature data on the given axes.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        units : str, optional
            Units to convert temperature to (default is 'degC')
        cmap : str, optional
            Colormap to use (default is 'plasma')
        levels : int, optional
            Number of contour levels (default is 30)
        colorbar_shrink : float, optional
            Shrink factor for colorbar (default is 0.6)

        Returns:
        --------
        matplotlib.contour.QuadContourSet
            The contour plot object
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call fetch_temperature_data() first.")

        # Convert the temperature from original units to specified units
        temp = self.dataset['t2m'].metpy.convert_units(units)

        # Plot the temperature field using the assigned x and y coordinates
        contour = ax.contourf(self.dataset['x'].values,
                              self.dataset['y'].values,
                              temp,
                              levels=levels,
                              cmap=cmap,
                              transform=self.cartopy_proj)

        # Add colorbar
        units_label = '°C' if units == 'degC' else units
        plt.colorbar(contour, ax=ax, shrink=colorbar_shrink).set_label(f'2m Temperature ({units_label})')

        return contour

    def add_title(self, ax):
        """
        Add a title to the plot based on the dataset's valid time.

        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to add the title to
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call fetch_temperature_data() first.")

        ax.set_title(f"{self.model.upper()} 2m Temperature Analysis: {self.dataset.valid_time.values} UTC")

    def plot(self, figsize=(12, 10), units='degC', cmap='plasma', levels=30,
             coastlines_scale='50m', states_scale='50m', colorbar_shrink=0.6):
        """
        Create a complete temperature plot with all features.

        Parameters:
        -----------
        figsize : tuple, optional
            Figure size (width, height) in inches (default is (12, 10))
        units : str, optional
            Units to convert temperature to (default is 'degC')
        cmap : str, optional
            Colormap to use (default is 'plasma')
        levels : int, optional
            Number of contour levels (default is 30)
        coastlines_scale : str, optional
            Scale for coastlines (default is '50m')
        states_scale : str, optional
            Scale for state boundaries (default is '50m')
        colorbar_shrink : float, optional
            Shrink factor for colorbar (default is 0.6)

        Returns:
        --------
        tuple
            (fig, ax) matplotlib figure and axes objects
        """
        # Fetch data if not already fetched
        if self.dataset is None:
            self.fetch_temperature_data()

        # Create plot
        fig, ax = self.create_plot(figsize=figsize)

        # Add features
        self.add_geographic_features(ax, coastlines_scale, states_scale)

        # Plot temperature
        self.plot_temperature(ax, units, cmap, levels, colorbar_shrink)

        # Add title
        self.add_title(ax)

        return fig, ax

    def show(self):
        """
        Display the current plot.
        """
        plt.show()

    def save(self, filename, dpi=300):
        """
        Save the current plot to a file.

        Parameters:
        -----------
        filename : str
            Filename to save to (including extension)
        dpi : int, optional
            Resolution in dots per inch (default is 300)
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')


# Example usage
def main():
    # Create a weather plotter instance for a specific date and time
    plotter = WeatherPlotter("2021-01-25 06:00")

    # Option 1: One-step approach - create a complete plot
    fig, ax = plotter.plot()
    plotter.show()

    # Option 2: Step-by-step approach for more customization
    # plotter.fetch_temperature_data()
    # fig, ax = plotter.create_plot()
    # plotter.add_geographic_features(ax)
    # plotter.plot_temperature(ax, cmap='viridis')  # Use a different colormap
    # plotter.add_title(ax)
    # plotter.show()

    # Save the figure to a file
    # plotter.save("temperature_map.png")


if __name__ == "__main__":
    main()