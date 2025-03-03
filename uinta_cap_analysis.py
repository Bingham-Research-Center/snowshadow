import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# MetPy tools
from metpy import calc as mpcalc
from metpy.interpolate import cross_section
from metpy.units import units

# Herbie for data retrieval
from herbie import Herbie

def retrieve_hrrr_data(date_time="2024-03-03 04:00", fxx=0):
    """Retrieve HRRR data with Herbie."""
    H = Herbie(date_time, model="hrrr", product="prs", fxx=fxx, priority="aws")
    ds = H.xarray(":(TMP|RH|VVEL|HGT):(850|900|950) mb")
    ds = ds.metpy.assign_crs({
        "grid_mapping_name": "lambert_conformal_conic",
        "standard_parallel": (38.5, 38.5),
        "latitude_of_projection_origin": 38.5,
        "longitude_of_central_meridian": 262.5,
    })
    return ds

def interpolate_cross_section(ds, start=(40.299, -109.99), end=(40.455, -109.53), steps=50):
    """Get a cross-section from start to end."""
    ds = ds.metpy.parse_cf().metpy.assign_y_x()
    cs = cross_section(ds, start, end, steps=steps, interp_type="linear")
    return cs.set_coords(["latitude", "longitude"])

def detect_inversion_layers(cross):
    """Compute potential temperature and mark where dθ/dz >= 0.05 K/m."""
    cross["TMP_K"] = cross["t"].metpy.convert_units("K")
    pressure = cross["isobaricInhPa"] * 100 * units.Pa
    cross["theta"] = mpcalc.potential_temperature(pressure, cross["TMP_K"])

    # Convert geopotential to geometric height in meters: height = gh / g
    height = cross["gh"] / 9.80665

    # We'll add height as a coordinate for easier plotting
    cross = cross.assign_coords(height=height)

    # We'll also assume that the horizontal dimension is "index"
    # (added by MetPy's cross_section function).
    # Compute the vertical gradient for each horizontal index
    # using np.gradient on the "theta" and "height" profiles.
    dtheta_dz_values = np.empty_like(cross["theta"].values)
    for i in range(cross.sizes["index"]):
        theta_profile = cross["theta"].isel(index=i).values
        height_profile = cross["height"].isel(index=i).values
        grad = np.gradient(theta_profile, height_profile)
        dtheta_dz_values[:, i] = grad

    # Build an xarray DataArray for the gradient
    dtheta_dz = xr.DataArray(
        dtheta_dz_values,
        coords=cross["theta"].coords,
        dims=cross["theta"].dims
    )

    # Inversion if dθ/dz >= 0.05 K/m
    cross["inversion_mask"] = dtheta_dz >= 0.05
    return cross

def visualize_cross_section(cross, output_file="cross_section.png"):
    """Plot a vertical cross-section of potential temperature vs. height."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot potential temperature (theta) as a contour fill
    # using "index" for the horizontal axis and "height" for the vertical axis.
    cross["theta"].plot.contourf(
        x="index",
        y="height",
        cmap="coolwarm",
        levels=20,
        add_colorbar=True,
        ax=ax
    )

    # Overlay inversion mask as a contour
    # We can do this by converting True/False to numeric (1/0).
    cross["inversion_mask"].where(cross["inversion_mask"]).plot.contour(
        x="index",
        y="height",
        colors="k",
        linewidths=2,
        add_colorbar=False,
        ax=ax
    )

    ax.set_title("Potential Temperature Cross-Section with Inversion Layers")
    ax.set_xlabel("Cross-Section Index (along transect)")
    ax.set_ylabel("Height (m)")
    plt.savefig(output_file, dpi=300)
    plt.show()

def main():
    ds = retrieve_hrrr_data()
    cross = interpolate_cross_section(ds)
    cross = detect_inversion_layers(cross)
    visualize_cross_section(cross)

if __name__ == "__main__":
    main()
