### Snow Shadow

Michael Davies and John Lawson, USU BRC Uintah Basin 2024

This package downloads weather-observation data in Utah and compares snowfall amounts between different areas.
We want to measure the ratio of snowfall between Salt Lake City, Vernal, and Uinta Mountains/Ashley Forest.
Predicting snowfall is very important for the Basin as it is a prerequisite for winter-ozone events.
Our hypothesis is that Vernal receives much less snow than the nearby peaks and Salt Lake City.
We want to look at time series of major snowstorms moving through the state for different sites.
Future work will move into predicting ozone from large-scale weather data but local-scale AI logic.
We will use the observation database methods and develop download methods for forecast data (GEFS, RRFS, HRRR)

Science questions & TODOs:
* How does snowfall in Vernal compare to the Uinta Mountains and Salt Lake City?
* Is there a simple relationship between altitude and snowfall for the Basin?
* What causes this snow shadow effect? Is it a well documented phenomenon? What about for the Basin itself?
The issue of snow-water equivalent - how does this compare between sites and events?
* What is the variation of snow depth between sites on the Basin floor?
* How sensitive is ozone production to SWE? Drier snow will be shallower; how else would it affect cold-pool formation?
* What can we learn about ozone from the snow shadow results, if anything? 

To Dos:
* Create project structure from spaghetti code.
* Download dataset of weather observations for Uinta Basin and Salt Lake City area.
* Find the big snowfalls in 2022-2023 winter and contrast snowfall between sites for each event.

**Stretch goal: a case-study of a strong storm, potentially with verification of the forecast data with observations.**
