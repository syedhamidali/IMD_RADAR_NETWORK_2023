#!/usr/bin/env python
# coding: utf-8

# # Visuaization
# - @author: Hamid Ali Syed
# - @email: hamidsyed37[at]gmail.com

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import hvplot.pandas
import geopandas as gpd
import geoviews as gv
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from IPython.display import display


# In[2]:


df = pd.read_csv("IMD_Radar_Sites_2022.csv").drop(["Unnamed: 0", "State"], axis=1)


# In[3]:


counts = df.groupby('Band').agg(count=('Band', 'size'))


# In[4]:


counts


# In[5]:


import shapely.geometry as sgeom
import numpy as np
from cartopy.geodesic import Geodesic
def draw_circle_on_map(df):
    gd = Geodesic()
    geoms = []
    for _, row in df.iterrows():
        lon, lat = row['Longitude'], row['Latitude']
        if row['Band'] == "X":
            radius=100e3
        else:
            radius=250e3
        cp = gd.circle(lon=lon, lat=lat, radius=radius)
        geoms.append(sgeom.Polygon(cp))
    gdf = gpd.GeoDataFrame(df, geometry=geoms)
    return gdf


# In[6]:


gdf = draw_circle_on_map(df)
gdf


# In[7]:


points = df.hvplot.points(x='Longitude', y='Latitude', geo=True, color='Band',
                          alpha=0.7, coastline = True,
                 xlim=(df.Longitude.min()-5, df.Longitude.max()+3),
                 ylim=(df.Latitude.min()-3, df.Latitude.max()+3),
                 tiles='OpenTopoMap', frame_height=800, hover_cols=['Site', 'Band'], value_label='Count')

# Create the circle plot
circles = gv.Polygons(data=gdf.geometry,).opts(color = "gray", fill_alpha=0.2, xlabel = "Longitude˚E", ylabel = "Latitude˚N")
# Overlay the circle plot on top of the point plot
plot = points * circles

# Show the plot
display(plot)


# In[8]:


import urllib.request
url = "https://raw.githubusercontent.com/syedhamidali/test_scripts/master/map_features.py"
urllib.request.urlretrieve(url, "map_features.py")
import map_features as mf


# In[9]:


get_ipython().system('git clone https://github.com/aman1chaudhary/India-Shapefiles.git')


# In[10]:


india = gpd.read_file("India-Shapefiles/India Boundary/")
states = gpd.read_file("India-Shapefiles/India States Boundary/")


# In[11]:


fig = plt.figure(figsize = [10,12], dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree(), frameon=False)
BAND = ["X", "C", "S"]
col = ['red', '#4B04E2', '#58D68D']

# Count occurrences of each 'Band' value
band_counts = df['Band'].value_counts().to_dict()

for band, c in zip(BAND, col):
    # Get the count for the current 'Band' value
    count = band_counts[band]
    # Create the label for the legend
    label = f"{band} - Band ({count})"
    df[df['Band']==band].plot.scatter(x='Longitude', y='Latitude', ax=ax, c=c, s=10, label=label, zorder=10)

    ax.add_geometries(gdf[gdf.Band == band].geometry, crs=ccrs.PlateCarree(), 
                      alpha=0.4, edgecolor="k", facecolor=c) 
    
# Add text labels to each point
for i, txt in enumerate(df['Site']):
    x = df['Longitude'][i]
    y = df['Latitude'][i]
    if txt == "Delhi":
        y -= 0.5
    dx = 0.01 * (max(df['Longitude']) - min(df['Longitude']))
    dy = 0.01 * (max(df['Latitude']) - min(df['Latitude']))
    ax.text(x + dx, y + dy, txt, fontsize=8)
india.plot(ax=ax, ec = "k", fc = "none", lw=0.5, alpha = 0.6, )
states.plot(ax=ax, ec ="k", fc = "none", lw=0.2, alpha = 0.5, ls=":")
ax.legend(title = f"Total DWRs: {counts.sum()[0]}", shadow = True)
mf.map_features(ax=ax, ocean=True, borders=False, states=False, land=True)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=12, width=0.5, color='#555555', length=8, direction='out')
ax.tick_params(axis='both', which='minor', labelsize=10, width=0.5, color='#555555', length=4, direction='out')
ax.set_xticks(np.arange(df.Longitude.min(), df.Longitude.max()+1, 5))
ax.set_yticks(np.arange(df.Latitude.min(), df.Latitude.max()+1, 5))
ax.set_xlabel("Longitude˚E")
ax.set_ylabel("Latitude˚N")
ax.set_extent([65, 98, 5, 37])
ax.set_autoscale_on(True)
plt.show()


# In[12]:


import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import urllib.request
from PIL import Image
import io

url = "https://mausam.imd.gov.in/imd_latest/contents/map-marker-icon-png-green.png"

# Open the URL and read the image data into a bytes object
with urllib.request.urlopen(url) as response:
    img_data = response.read()

# Create a PIL Image object from the image data
ma_img = Image.open(io.BytesIO(img_data))
# Convert the PIL Image to a NumPy array
marker_img = np.array(ma_img)

# Create a function to create an OffsetImage object for each marker
def make_marker(lon, lat):
    # Set the size of the marker image
    size = 0.05

    # Convert the coordinates to the map's coordinate system
    x, y = ax.projection.transform_point(lon, lat, ccrs.PlateCarree())[:2]

    # Create the OffsetImage object
    img = OffsetImage(marker_img, zoom=size)
    img.image.axes = ax
    ab = AnnotationBbox(img, (x,y), frameon=False)
    ax.add_artist(ab)

    
BAND = ["X", "C", "S"]
col = ['red', '#4B04E2', '#04E2D8']

# Create the map figure
fig = plt.figure(figsize=[10, 12], dpi=300)
ax = plt.axes(projection=ccrs.PlateCarree(), frameon=False)

# Plot the data points
for band, c in zip(BAND, col):
    count = band_counts[band]
    label = f"{band} - Band ({count})"
    df[df['Band'] == band].plot.scatter(x='Longitude', y='Latitude', 
                                        ax=ax, c=c, s=10, label=label)
    ax.add_geometries(gdf[gdf.Band == band].geometry, crs=ccrs.PlateCarree(), 
                      alpha=0.4, edgecolor="k", facecolor=c)

# Add the custom marker to each data point
for i, row in df.iterrows():
    make_marker(row['Longitude'], row['Latitude'])

# Add text labels to each point
for i, txt in enumerate(df['Site']):
    x = df['Longitude'][i]
    y = df['Latitude'][i]
    if txt == "Delhi":
        y -= 0.5
    dx = 0.01 * (max(df['Longitude']) - min(df['Longitude']))
    dy = 0.01 * (max(df['Latitude']) - min(df['Latitude']))
    ax.text(x + dx, y + dy, txt, fontsize=8)

# Add the map features and labels
india.plot(ax=ax, ec="k", fc="none", lw=0.5, alpha=0.6)
states.plot(ax=ax, ec="k", fc="none", lw=0.2, alpha=0.5, ls=":")
mf.map_features(ax=ax, ocean=True, borders=False, states=False, land=True)
ax.minorticks_on()
ax.tick_params(axis='both', which='major', labelsize=12, width=0.5, 
               color='#555555', length=8, direction='out')
ax.tick_params(axis='both', which='minor', labelsize=10, width=0.5, 
               color='#555555', length=4, direction='out')
ax.set_xticks(np.arange(df.Longitude.min(), df.Longitude.max()+1, 5))
ax.set_yticks(np.arange(df.Latitude.min(), df.Latitude.max()+1, 5))
ax.set_xlabel("Longitude˚E")
ax.set_ylabel("Latitude˚N")
ax.set_extent([65, 98, 5, 37])
ax.set_autoscale_on(True)

# Show the legend and the plot
ax.legend(title=f"Total DWRs: {counts.sum()[0]}", shadow=True)
plt.show()

