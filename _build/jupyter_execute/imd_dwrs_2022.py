#!/usr/bin/env python
# coding: utf-8

# # Data Preparation
# - author: Hamid Ali Syed
# - email: hamidsyed37[at]gmail[dot]com

# In[1]:


import warnings
warnings.filterwarnings("ignore")
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


# ## Data collection
# ### Let's do some Web Scrapping
# Let's do some Web Scrapping. We can scrap the radar site information from the website of the Indian Meteorological Department (IMD) and extract location information from the HTML using BeautifulSoup. Then, we can clean and transform the extracted data into a Pandas DataFrame, with longitude and latitude coordinates for each location. Finally, we can plot the DataFrame as a scatter plot using longitude and latitude as x and y axes, respectively.

# In[2]:


url = "https://mausam.imd.gov.in/imd_latest/contents/index_radar.php"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# extract the relevant part of the HTML
images_html = soup.find_all("script")[-2].text.split("images: [")[0].split("],\n")[0]

# split the HTML into individual locations and extract the relevant information
locations = []
for image in soup.find_all("script")[-2].text.split("images: [")[0].split("],\n")[0].split("{")[1:]:
    location_dict = {}
    for line in image.split("\n"):
        if "title" in line:
            location_dict["title"] = line.split(": ")[-1].strip(',')
        elif "latitude" in line:
            location_dict["latitude"] = line.split(": ")[-1].strip(',')
        elif "longitude" in line:
            location_dict["longitude"] = line.split(": ")[-1].strip(',')
    locations.append(location_dict)

# create a DataFrame from the list of dictionaries
df = pd.DataFrame(locations)
df = df.dropna()
df['title'] = df['title'].str.strip(", ").str.strip('"')
df['longitude'] = df['longitude'].str.strip(", ").str.strip('longitude":')
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
df['title'].replace("Goa", "Panaji", inplace=True)
df.plot(kind='scatter', x='longitude', y='latitude')
df


# We have procured the name, lat, & lon info of all the radar sites and is saved in `df`. Now, let's search for their frequency bands. I have found a webpage on the IMD website that contains this information for most of the radars. Let's make a request to a URL and create a BeautifulSoup object to parse the HTML content. It will find a table on the page, then we can extract the headers and rows of the table, and create a Pandas DataFrame `df2` from the table data.
# <p>Drop the "S No" column, clean up the "Type of DWR" and "DWR Station" columns by removing certain text, and replace some values in the "DWR Station" column.</p>

# In[3]:


# make a request to the URL
url = "https://mausam.imd.gov.in/imd_latest/contents/imd-dwr-network.php"
response = requests.get(url)

# create a BeautifulSoup object
soup = BeautifulSoup(response.content, "html.parser")

# find the table on the page
table = soup.find("table")

# extract the table headers
headers = [header.text.strip() for header in table.find_all("th")]

# extract the table rows
rows = []
for row in table.find_all("tr")[1:]:
    cells = [cell.text.strip() for cell in row.find_all("td")]
    rows.append(cells)

# create a DataFrame from the table data
df2 = pd.DataFrame(rows, columns=headers)
df2.drop("S No", axis=1, inplace=True)
df2['Type of DWR'] = df2['Type of DWR'].str.replace(' - Band', '')
df2['DWR Station'].replace('Delhi (Palam)', 'Palam', inplace=True)
df2['DWR Station'] = df2['DWR Station'].str.replace('\(ISRO\)', '').str.replace('\(Mausam Bhawan\)', 
                                                                                '').str.strip()
df2


# Let's merge two previously created Pandas DataFrames, `df` and `df2`, using the "title" and "DWR Station" columns as keys, respectively. It will drop the "DWR Station" column, rename the "Type of DWR" column as "Band", and replace some values in the "title" column. The code will count the number of NaN values in the "Band" column, print this count, and return the resulting merged DataFrame.

# In[4]:


merged_df = df.merge(df2, left_on='title', right_on='DWR Station', how='left')
merged_df = merged_df.drop(columns=['DWR Station'])
merged_df = merged_df.rename(columns={'Type of DWR': 'Band'})
merged_df['title'].replace("Goa", "Panaji", inplace=True)
num_nans = merged_df['Band'].isna().sum()
print(num_nans)
merged_df


# Since there are NaN values in the "State" column, we can find the state names using lat and lon info. We can use the Cartopy library to create a map of India with state boundaries and labels. Then we will create a pandas DataFrame `gdf` containing the latitude and longitude coordinates of each state and union territory, and try to map the names of these places in the `merged_df`

# In[5]:


import cartopy.io.shapereader as shpreader
import geopandas as gpd
# Load the Natural Earth dataset
states_shp = shpreader.natural_earth(resolution='10m',
                                     category='cultural',
                                     name='admin_1_states_provinces')


# In[6]:


import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cartopy.feature as feat

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

# get the data
fn = shpreader.natural_earth(
    resolution='10m', category='cultural', 
    name='admin_1_states_provinces',
)
reader = shpreader.Reader(fn)
states = [x for x in reader.records() if x.attributes["admin"] == "India"]
states_geom = feat.ShapelyFeature([x.geometry for x in states], ccrs.PlateCarree())

data_proj = ccrs.PlateCarree()

# create the plot
fig, ax = plt.subplots(
    figsize=(10,10), dpi=70, facecolor="w",
    subplot_kw=dict(projection=data_proj),
)

ax.add_feature(feat.BORDERS, color="k", lw=0.1)
# ax.add_feature(feat.COASTLINE, color="k", lw=0.2)
ax.set_extent([60, 100, 5, 35], crs=ccrs.Geodetic())

ax.add_feature(states_geom, facecolor="none", edgecolor="k")

# # add the names
for state in states:
    lon = state.geometry.centroid.x
    lat = state.geometry.centroid.y
    name = state.attributes["name"] 
    
    ax.text(
        lon, lat, name, size=7, transform=data_proj, ha="center", va="center",
        path_effects=[PathEffects.withStroke(linewidth=5, foreground="w")]
    )


# In[7]:


locs = {}
for state in states:
    lon = state.geometry.centroid.x
    lat = state.geometry.centroid.y
    name = state.attributes["name"]
    locs[name] = {"lat": lat, "lon": lon}
gdf = pd.DataFrame(locs, ).T
gdf.reset_index(inplace=True)
gdf = gdf.rename({'index':'state'}, axis=1)
gdf.index = np.arange(1, len(gdf) + 1)


# In[8]:


gdf


# In[9]:


# merged_df.sort_values(by=['latitude', 'longitude'], ascending=False)


# In[10]:


merged_df = df.merge(df2, left_on='title', right_on='DWR Station', how='left')
merged_df = merged_df.drop(columns=['DWR Station'])
merged_df = merged_df.rename(columns={'Type of DWR': 'Band'})
merged_df


# In[11]:


from math import radians, sin, cos, sqrt, asin

# Function to calculate the haversine distance between two coordinates in km
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # radius of Earth in km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R*c

# Loop through each row in merged_df
for i, row in merged_df.iterrows():
    if pd.isna(row["State"]):
        min_dist = float('inf')
        closest_state = ""
        # Loop through each row in gdf to find the closest one
        for j, gdf_row in gdf.iterrows():
            dist = haversine(row["latitude"], row["longitude"], gdf_row["lat"], gdf_row["lon"])
            if dist < min_dist:
                min_dist = dist
                closest_state = gdf_row["state"]
        merged_df.at[i, "State"] = closest_state


# In[12]:


merged_df.sort_values(by = "latitude", ascending=False)


# In[13]:


# Merge merged_df and df2 on the "State" column
merged_df_with_band = pd.merge(merged_df, df2[['State', 'Type of DWR']], on='State', how='left')

# Replace NaN values in the "Band" column with corresponding values from the "Type of DWR" column
merged_df_with_band['Band'].fillna(merged_df_with_band['Type of DWR'], inplace=True)

# Drop the "Type of DWR" column
merged_df_with_band.drop('Type of DWR', axis=1, inplace=True)


# In[14]:


merged_df_with_band.drop(2, inplace=True)


# In[15]:


merged_df_with_band.drop_duplicates("latitude", inplace=True)
merged_df_with_band.drop_duplicates("longitude", inplace=True)
merged_df_with_band.sort_values(by="latitude", ascending=False, inplace=True)


# In[16]:


merged_df_with_band.index = np.arange(1, len(merged_df_with_band)+1, 1)


# In[17]:


f_df = merged_df_with_band.copy()


# In[18]:


f_df.loc[f_df['title'] == 'Veravali', 'Band'] = 'C'


# In[19]:


nan_mask = f_df['Band'].isna()
nan_df = f_df[nan_mask]
nan_df.loc[:, 'Band'] = nan_df['Band'].fillna('X')
f_df.update(nan_df)


# In[20]:


f_df.rename(columns={'title': 'Site',
                     "latitude": "Latitude", 
                     "longitude":"Longitude"}, inplace=True)


# In[21]:


f_df.attrs["Range"]={"C":250,
                    "X":100,
                    "S":250
                   }


# In[22]:


f_df.to_csv("IMD_Radar_Sites_2022.csv")

