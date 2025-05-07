# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:14:32 2024

@author: chypu
"""

#%% import library
import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt 
import pykrige.kriging_tools as kt
# for classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import cluster
import scipy.cluster

#%%
# Updating Parameters for Paper
plt.rcdefaults()
params = {
    'lines.linewidth' :0.3,
    'lines.markersize' :1,
   'axes.labelsize': 8,
    'axes.titlesize':8,
    'axes.titleweight':'normal',
    'font.size': 8,
    'font.family': 'Times New Roman', # 'Times New RomanArial'
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
    'legend.shadow':'False',
   'legend.fontsize': 8,
   'xtick.labelsize':8,
   'ytick.labelsize':8,
   'text.usetex': False,
    'figure.autolayout': True,
   'figure.figsize': [2.5,2.5] # width and height in inch 3.5,2.54(max width = 7.5", max length = 10") [6.0,4.05
   }
plt.rcParams.update(params)

#%%  ordinary kriging with y, z axies
#%% conductivity 2D y and z
#%% data of a single time stamp (like 0 time)
x=np.array([0.85,0.85,0.85,1.15,1.15,1.15,1.45,1.45,1.45]) # x-coordinate
y=np.array([0.2, 0.5, 0.8, 0.2,0.5,0.8, 0.2, 0.5,0.8]) # y-coordinate
z=np.array([0.25,0.25,0.25,0.35, 0.35,0.35,0.45,0.45,0.45]) # z-coordinate
# V=np.array([0.0032, 0,0,0,0,0.0106, 0.00036,0,0]) # 1st time stamp TS1: 1448 sec
# V=np.array([0.0546, 0.0257, 0.0578, 0.0385, 0.0385,0.0176, 0.0215, 0.0802, 0.0835]) # 2nd time stamp TS2: 2349.091 sec
# V=np.array([0.0963,0.1733,0.061,0.0674, 0.0674, 0.0176, 0.0215, 0.0867, 0.0867]) # 3rd time stamp TS3: 3188.7081 sec
# V=np.array([0.0738,0.077, 0.0578,0.0706, 0.0738,0.0176, 0.0215, 0.0835, 0.0835]) # 4th time stamp TS4: 4908.402 sec
V=np.array([0.0706,0.0738,0.061,0.0706,0.077,0.0247,0.0322,0.0899,0.0867]) # 5th time stamp TS5: 6567.4691 sec
#%% Ordinary kriging
OK = OrdinaryKriging(
    y,
    z,
    V,
    variogram_model='gaussian',
    verbose=True,
    enable_plotting=False, # for remove below red dot line make it false
)
#%% 
gridy=np.arange(0.0,1,0.01,dtype='float64') 
gridz=np.arange(0.0,0.50,0.005, dtype='float64')  #  for make make both array size same. ValueError: All arrays must be of the same length
# gridz=np.arange(0.0,0.45,0.0288888888888889, dtype='float64') 
zstar,ss=OK.execute("grid", gridy, gridz)


#%% 
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Plot the data
fig, ax = plt.subplots()  # Use the figsize set in your parameters
cax = ax.imshow(zstar, extent=(0, 1, 0, 0.5), origin='lower')

# Add scatter plot
ax.scatter(y, z, marker='.', s=100, facecolors='none', edgecolors='k')

# Create an axes divider to adjust the colorbar's size and position
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="5%", pad=0.05)  # "right" means to the right of the plot

# Add the colorbar
cbar = fig.colorbar(cax, cax=cbar_ax)
cbar.set_label('voltage (V)')

# Set x and y labels
ax.set_xlabel('y-coordinate (m)')
ax.set_ylabel('z-coordinate (m)')

# Set more x-ticks
ax.set_xticks(np.arange(0, 1.1, 0.25))  # Adjust step size for more tick marks
ax.set_yticks(np.arange(0, 0.5, 0.10))  # Adjust step size for more tick marks
# Save the plot
# plt.savefig('plots/spatial_krigtest_2Dyz_TS1_1448s.png', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS1_1448s.pdf', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS2_2349.091s.png', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS2_2349.091s.pdf', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS3_3188.7081s.png', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS3_3188.7081s.pdf', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS4_4908.402s.png', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS4_4908.402s.pdf', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS5_6567.4691s.png', dpi=400)
# plt.savefig('plots/spatial_krigtest_2Dyz_TS5_6567.4691s.pdf', dpi=400)
plt.show()

# Assuming zstar, gridx, gridy are already defined

# Formatting data for clustering
all_y = np.tile(gridy, gridy.shape[0])
all_z = np.transpose(np.tile(gridz, (gridz.shape[0], 1))).flatten()
all_v = zstar.flatten()
df = pd.DataFrame({"y_cord": all_y,
                   "z_cord": all_z,
                   "voltage": all_v})

# Clustering
k = 3
model = cluster.KMeans(n_clusters=k, init='k-means++')
X = df[["voltage"]]
df_X = X.copy()
df_X["cluster"] = model.fit_predict(X)

# Find real centroids
closest, distances = scipy.cluster.vq.vq(model.cluster_centers_, df_X.drop("cluster", axis=1).values)
df_X["centroids"] = 0
for i in closest:
    df_X["centroids"].iloc[i] = 1

# Add clustering info to the original dataset
df[["cluster", "centroids"]] = df_X[["cluster", "centroids"]]

# Define colors for clusters (fixed for each cluster)
# colors = np.array(['orange', 'g', 'c'])  # You can add more colors if k > 3 TS1_TS1_1448s
# colors = np.array(['orange','g', 'c'])  # You can add more colors if k > 3 TS2_2349.091s
# colors = np.array(['orange','c','g'])  # You can add more colors if k > 3  TS3_3188.7081s
# colors = np.array(['orange','c','g'])  # You can add more colors if k > 3  TS4 4908.402s
colors = np.array(['orange','c','g'])  # You can add more colors if k > 3  TS5_6567.4691s
colordict = dict(zip(range(k), colors))  # Map cluster numbers to specific colors

# Apply the color mapping based on the cluster number
df["Color"] = df['cluster'].apply(lambda x: colordict[x])

# Plot the data
fig, ax = plt.subplots()
scatter = ax.scatter(df["y_cord"], df["z_cord"], c=df["Color"])

# Get centroid values and their corresponding colors
centroid_values = model.cluster_centers_

# Create a dictionary to map centroid values with their respective cluster color
centroid_color_map = {i: {'centroid_value': centroid_values[i], 'color': colordict[i]} for i in range(k)}

# Print centroid values with colors
print("Centroid Values with Colors:")
for cluster_idx, info in centroid_color_map.items():
    print(f"Cluster {cluster_idx}: Centroid {info['centroid_value']}, Color: {info['color']}")

# Optionally, plot centroids on the scatter plot
for cluster_idx, info in centroid_color_map.items():
    ax.scatter([], [], c=info['color'], label=f'Centroid {cluster_idx}')  # Add legend for centroids

# plt.legend()
plt.show()
plt.ylabel('z-coordinate')
plt.xlabel('y-coordinate')
# save clustering plots
# plt.savefig('plots/clustering_2Dyz_TS1_1448s.png',dpi=400)
# plt.savefig('plots/clustering_2Dyz_TS1_1448s.pdf',dpi=50)
# plt.savefig('plots/clustering_2Dyz_TS2_2349.091s.png',dpi=400)
# plt.savefig('plots/clustering_2Dyz_TS2_2349.091s.pdf',dpi=50)
# plt.savefig('plots/clustering_2Dyz_TS3_3188.7081s.png',dpi=400)
# plt.savefig('plots/clustering_2Dyz_TS3_3188.7081s.pdf',dpi=50)
# plt.savefig('plots/clustering_2Dyz_TS4_4908.402s.png',dpi=400)
# plt.savefig('plots/clustering_2Dyz_TS4_4908.402s.pdf',dpi=50)
plt.savefig('plots/clustering_2Dyz_TS5_6567.4691s.png',dpi=400)
plt.savefig('plots/clustering_2Dyz_TS5_6567.4691s.pdf',dpi=50)
