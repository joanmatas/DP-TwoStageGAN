import pandas as pd
import json
import numpy as np
from tqdm import tqdm

real_data_path = '../data/train_filtered_interpolated.csv'
df = pd.read_csv(real_data_path)
real_trajectories = df['POLYLINE'].values
coords = []
for traj in tqdm(real_trajectories):
    traj = json.loads(traj)
    for lon, lat in traj:
        lon = float('%.3f'%(lon))
        lat = float('%.3f'%(lat))
        coords.append([lon, lat])
coords = np.array(coords)
unique_coords = np.unique(coords, axis=0)
print(len(unique_coords))
print(unique_coords)
np.save('../data/real_unique_coords_3_decimals.npy', unique_coords)