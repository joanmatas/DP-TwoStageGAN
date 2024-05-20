import sys
import pandas as pd
import numpy as np
import json
import scipy.stats
import os
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

MIN_LON=-8.687466
MIN_LAT=41.123232
MAX_LON=-8.553186
MAX_LAT=41.237424

D_LON = (MAX_LON-MIN_LON)/32
D_LAT = (MAX_LAT-MIN_LAT)/32

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance

def get_point_to_point_geodistances(trajs):
    distances = []
    for traj in trajs:
        traj = json.loads(traj)
        for i in range(len(traj) - 1):
            lng1 = traj[i][0]
            lat1 = traj[i][1]
            lng2 = traj[i + 1][0]
            lat2 = traj[i + 1][1]
            distances.append(geodistance(lng1, lat1, lng2, lat2))
    distances = np.array(distances, dtype=float)
    return distances

def get_trajectory_geodistances(trajs):
    distances = []
    for traj in trajs:
        travel_distance = 0
        traj = json.loads(traj)
        for i in range(len(traj) - 1):
            lng1 = traj[i][0]
            lat1 = traj[i][1]
            lng2 = traj[i + 1][0]
            lat2 = traj[i + 1][1]
            travel_distance += geodistance(lng1, lat1, lng2, lat2)
        distances.append(travel_distance)
    distances = np.array(distances, dtype=float)
    return distances

def get_trajectory_durations(trajs):
    durations = []
    for traj in trajs:
        traj = json.loads(traj)
        travel_time = len(traj) # * 15.0 / 60.0
        durations.append(travel_time)
    durations = np.array(durations, dtype=float)
    return durations

@staticmethod
def arr_to_distribution(arr, min, max, bins):
    """
    convert an array to a probability distribution
    :param arr: np.array, input array
    :param min: float, minimum of converted value
    :param max: float, maximum of converted value
    :param bins: int, number of bins between min and max
    :return: np.array, output distribution array
    """
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution, base[:-1]

@staticmethod
def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-14)
        p2 = p2 / (p2.sum()+1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js

def get_N_most_visited_grids(trajs, N):
    visited_grids = []
    for traj in trajs:
        traj = json.loads(traj)
        for lon, lat in traj:
            a = int((float(lat) - MIN_LAT) // D_LAT)
            b = int((float(lon) - MIN_LON) // D_LON)
            if a == 32:
                a = 31
            if b == 32:
                b = 31
            if a > 32 or b > 32:
                print('Grid out of bounds')
                print(lat, lon)
                print(a, b)
                sys.exit()
            grid = a * 32 + b
            visited_grids.append(grid)
    num_coords = len(visited_grids)
    visited_grids = np.array(visited_grids)
    unique_grids, grid_counts = np.unique(visited_grids, return_counts=True)
    return np.divide(np.flip(np.sort(grid_counts))[:N], num_coords)

def get_retention_percentage(fake_trajs):
    real_unique_coords = np.load('real_unique_coords_3_decimals.npy', allow_pickle=True)
    fake_coords = []
    retention_count = 0
    for traj in fake_trajs:
        traj = json.loads(traj)
        for lon, lat in traj:
            lon = float('%.3f'%(lon))
            lat = float('%.3f'%(lat))
            coord = [lon, lat]
            if coord in real_unique_coords and not coord in fake_coords:
                retention_count += 1
                fake_coords.append(coord)
    return retention_count / len(real_unique_coords) * 100

num_bins = 10000
real_data_path = '../data/train_filtered_interpolated.csv'
fake_data_path = '../output/final_trajectories/numpys'
figures_path = 'evaluation_figures'
print('Epsilon 10')
print('Loading real data...')
df = pd.read_csv(real_data_path)
real_trajectories = df['POLYLINE'].values
print('Loading generated data...')
fake_trajectories = []
for fake_trajectory_name in os.listdir(fake_data_path):
    fake_trajectory = np.load(os.path.join(fake_data_path, fake_trajectory_name), allow_pickle=True)
    fake_trajectory = fake_trajectory.tolist()
    fake_trajectories.append(json.dumps(fake_trajectory))

num_real_trajs = len(real_trajectories)
num_fake_trajs = len(fake_trajectories)

print('Calculating distance between travel distance distributions...')
real_trajectory_distances = get_trajectory_geodistances(real_trajectories)
r_traj_dist_min = real_trajectory_distances.min()
r_traj_dist_max = real_trajectory_distances.max()

fake_trajectory_distances = get_trajectory_geodistances(fake_trajectories)
f_traj_dist_min = fake_trajectory_distances.min()
f_traj_dist_max = fake_trajectory_distances.max()

pdl_distribution_min = min(r_traj_dist_min, f_traj_dist_min)
pdl_distribution_max = max(r_traj_dist_max, f_traj_dist_max)

real_pdl, real_pdl_base = arr_to_distribution(real_trajectory_distances, pdl_distribution_min, pdl_distribution_max, num_bins)
real_pdl = np.divide(real_pdl, num_real_trajs)
fake_pdl, fake_pdl_base = arr_to_distribution(fake_trajectory_distances, pdl_distribution_min, pdl_distribution_max, num_bins)
fake_pdl = np.divide(fake_pdl, num_fake_trajs)

fig1, ax1 = plt.subplots()
ax1.bar(real_pdl_base, real_pdl)
ax1.set_ybound(lower=0, upper=max(real_pdl.max(), fake_pdl.max()))
ax1.set_title('Original Dataset Travel Distance Distribution')
ax1.set_ylabel('frequency')
ax1.set_xlabel('trajectory travel distance')
fig1.savefig(os.path.join(figures_path, 'real_pdl.jpg'))

fig2, ax2 = plt.subplots()
ax2.bar(fake_pdl_base, fake_pdl)
ax2.set_ybound(lower=0, upper=max(real_pdl.max(), fake_pdl.max()))
ax2.set_title('Generated Dataset Travel Distance Distribution')
ax2.set_ylabel('frequency')
ax2.set_xlabel('trajectory travel distance')
fig2.savefig(os.path.join(figures_path, 'fake_pdl.jpg'))

pdl_jsd = get_js_divergence(real_pdl, fake_pdl)
print('Pd(l) distance:', pdl_jsd)

print('Calculating distance between travel time distributions...')
real_trajectory_durations = get_trajectory_durations(real_trajectories)
r_traj_dur_min = real_trajectory_durations.min()
r_traj_dur_max = real_trajectory_durations.max()

fake_trajectory_durations = get_trajectory_durations(fake_trajectories)
f_traj_dur_min = fake_trajectory_durations.min()
f_traj_dur_max = fake_trajectory_durations.max()

psl_distribution_min = min(r_traj_dur_min, f_traj_dur_min)
psl_distribution_max = max(r_traj_dur_max, f_traj_dur_max)

real_psl, real_psl_base = arr_to_distribution(real_trajectory_durations, psl_distribution_min, psl_distribution_max, num_bins)
real_psl = np.divide(real_psl, num_real_trajs)
fake_psl, fake_psl_base = arr_to_distribution(fake_trajectory_durations, psl_distribution_min, psl_distribution_max, num_bins)
fake_psl = np.divide(fake_psl, num_fake_trajs)

fig3, ax3 = plt.subplots()
ax3.bar(real_psl_base, real_psl)
ax3.set_ybound(lower=0, upper=max(real_psl.max(), fake_psl.max()))
ax3.set_title('Original Dataset Travel Duration Distribution')
ax3.set_ylabel('frequency')
ax3.set_xlabel('trajectory points')
fig3.savefig(os.path.join(figures_path, 'real_psl.jpg'))

fig4, ax4 = plt.subplots()
ax4.bar(fake_psl_base, fake_psl)
ax4.set_ybound(lower=0, upper=max(real_psl.max(), fake_psl.max()))
ax4.set_title('Generated Dataset Travel Duration Distribution')
ax4.set_ylabel('frequency')
ax4.set_xlabel('trajectory points')
fig4.savefig(os.path.join(figures_path, 'fake_psl.jpg'))

psl_jsd = get_js_divergence(real_psl, fake_psl)
print('Ps(l) distance:', psl_jsd)

print('Calculating distance between top N visited grids distributions...')
N = 50
real_por = get_N_most_visited_grids(real_trajectories, N)
fake_por = get_N_most_visited_grids(fake_trajectories, N)

fig5, ax5 = plt.subplots()
ax5.bar(range(N), real_por)
ax5.set_ybound(lower=0, upper=max(real_por.max(), fake_por.max()))
ax5.set_title('Original Dataset Top 50 Visited Places')
ax5.set_ylabel('frequency')
ax5.set_xlabel('50 most visited grids')
fig5.savefig(os.path.join(figures_path, 'real_por.jpg'))

fig6, ax6 = plt.subplots()
ax6.bar(range(N), fake_por)
ax6.set_ybound(lower=0, upper=max(real_por.max(), fake_por.max()))
ax6.set_title('Generated Dataset Top 50 Visited Places')
ax6.set_ylabel('frequency')
ax6.set_xlabel('50 most visited grids')
fig6.savefig(os.path.join(figures_path, 'fake_por.jpg'))

por_jsd = get_js_divergence(real_por, fake_por)
print('Po(r) distance:', por_jsd)

print('Calculating retention percentage with 3 decimals...')
print(get_retention_percentage(fake_trajectories))