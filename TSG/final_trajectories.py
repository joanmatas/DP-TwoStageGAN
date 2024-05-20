import os
import numpy as np
import cv2 as cv
import random
import math
from tqdm import tqdm
from connect import get_direction_list, to_seq

MIN_LON=-8.687466
MIN_LAT=41.123232
MAX_LON=-8.553186
MAX_LAT=41.237424

D_LON = (MAX_LON-MIN_LON)/32
D_LAT = (MAX_LAT-MIN_LAT)/32

def to_coordinate(x, y, pred):
    i, j = int(x), int(y)
    pred = np.add(pred / 2, np.array([i, j]))
    pred = pred * np.array([D_LON, D_LAT]) + np.array([MIN_LON+D_LON/2, MIN_LAT+D_LAT/2])
    return pred

def get_road_points(map_grid):
    WHITE1 = (255, 255, 255)
    WHITE2 = (254, 254, 254)
    WHITE3 = (254, 255, 255)
    WHITE4 = (254, 254, 255)
    YELLOW = (255, 204, 102)
    ORANGE = (255, 179, 112)
    h, w, c = map_grid.shape
    map_grid[:math.ceil(h/20), :, :] = (0, 0, 0)
    map_grid[-math.floor(h/20):, :, :] = (0, 0, 0)
    map_grid[:, :math.ceil(w/20), :] = (0, 0, 0)
    map_grid[:, -math.floor(w/20):, :] = (0, 0, 0)
    b, g, r = map_grid[:, :, 0], map_grid[:, :, 1], map_grid[:, :, 2]
    road_points = [[i, j] for i in range(h) for j in range(w) if
                   (r[i][j], g[i][j], b[i][j]) == WHITE1 or
                   (r[i][j], g[i][j], b[i][j]) == WHITE2 or
                   (r[i][j], g[i][j], b[i][j]) == WHITE3 or
                   (r[i][j], g[i][j], b[i][j]) == WHITE4 or
                   (r[i][j], g[i][j], b[i][j]) == YELLOW or
                   (r[i][j], g[i][j], b[i][j]) == ORANGE]
    return road_points

def get_seq_inside_grid(road_points, h, w, enter_direction, exit_direction, number_of_points):
    points = random.sample(road_points, number_of_points)        
    points = points * np.array([[2/(1-h), 2/(w-1)]]) + np.array([[1.0, -1.0]])
    if enter_direction == 'down' or exit_direction == 'up':
        ind = np.lexsort((points[:,1],points[:,0]))
    elif enter_direction == 'up' or exit_direction == 'down':
        ind = np.lexsort((points[:,1],points[:,0]))
        ind = np.flip(ind)
    elif enter_direction == 'left' or exit_direction == 'right':
        ind = np.lexsort((points[:,0],points[:,1]))
    elif enter_direction == 'right' or exit_direction == 'left':
        ind = np.lexsort((points[:,0],points[:,1]))
        ind = np.flip(ind)
    points = points[ind]

    swapped_points = np.empty_like(points)
    swapped_points[:, 0] = points[:, 1]
    swapped_points[:, 1] = points[:, 0]

    return swapped_points

def main():
    step_1_output = '../data/generated_coarse'
    map_dir = '../data/pics'
    to_seq_tries = 5
    count = 0
    h, w = 526, 467
    downsample = 0.25
    h, w = int(h*downsample), int(w*downsample)
    grid_road_points = []

    print('Loading map images and extracting road points...')
    for i in range(1024):
        pic_name = str(i // 32) + '_' + str(i % 32) + '.png'
        map_pic = cv.resize(cv.imread(os.path.join(map_dir, pic_name)), (w, h))
        grid_road_points.append(get_road_points(map_pic))
        del map_pic

    print('Generating trajectories...')
    for grid_data_name in tqdm(os.listdir(step_1_output)):
        traj = []
        flag = True
        try:
            generated_traj_mat = np.load(os.path.join(
                step_1_output, grid_data_name), allow_pickle=True)
        except OSError:
            continue

        if not np.any(generated_traj_mat[0]):
            count += 1
            # print('matrix drop out: %d' % (count))
            continue

        # choose the longest obtained sequence in to_seq_tries tries
        grid_data = []
        for _ in range(to_seq_tries):
            new_grid_data = to_seq(generated_traj_mat)    # [[x0, y0, t0], [x1, y1, t1], ... ]
            if len(new_grid_data) > len(grid_data):
                grid_data = new_grid_data

        if len(grid_data) <= 2:
            count += 1
            # print('sequence drop out: %d' % (count))
            # print(grid_data)
            continue

        enter_list, exit_list = get_direction_list(grid_data)
        for grid_point, enter, exit in zip(grid_data, enter_list, exit_list):
            x = grid_point[0]
            y = grid_point[1]
            duration = grid_point[2]
            seq_length = math.ceil(duration/15)
            grid_pic_index = y * 32 + (31-x)
            # grid_pic_name = '%d_%d' % (y, 31-x)
            # grid_pic = grid_pic_name + '.png'
            try:
                points = get_seq_inside_grid(
                    grid_road_points[grid_pic_index], h, w, enter, exit, seq_length
                )
            except ValueError:
                flag = False
                break
            coords = to_coordinate(y, 31-x, points)
            traj.append(coords)
        if flag == False:
            continue
        traj = np.concatenate(traj, axis=0)
        text_file_name = '../output/final_trajectories/text_files/' + grid_data_name[:-4] + '.txt'
        with open(text_file_name, 'w') as file:
            for coord in traj:
                file.write(str(coord[1]) + ',' + str(coord[0]) + ',\n')
        # print('Save trajectory: %s'%(grid_data_name[:-4]))
        np.save(os.path.join('../output/final_trajectories/numpys/', grid_data_name[:-4]), traj)

if __name__ == '__main__':
    main()