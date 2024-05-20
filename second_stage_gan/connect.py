import cv2 as cv
import numpy as np
import random
import torch
from link_mod import DFS

def transform(enter_points, exit_points, w, h):
    '''
    e.g. (0, h - 1) --> (1, 1); (w - 1, 0) --> (-1, -1)
    '''
    enter_points = enter_points[:, ::-1]
    exit_points = exit_points[:, ::-1]
    enter_points = enter_points * \
        np.array([[2/(h-1), 2/(1-w)]]) + np.array([[-1.0, 1.0]])
    exit_points = exit_points * \
        np.array([[2/(h-1), 2/(1-w)]]) + np.array([[-1.0, 1.0]])
    return enter_points, exit_points

def get_position(side):
    '''
    return the indexes of road points
    '''
    WHITE1 = (255, 255, 255)
    WHITE2 = (254, 254, 254)
    WHITE3 = (254, 255, 255)
    WHITE4 = (254, 254, 255)
    YELLOW = (255, 204, 102)
    ORANGE = (255, 179, 112)
    points = []
    l, c = side.shape
    b, g, r = side[:, 0], side[:, 1], side[:, 2]
    points = [i for i in range(l) if 
              (r[i], g[i], b[i]) == WHITE1 or 
              (r[i], g[i], b[i]) == WHITE2 or 
              (r[i], g[i], b[i]) == WHITE3 or 
              (r[i], g[i], b[i]) == WHITE4 or
              (r[i], g[i], b[i]) == YELLOW or
              (r[i], g[i], b[i]) == ORANGE]
    if len(points) == 0: ### # In case points is empty
        points = [int(l/2)]
    return points

def get_point(map_grid, direction):
    '''
    return all the road points on the given side
    '''
    side = None
    w, h, c = map_grid.shape
    if direction == 'left':
        side = map_grid[:, 0, :]
        x_position = get_position(side)
        points = [[x, 0] for x in x_position]
    elif direction == 'right':
        side = map_grid[:, h-1, :]
        x_position = get_position(side)
        points = [[x, h-1] for x in x_position]
    elif direction == 'up':
        side = map_grid[0, :, :]
        y_position = get_position(side)
        points = [[0, y] for y in y_position]
    elif direction == 'down':
        side = map_grid[w-1, :, :]
        y_position = get_position(side)
        points = [[w-1, y] for y in y_position]
    assert(len(points) != 0) # In case points is empty
    return np.array(points)

def get_all_enter_exit_point(map_grids, enter_direction, exit_direction, downsample=1.0):
    '''
    return all the points on the road in the enter direction & exit direction
    '''
    w, h, c = map_grids.shape
    if downsample < 1:
        w, h = int(w*downsample), int(h*downsample)
        map_grids = cv.resize(map_grids, (h, w))
    enter_points = get_point(map_grids, enter_direction)
    exit_points = get_point(map_grids, exit_direction)
    enter_points, exit_points = transform(enter_points, exit_points, w, h)
    return enter_points, exit_points


def get_random_enter_exit_point(map_grid, enter_direction, exit_direction, downsample=1.0, last_exit_point=None):
    '''
    return the enter point & exit point
    '''
    enter_points, exit_points = get_all_enter_exit_point(
        map_grid, enter_direction, exit_direction, downsample)
    if last_exit_point is None:
        enter_point = enter_points[random.sample(range(enter_points.shape[0]), 1)]
    else:
        if enter_direction == 'left':
            enter_point = np.array([[-1, last_exit_point[0,1]]])
        elif enter_direction == 'right':
            enter_point = np.array([[1, last_exit_point[0,1]]])
        elif enter_direction == 'up':
            enter_point = np.array([[last_exit_point[0,0], 1]])
        elif enter_direction == 'down':
            enter_point = np.array([[last_exit_point[0,0], -1]])
    exit_point = exit_points[random.sample(range(exit_points.shape[0]), 1)] #shape=(1,2)
    return enter_point, exit_point


def grid(map_whole):
    '''
    return a dict of map grids
    '''
    map_dict = {}
    w, h, c = map_whole.shape
    wn = int(w/32)
    hn = int(h/32)
    for i in range(32):
        for j in range(32):
            map_dict['%d_%d' % (i, j)] = map_whole[i *
                                                       wn:i*wn+wn, j*hn:j*hn+hn, :]
    return map_dict

def get_direction(grid_a, grid_b):
    dx, dy = grid_b[0]-grid_a[0], grid_b[1]-grid_a[1]
    direction = {(-1, 0): 'up', (1, 0): 'down',
                 (0, 1): 'right', (0, -1): 'left'}
    return (direction[(dx, dy)], direction[(-dx, -dy)])

def get_direction_list(grid_data):
    enter_list = []
    exit_list = []
    length = len(grid_data)
    for i in range(length-1):
        exit, enter = get_direction(grid_data[i], grid_data[i+1])
        enter_list.append(enter)
        exit_list.append(exit)
    enter_list.insert(0, get_direction(grid_data[0], grid_data[1])[1])
    exit_list.insert(-1, get_direction(grid_data[-2], grid_data[-1])[0])
    return enter_list, exit_list

def to_seq(mat):
    # Create the tensor
    shape = [len(mat), len(mat[0]), len(mat[0][0])]
    flat_mat = mat.flatten()
    tensor = torch.Tensor(flat_mat).reshape(shape)[0]

    # Call the class DFS
    dfs = DFS(tensor)
    # Run the fit function to obtain the trajectory vector g
    g = dfs.fit()

    return g