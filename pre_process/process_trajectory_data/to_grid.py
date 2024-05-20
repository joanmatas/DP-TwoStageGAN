import numpy as np
import pandas as pd
import json

def to_grid(
    traj,
    min_lon=-8.687466,
    min_lat=41.123232,
    max_lon=-8.553186,
    max_lat=41.237424,
    grids_num=32,
    d=0.01,
    name=None,
    max_c=0,
):
    """
    traj: single track data, ny.array format: (traj[0]-lon;traj[1]-lat)
    min_lon, min_lat: Starting value of longitude and latitude
    is_num: Whether d refers to the number of grid points
    d: Number of grid points or side length of grid points
    """
    d_lat = (max_lat - min_lat) / grids_num
    d_lon = (max_lon - min_lon) / grids_num

    traj_mat_1 = np.zeros(shape=(grids_num, grids_num))
    traj_mat_2 = np.zeros(shape=(grids_num, grids_num))
    traj_mat_3 = np.zeros(shape=(grids_num, grids_num))

    t = np.array(range(0, traj.shape[0] * 15, 15))
    t1 = np.column_stack((traj, t))
    t2 = []
    t3 = []
    # print('t1\n',t1)
    for i in t1:
        a = int((float(i[1]) - min_lat) // d_lat)
        b = int((float(i[0]) - min_lon) // d_lon)
        if a >= 32 or b >= 32 or a < 0 or b < 0:
            return None
        t2.append([b, a, i[2]])  # First lon abscissa/then lat ordinate
    # print('t2\n',t2)
    for i in t2:
        if len(t3) == 0:
            t3.append(i)
        else:
            if t3[-1][0:2] != i[0:2]:
                t3.append(i)

    for i in range(len(t3) - 1):
        t3[i][2] = t3[i + 1][2] - t3[i][2]

    t3[-1][2] = t1[-1][2] - t3[-1][2]

    for point in t3:
        # print(point)
        if traj_mat_1[-point[0], point[1]] == 0:
            traj_mat_1[-point[0], point[1]] = point[2]
        elif traj_mat_2[-point[0], point[1]] == 0:
            traj_mat_2[-point[0], point[1]] = point[2]
        elif traj_mat_3[-point[0], point[1]] == 0:
            traj_mat_3[-point[0], point[1]] = point[2]
        else:
            return
    m1, m2, m3 = np.max(traj_mat_1), np.max(traj_mat_2), np.max(traj_mat_3)
    mc = max(m1, m2, m3)
    max_c = mc if mc > max_c else max_c
    log = "%s, %d, %d, %d, %d\n" % (name + ".npy", m1, m2, m3, max_c)
    return (traj_mat_1, traj_mat_2, traj_mat_3, log, max_c, mc)


def main():
    data = pd.read_csv("../../data/train_filtered_interpolated.csv")
    trajectories = data["POLYLINE"].values
    grid_nums = 32
    max_c = 0
    with open("../../data/traj_all.txt", "w") as file:
        for i, tj in enumerate(trajectories):
            coords = np.array(json.loads(tj))
            result = to_grid(coords, grids_num=grid_nums, name=str(i), max_c=max_c)
            if result is not None:
                m1, m2, m3, log, max_c, mc = result
                if mc <= 800:
                    m = np.zeros(shape=(3, grid_nums, grid_nums))
                    m[0, :, :] = m1
                    m[1, :, :] = m2
                    m[2, :, :] = m3
                    outfile_name = (
                        "../../data/grid32_train" + str(i) + ".npy"
                    )
                    print(outfile_name)
                    np.save(outfile_name, m)
                else:
                    log = "delete" + log
                file.write(log)
    file.close()


if __name__ == "__main__":
    main()