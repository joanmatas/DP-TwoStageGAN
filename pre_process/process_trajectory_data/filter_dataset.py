import pandas as pd

df = pd.read_csv("../../data/train_with_filter_parameters_350_700_within_limits.csv")
del df['Unnamed: 0']

short_trajectories = df.loc[df['LENGTH_TOO_SHORT'] > 0]
long_trajectories = df.loc[df['LENGTH_TOO_LONG'] > 0]
# trajectories_out_of_limits = df.loc[df['POINTS_OUT_OF_LIMITS'] > 0]
discontinuous_trajectories = df.loc[df['DISCONTINUITIES'] > 0]
trajectories_with_outliers = df.loc[df['OUTLIERS'] > 0]
print("Trajectories with less than 10 points:", len(short_trajectories), "(" + str(round(len(short_trajectories) / len(df) * 100, 1)) + "%)")
print("Trajectories with more than 500 points:", len(long_trajectories), "(" + str(round(len(long_trajectories) / len(df) * 100, 1)) + "%)")
# print("Trajectories out of grid limits:", len(trajectories_out_of_limits), "(" + str(round(len(trajectories_out_of_limits) / len(df) * 100, 1)) + "%)")
print("Trajectories with discontinuities larger than 350m:", len(discontinuous_trajectories), "(" + str(round(len(discontinuous_trajectories) / len(df) * 100, 1)) + "%)")
print("Trajectories with outliers further away than 700m:", len(trajectories_with_outliers), "(" + str(round(len(trajectories_with_outliers) / len(df) * 100, 1)) + "%)")

trajs_with_outliers_within_limits = df.loc[
    (df['LENGTH_TOO_SHORT'] == 0) &
    (df['LENGTH_TOO_LONG'] == 0) &
    (df['DISCONTINUITIES_350m'] == 2) &
    (df['DISCONTINUITIES_700m'] == 0)
]
print("Trajectories 2 discontinuities between 350m and 700m:", len(trajs_with_outliers_within_limits), "(" + str(round(len(trajs_with_outliers_within_limits) / len(df) * 100, 2)) + "%)")

trajs_with_discontinuities_within_limits = df.loc[
    (df['LENGTH_TOO_SHORT'] == 0) &
    (df['LENGTH_TOO_LONG'] == 0) &
    (df['DISCONTINUITIES_700m'] > 1)
]
print("Trajectories with discontinuities larger than 700m:", len(trajs_with_discontinuities_within_limits), "(" + str(round(len(trajs_with_discontinuities_within_limits) / len(df) * 100, 2)) + "%)")

filered_df = df.loc[(df['LENGTH_TOO_SHORT'] == 0) & (df['LENGTH_TOO_LONG'] == 0) & (df['DISCONTINUITIES_350m'] != 2) & (df['DISCONTINUITIES_700m'] < 2)]

filered_df.to_csv("../../data/filtered_train.csv")