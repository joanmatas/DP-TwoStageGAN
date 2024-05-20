import pandas as pd
import json
from geopy.distance import geodesic

data = pd.read_csv("../../data/train.csv")
data = data[data.MISSING_DATA == False]

# detect invalid trajectories (in new df)
df_length = len(data)
length_too_short = [0] * df_length
length_too_long = [0] * df_length
discontinuities = [0] * df_length
outliers = [0] * df_length
trajectories = [""] * df_length

trip_ids = data['TRIP_ID'].values
polylines = data['POLYLINE'].values

MIN_LENGTH = 10    # (points) => 2.5 minutes
MAX_LENGTH = 500   # (points) => 125 minutes
OUTLIER_LIMIT = 700    # (meters) => 168 km/h
DISCONTINUITY_LIMIT = 350   # (meters) => 84 km/h
MIN_LON = -8.687466
MIN_LAT = 41.123232
MAX_LON = -8.553186
MAX_LAT = 41.237424

for index in range(df_length):
    trajectory = json.loads(polylines[index])
    trajectory_within_limits = []
    for lon, lat in trajectory:
        if lon > MIN_LON and lon < MAX_LON and lat > MIN_LAT and lat < MAX_LAT:
            trajectory_within_limits.append([lon, lat])
    trajectory_length = len(trajectory_within_limits)
    if trajectory_length < MIN_LENGTH:
        length_too_short[index] = MIN_LENGTH - trajectory_length
    if trajectory_length > MAX_LENGTH:
        length_too_long[index] = trajectory_length - MAX_LENGTH
    trajectories[index] = json.dumps(trajectory_within_limits)
    if not trajectory_length:
        continue
    previous_lon = trajectory_within_limits[0][0]
    previous_lat = trajectory_within_limits[0][1]
    discontinuities_count = 0
    outliers_count = 0
    for lon, lat in trajectory_within_limits:
        dist = geodesic((previous_lat, previous_lon), (lat, lon)).meters
        previous_lon = lon
        previous_lat = lat
        if dist > DISCONTINUITY_LIMIT:
            discontinuities_count += 1
        if dist > OUTLIER_LIMIT:
            outliers_count += 1
    discontinuities[index] = discontinuities_count
    outliers[index] = outliers_count

d = { 
        'TRIP_ID': trip_ids,
        'POLYLINE' : trajectories,
        'LENGTH_TOO_SHORT' : length_too_short,
        'LENGTH_TOO_LONG' : length_too_long,
        'DISCONTINUITIES' : discontinuities,
        'OUTLIERS' : outliers
    }

df = pd.DataFrame(d)
df.to_csv("../../data/train_with_filter_parameters_350_700_within_limits.csv")