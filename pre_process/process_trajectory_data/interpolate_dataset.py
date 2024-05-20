import pandas as pd
import json
from geopy.distance import geodesic
import shapely

df = pd.read_csv("../../data/filtered_train.csv")

def interpolate(polyline: str) -> str:
    trajectory = json.loads(polyline)
    previous_lon = trajectory[0][0]
    previous_lat = trajectory[0][1]
    point_index = 0
    for lon, lat in trajectory:
        dist = geodesic((previous_lat, previous_lon), (lat, lon)).meters
        if dist > 1050:
            new_points = shapely.line_interpolate_point(shapely.LineString([[previous_lat, previous_lon], [lat, lon]]), [0.25, 0.5, 0.75], normalized=True)
            trajectory.insert(point_index, [new_points[2].y, new_points[2].x])
            trajectory.insert(point_index, [new_points[1].y, new_points[1].x])
            trajectory.insert(point_index, [new_points[0].y, new_points[0].x])
        elif dist > 700:
            new_points = shapely.line_interpolate_point(shapely.LineString([[previous_lat, previous_lon], [lat, lon]]), [1/3, 2/3], normalized=True)
            trajectory.insert(point_index, [new_points[1].y, new_points[1].x])
            trajectory.insert(point_index, [new_points[0].y, new_points[0].x])
        elif dist > 350:
            new_point = shapely.line_interpolate_point(shapely.LineString([[previous_lat, previous_lon], [lat, lon]]), 0.5, normalized=True)
            trajectory.insert(point_index, [new_point.y, new_point.x])
        previous_lon = lon
        previous_lat = lat
        point_index += 1
    return json.dumps(trajectory)

df['POLYLINE'] = df.apply(lambda row: interpolate(row['POLYLINE']) if row['DISCONTINUITIES_350m'] > 0 else row['POLYLINE'], axis=1)

df.to_csv("../../data/train_filtered_interpolated.csv")