import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pyproj import Proj
import torch
from torch.utils.data import Dataset

# Constants
RAW_SEQ_LEN = 50
PAD = 0
K = 5
DISTANCE_THRESHOLD = 100

def split_indices_for_next_prediction(df):
    grouped = df.groupby('user_id').cumcount()
    total_len = len(df)
    train_end = int(total_len * 0.7)
    val_end = int(total_len * 0.85)

    train_indices = pd.DataFrame({'first_idx': range(0, train_end), 'last_idx': range(0, train_end)}).iloc[::RAW_SEQ_LEN]
    val_indices = pd.DataFrame({'first_idx': range(train_end, val_end), 'last_index': range(train_end, val_end)}).iloc[::RAW_SEQ_LEN]
    test_indices = pd.DataFrame({'first_idx': range(val_end, total_len), 'last_idx': range(val_end, total_len)}).iloc[::RAW_SEQ_LEN]

    return train_indices, val_indices, test_indices

# 1. Load the Data
poi_cat_vecs_df = pd.read_parquet("/Users/mehul/Downloads/novateur.phase2.trial4/poi_cat_vecs.parquet")
poi_df = pd.read_csv("/Users/mehul/Downloads/novateur.phase2.trial4/poi.csv")
stay_poi_df = pd.read_parquet("/Users/mehul/Downloads/novateur.phase2.trial4/stay_poi_dfs/group=0/stay_poi.parquet")

# 2. Project Coordinates to UTM
def project_to_utm(df, lat_col = 'latitude', lon_col = 'longitude'):
    zone_number = int((df[lon_col].mean() + 180) // 6) + 1
    utm_proj = Proj(proj='utm', zone=zone_number, ellps='WGS84')
    df['x'], df['y'] = utm_proj(df[lon_col].values, df[lat_col].values)
    return df

stay_poi_df = project_to_utm(stay_poi_df, lat_col='latitude', lon_col='longitude')
poi_df = project_to_utm(poi_df, lat_col='latitude', lon_col='longitude')

# 3. Calculate Duration (in Hours)
stay_poi_df['arrival_time'] = pd.to_datetime(stay_poi_df['arrival_time'])
stay_poi_df['departure_time'] = pd.to_datetime(stay_poi_df['departure_time'])
stay_poi_df['duration'] = (stay_poi_df['departure_time'] - stay_poi_df['arrival_time']).dt.total_seconds() / 3600

