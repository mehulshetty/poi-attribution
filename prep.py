import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from pyproj import Proj
import math
import torch
from torch.utils.data import Dataset

# Constants
RAW_SEQ_LEN = 50
PAD = 0
K = 5
DISTANCE_THRESHOLD = 100

# 1. Load the Data
poi_cat_vecs_df = pd.read_parquet("/Users/mehul/Downloads/novateur.phase2.trial4/poi_cat_vecs.parquet")
poi_df = pd.read_csv("/Users/mehul/Downloads/novateur.phase2.trial4/poi.csv")
stay_poi_df = pd.read_parquet("/Users/mehul/Downloads/novateur.phase2.trial4/stay_poi_dfs/group=0/stay_poi.parquet")

# 2. Project Coordinates to UTM

### Borrowed from TrajGPT ###
def get_utm_zone(longitude):
    return int((math.floor((longitude + 180) / 6) % 60) + 1)


def project_latlon_to_xy(df):
    # Get centroid
    lat_c, lon_c = df[['latitude', 'longitude']].mean().values

    # Initialize UTM projection
    zone_number = get_utm_zone(lon_c)
    project_utm = Proj(proj='utm', zone=zone_number, ellps='WGS84')

    # Project centroid to UTM
    utm_x_c, utm_y_c = project_utm(lon_c, lat_c)

    utm_x, utm_y = project_utm(df['longitude'], df['latitude'])
    x = utm_x - utm_x_c  # Unit: meter
    y = utm_y - utm_y_c  # Unit: meter
    return x, y

### 

stay_poi_df['x'], stay_poi_df['y'] = project_latlon_to_xy(stay_poi_df)
poi_df['x'], poi_df['y'] = project_latlon_to_xy(poi_df)

# 3. Calculate Duration (in Hours)
stay_poi_df['start_timestamp'] = pd.to_datetime(stay_poi_df['start_timestamp'])
stay_poi_df['stop_timestamp'] = pd.to_datetime(stay_poi_df['stop_timestamp'])
stay_poi_df['duration'] = (stay_poi_df['stop_timestamp'] - stay_poi_df['start_timestamp']).dt.total_seconds() / 3600

# 4. Assign K nearest POIs from poi_df to each visit in stay_poi_df (create subset for POI attribution)
poi_tree = cKDTree(poi_df[['x', 'y']].values)

def find_candidate_pois(row):
    """Find the K nearest POIs within the distance threshold."""
    distances, indices = poi_tree.query([row['x'], row['y']], k=K, distance_upper_bound=DISTANCE_THRESHOLD)
    valid_indices = indices[distances < DISTANCE_THRESHOLD]
    if len(valid_indices) == 0:
        return np.array([-1])  # Indicates no POIs within threshold
    return poi_df.iloc[valid_indices]['poi_id'].values

stay_poi_df['candidate_pois'] = stay_poi_df.apply(find_candidate_pois, axis=1)

# 5. Map true POI to candidate index if poi_id exists in stay_poi_df
if 'poi_id' in stay_poi_df.columns:
    def get_true_index(row):
        """Map true poi_id to its index in candidate_pois."""
        if row['poi_id'] in row['candidate_pois']:
            return list(row['candidate_pois']).index(row['poi_id'])
        return -1  # True POI not in candidates
    stay_poi_df['true_poi_index'] = stay_poi_df.apply(get_true_index, axis=1)

# 6. Normalize time and compute travel time
oldest_time = stay_poi_df['start_timestamp'].min()
stay_poi_df['start_timestamp'] = (stay_poi_df['start_timestamp'] - oldest_time).dt.total_seconds() / (24 * 3600)  # Days
stay_poi_df['stop_timestamp'] = (stay_poi_df['stop_timestamp'] - oldest_time).dt.total_seconds() / (24 * 3600)  # Days
stay_poi_df['travel_time'] = stay_poi_df['start_timestamp'] - stay_poi_df['stop_timestamp'].shift(1).fillna(0)  # Days
stay_poi_df.loc[stay_poi_df.groupby('agent').head(1).index, 'travel_time'] = 0  # Reset travel time for first visit per user

# 7. Clip outliers
max_duration = stay_poi_df['duration'].quantile(0.99)
stay_poi_df.loc[stay_poi_df['duration'] > max_duration, 'duration'] = max_duration
max_travel_time = stay_poi_df['travel_time'].quantile(0.99) * 24  # Convert to hours for clipping
stay_poi_df.loc[stay_poi_df['travel_time'] * 24 > max_travel_time, 'travel_time'] = max_travel_time / 24

# 8. Sort and split into sequences

### Borrowed from TrajGPT ###
def split_indices_for_next_prediction(df):
    # Identify instances for next visit prediction
    groupby_user = df.groupby('agent')

    # Find the first index of each instance
    # 1. First instance of each user
    user_first_indices = groupby_user.nth(0).index
    # 2. Instance with at least RAW_SEQ_LEN visits
    max_num_visits_per_user = groupby_user.size().max()
    long_enough_indices = groupby_user.nth(list(range(-max_num_visits_per_user, -RAW_SEQ_LEN))).index
    # 3. Combine the two
    first_indices = user_first_indices.union(long_enough_indices)

    # Find the corresponding last indices
    # 1. Last index of each user
    each_user_last_index = groupby_user.nth(-1).set_index('agent')['id']
    # 2. Match it with first_indices
    user_last_indices = df.loc[first_indices]['agent'].apply(lambda user_id: each_user_last_index.loc[user_id]).values
    # 3. Last index of each rolling window
    rolling_window_last_indices = (first_indices + RAW_SEQ_LEN - 1).values
    last_indices = np.minimum(rolling_window_last_indices, user_last_indices)

    # Sort instances by arrival_time
    index_df = pd.DataFrame({'first_index': first_indices, 'last_index': last_indices, 'start_timestamp': df.loc[first_indices]['start_timestamp']})
    index_df.sort_values(by='start_timestamp', inplace=True, ignore_index=True)
    index_df.drop(columns=['start_timestamp'], inplace=True)

    # Split instances into train, validation, and test sets
    train_size = int(0.8 * len(index_df))
    val_size = int(0.1 * len(index_df))
    train_indices = index_df[:train_size].reset_index(drop=True)
    val_indices = index_df[train_size: train_size + val_size].reset_index(drop=True)
    test_indices = index_df[train_size + val_size:].reset_index(drop=True)

    return train_indices, val_indices, test_indices
###

stay_poi_df.sort_values(by=['agent', 'start_timestamp'], inplace=True)
train_indices, val_indices, test_indices = split_indices_for_next_prediction(stay_poi_df)

# 9. Custom dataset classes
class POIPredictionDataset(Dataset):
    def __init__(self, df, indices):
        self.df = df
        self.indices = indices.astype(int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        first_idx, last_idx = self.indices.iloc[idx].values
        instance = self.df.iloc[first_idx:last_idx + 1]
        visits = instance[['x', 'y', 'start_timestamp', 'end_timestamp', 'duration', 'travel_time']].values
        candidate_pois = np.stack(instance['candidate_pois'].values)
        true_poi_indices = instance['true_poi_index'].values if 'true_poi_index' in instance.columns else np.full(len(instance), -1)

        seq_len = visits.shape[0]
        if seq_len < RAW_SEQ_LEN:
            pad_len = RAW_SEQ_LEN - seq_len
            visits = np.pad(visits, ((pad_len, 0), (0, 0)), constant_values=PAD)
            candidate_pois = np.pad(candidate_pois, ((pad_len, 0), (0, 0)), constant_values=-1)
            true_poi_indices = np.pad(true_poi_indices, (pad_len, 0), constant_values=-1)

        return {
            'visits': torch.tensor(visits, dtype=torch.float),         # Visit features
            'candidate_pois': torch.tensor(candidate_pois, dtype=torch.long),  # POI IDs of candidates
            'true_poi_indices': torch.tensor(true_poi_indices, dtype=torch.long)  # Index of true POI in candidates
        }

# Create datasets
train_data = POIPredictionDataset(stay_poi_df, train_indices)
val_data = POIPredictionDataset(stay_poi_df, val_indices)
test_data = POIPredictionDataset(stay_poi_df, test_indices)