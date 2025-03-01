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
    grouped = df.groupby('agent').cumcount()
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