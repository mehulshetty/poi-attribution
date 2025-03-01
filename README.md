# POI-Attribution

## Phase 1: Data Preprocessing and Input Preparation

### Preprocessing Changes

1. We identify the kNNs for our current location to create a shortlist of potential candidates.

2. Duration and travel_time are calculated directly from the data.

3. We combine all the feature to simplify processing.

### Input Representation Changes

1. Since region data isn't important we completely do away with region_id.

2. Travel_time and duration get their own Time2Vec encodings since they might help capture more information about user behavior.

3. We ditch the spatial encodings in TrajGPT for a frequency-based encoding which might help us capture the trends in smaller location and time data better. We also introduce a cosine activation function along with a sinoidal for this purpose.
