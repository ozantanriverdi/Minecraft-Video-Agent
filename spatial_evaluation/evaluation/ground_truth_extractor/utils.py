import math
import datetime
import numpy as np
from pathlib import Path


def create_folders(run_id):
    # Get the parent directory (spatial_evaluation)
    base_dir = Path(__file__).parent.parent  # Moves up one level from "sampler"
    
    # Define the directory where samples should be stored
    ground_truths = base_dir / "ground_truths" / run_id

    ground_truths.mkdir(parents=True, exist_ok=True)

    return ground_truths

def calculate_distance(traced_block_x, traced_block_y, traced_block_z, entity_1_idx, entity_2_idx):
    distances = []
    for entity_1_id in entity_1_idx:
        for entity_2_id in entity_2_idx:
            distance = math.sqrt((traced_block_x[entity_1_id] - traced_block_x[entity_2_id]) ** 2 +
                                    (traced_block_y[entity_1_id] - traced_block_y[entity_2_id]) ** 2 +
                                    (traced_block_z[entity_1_id] - traced_block_z[entity_2_id]) ** 2)
            distances.append(distance)
    min_distance = np.min(distances)
    min_distance = round(min_distance, 3)
    return min_distance

