from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import os
import logging
from ast import literal_eval

from config import global_vars
from config import random_seed
from config.args_config import args
from . import file_loader
from utils.timefeatures import time_features
from data_provider.file_loader import file_loader

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

from tqdm import tqdm

import ast


def _normalize_int_sequence(sequence):
    return [int(value) for value in sequence]


from bisect import bisect_left


DELTA_TIME_BUCKET_BOUNDS = [
    5,
    10,
    30,
    60,
    120,
    300,
    600,
    1200,
]


def _compute_delta_times(timestamps):
    timestamps = [int(x) for x in timestamps]

    if len(timestamps) == 0:
        return []

    delta_times = [0]

    for i in range(1, len(timestamps)):
        delta_t = timestamps[i] - timestamps[i - 1]

        if delta_t < 0:
            delta_t = 0

        delta_times.append(delta_t)

    return delta_times


def _bucketize_delta_times(delta_times):
    bucket_ids = []

    for delta_t in delta_times:
        delta_t = int(delta_t)

        if delta_t <= 0:
            bucket_ids.append(0)
        else:
            bucket_id = bisect_left(
                DELTA_TIME_BUCKET_BOUNDS,
                delta_t,
            ) + 1

            bucket_ids.append(bucket_id)

    return bucket_ids


def _build_delta_time_buckets(timestamps):
    """
    timestamps -> delta_times -> bucket_ids
    """
    delta_times = _compute_delta_times(timestamps)
    delta_time_buckets = _bucketize_delta_times(delta_times)
    return delta_times, delta_time_buckets

def _build_gap_samples_for_one_sample(
        sample_id,
        observed_nodes,
        timestamps,
        target_nodes,
        keep_empty_gap=False,
):
    
    observed_nodes = _normalize_int_sequence(observed_nodes)
    timestamps = _normalize_int_sequence(timestamps)
    target_nodes = _normalize_int_sequence(target_nodes)

    if not observed_nodes:
        raise ValueError(
            f"Empty observed sequence for sample_id={sample_id}"
        )

    if not target_nodes:
        raise ValueError(
            f"Empty target trajectory for sample_id={sample_id}"
        )

    node_to_positions = {}
    for pos, node in enumerate(target_nodes):
        node_to_positions.setdefault(node, []).append(pos)

    anchors = []
    last_target_pos = -1

    for obs_idx, node in enumerate(observed_nodes):
        if node not in node_to_positions:
            continue

        matched_target_pos = None

        for pos in node_to_positions[node]:
            if pos > last_target_pos:
                matched_target_pos = pos
                break

        if matched_target_pos is None:
            continue

        anchors.append({
            "obs_idx": obs_idx,
            "node": node,
            "timestamp": timestamps[obs_idx],
            "target_pos": matched_target_pos,
        })

        last_target_pos = matched_target_pos

    gap_samples = []

    for gap_id in range(len(anchors) - 1):
        left = anchors[gap_id]
        right = anchors[gap_id + 1]

        start_pos = left["target_pos"]
        end_pos = right["target_pos"]

        if end_pos <= start_pos:
            continue

        middle_nodes = target_nodes[start_pos + 1:end_pos]

        if len(middle_nodes) == 0 and not keep_empty_gap:
            continue

        gap_timestamps = timestamps[
            left["obs_idx"]:right["obs_idx"] + 1
        ]

        gap_delta_times, gap_delta_time_buckets = _build_delta_time_buckets(
            gap_timestamps
        )

        gap_samples.append({
            "sample_id": int(sample_id),
            "gap_id": int(gap_id),

            "observed_nodes": observed_nodes,
            "timestamps": timestamps,

            "gap_observed_nodes": observed_nodes[
                left["obs_idx"]:right["obs_idx"] + 1
            ],
            "gap_timestamps": gap_timestamps,

            "gap_start_node": int(left["node"]),
            "gap_end_node": int(right["node"]),
            "gap_start_time": int(left["timestamp"]),
            "gap_end_time": int(right["timestamp"]),
            "gap_delta_time": int(
                right["timestamp"] - left["timestamp"]
            ),

            "gap_delta_times": gap_delta_times,
            "gap_delta_time_buckets": gap_delta_time_buckets,

            "gap_target_nodes": middle_nodes,

            "full_target_nodes": target_nodes,
            "start_target_pos": int(start_pos),
            "end_target_pos": int(end_pos),
        })

    return gap_samples


class DatasetTrajectoryRecovery(Dataset):

    def __init__(self):
        super().__init__()

        logging.info(
            "Start loading trajectory recovery dataset."
        )

        self.raw_sample_ids = file_loader.get_sample_ids()
        self.raw_dataset_len = len(self.raw_sample_ids)

        if self.raw_dataset_len == 0:
            raise ValueError(
                "No trajectory recovery samples were found."
            )

        self.gap_samples = []

        self.timestamps_by_sample = {}
        self.delta_times_by_sample = {}
        self.delta_time_buckets_by_sample = {}

        print("raw_sample_ids:", len(self.raw_sample_ids))

        for sample_id in self.raw_sample_ids:
            observed_nodes = _normalize_int_sequence(
                file_loader.get_observed_nodes(sample_id)
            ) 

            target_nodes = _normalize_int_sequence(
                file_loader.get_target_nodes(sample_id)
            )

            timestamps = _normalize_int_sequence(
                file_loader.get_tms(sample_id)
            )

            if len(observed_nodes) != len(timestamps):
                raise ValueError(
                    f"Length mismatch for sample_id={sample_id}: "
                    f"len(observed_nodes)={len(observed_nodes)}, "
                    f"len(timestamps)={len(timestamps)}"
                )

            delta_times, delta_time_buckets = _build_delta_time_buckets(
                timestamps
            )

            self.timestamps_by_sample[sample_id] = timestamps
            self.delta_times_by_sample[sample_id] = delta_times
            self.delta_time_buckets_by_sample[sample_id] = (
                delta_time_buckets
            )

            gap_samples = _build_gap_samples_for_one_sample(
                sample_id=sample_id,
                observed_nodes=observed_nodes,
                timestamps=timestamps,
                target_nodes=target_nodes,
                keep_empty_gap=True,
            )

            self.gap_samples.extend(gap_samples)

        self.dataset_len = len(self.gap_samples)
        print("dataset_len:", len(self.gap_samples))

        logging.info(
            f"(DatasetTrajectoryRecovery) "
            f"Raw samples: {self.raw_dataset_len}, "
            f"Gap samples: {self.dataset_len}, "
        )

        logging.info(
            "Finish loading trajectory recovery dataset."
        )

    def __getitem__(self, index):
        gap_sample = self.gap_samples[index]

        sample_id = gap_sample["sample_id"]
        gap_id = gap_sample["gap_id"]

        observed_nodes = gap_sample["observed_nodes"]
        timestamps = gap_sample["timestamps"]

        gap_target_nodes = gap_sample["gap_target_nodes"]

        if not observed_nodes:
            raise ValueError(
                f"Empty observed sequence for sample_id={sample_id}, "
                f"gap_id={gap_id}"
            )


        return {
            "sample_id": sample_id,
            "gap_id": gap_id,

            "observed_nodes": observed_nodes,
            "timestamps": timestamps,

            "gap_observed_nodes": gap_sample["gap_observed_nodes"],
            "gap_timestamps": gap_sample["gap_timestamps"],

            "gap_start_node": gap_sample["gap_start_node"],
            "gap_end_node": gap_sample["gap_end_node"],
            "gap_start_time": gap_sample["gap_start_time"],
            "gap_end_time": gap_sample["gap_end_time"],
            "gap_delta_time": gap_sample["gap_delta_time"],

            "gap_delta_times": gap_sample["gap_delta_times"],
            "gap_delta_time_buckets": gap_sample[
                "gap_delta_time_buckets"
            ],

            "gap_target_nodes": gap_target_nodes,

            "full_target_nodes": gap_sample["full_target_nodes"],
            "start_target_pos": gap_sample["start_target_pos"],
            "end_target_pos": gap_sample["end_target_pos"],
        }

    def __len__(self):
        return self.dataset_len


# class DatasetKG(Dataset):
#     def __init__(self, input_ids=None, attention_mask=None):
#         super().__init__()
#         self.data = {
#             "input_ids": input_ids,
#             "att_mask": attention_mask,
#         }

#     def __len__(self):
#         return self.data["input_ids"].size(0)

#     def __getitem__(self, index):
#         if isinstance(index, torch.Tensor):
#             index = index.item()
#         batch_data = dict()
#         for key in self.data.keys():
#             if self.data[key] is not None:
#                 batch_data[key] = self.data[key][index]
#         return batch_data




class DatasetTrajectoryRecoveryWithDirection(DatasetTrajectoryRecovery):
    """
    Extends DatasetTrajectoryRecovery to include image direction info.
    Loads direction labels from CSV and attaches to each gap sample.
    """

    def __init__(self):
        logging.info("Initializing DatasetTrajectoryRecoveryWithDirection")
        super().__init__()

        direction_label_csv = global_vars.direction_label_file

        import pandas as pd
        df = pd.read_csv(direction_label_csv)
        logging.info(f"Loaded {len(df)} direction label entries")

        self._dir_lookup = {}
        self._img_lookup = {}
        for _, row in df.iterrows():
            sid = int(row["sample_id"])
            oidx = int(row["obs_index"])
            self._dir_lookup.setdefault(sid, {})[oidx] = int(row["heading_bin"])
            self._img_lookup.setdefault(sid, {})[oidx] = str(row["image_name"])

        augmented_gaps = []
        for gap in self.gap_samples:
            sid = int(gap["sample_id"])
            num_obs = len(gap["observed_nodes"])

            image_names = []
            direction_bins = []
            for obs_idx in range(num_obs):
                dir_entry = self._dir_lookup.get(sid, {}).get(obs_idx, -1)
                img_entry = self._img_lookup.get(sid, {}).get(obs_idx, "")
                direction_bins.append(dir_entry)
                image_names.append(img_entry)

            gap["image_names"] = image_names
            gap["direction_bins"] = direction_bins

            gap_obs_set = set(int(x) for x in gap["gap_observed_nodes"])
            gap_image_names = []
            gap_direction_bins = []
            for obs_idx in range(num_obs):
                node_val = int(gap["observed_nodes"][obs_idx])
                if node_val in gap_obs_set:
                    gap_image_names.append(image_names[obs_idx])
                    gap_direction_bins.append(direction_bins[obs_idx])

            gap["gap_image_names"] = gap_image_names
            gap["gap_direction_bins"] = gap_direction_bins
            augmented_gaps.append(gap)

        self.gap_samples = augmented_gaps
        logging.info(f"Augmented {len(self.gap_samples)} gap samples with direction info")

    def __getitem__(self, index):
        gap_sample = self.gap_samples[index]

        return {
            "sample_id": gap_sample["sample_id"],
            "gap_id": gap_sample["gap_id"],

            "observed_nodes": gap_sample["observed_nodes"],
            "timestamps": gap_sample["timestamps"],

            "gap_observed_nodes": gap_sample["gap_observed_nodes"],
            "gap_timestamps": gap_sample["gap_timestamps"],

            "gap_start_node": gap_sample["gap_start_node"],
            "gap_end_node": gap_sample["gap_end_node"],
            "gap_start_time": gap_sample["gap_start_time"],
            "gap_end_time": gap_sample["gap_end_time"],
            "gap_delta_time": gap_sample["gap_delta_time"],

            "gap_delta_times": gap_sample["gap_delta_times"],
            "gap_delta_time_buckets": gap_sample[
                "gap_delta_time_buckets"
            ],

            "gap_target_nodes": gap_sample["gap_target_nodes"],

            "full_target_nodes": gap_sample["full_target_nodes"],
            "start_target_pos": gap_sample["start_target_pos"],
            "end_target_pos": gap_sample["end_target_pos"],

            # Direction fields
            "image_names": gap_sample["image_names"],
            "direction_bins": gap_sample["direction_bins"],
            "gap_image_names": gap_sample["gap_image_names"],
            "gap_direction_bins": gap_sample["gap_direction_bins"],
        }
