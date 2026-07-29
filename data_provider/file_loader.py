import pandas as pd
import logging
import os
import json
import torch
import torch.distributed as dist

from config.args_config import args
from config import global_vars

import networkx as nx
#from torch_geometric.data import HeteroData
from models.layers import Sentence_Transformer
from transformers import AutoModel, AutoTokenizer

from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np


from collections import deque



class FileLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileLoader, cls).__new__(cls)
        

            cls._instance.traj_data = None
            cls._instance.sample_ids = None
            cls._instance.traj_cnt = 0
            cls._instance.node_xy = None
            cls._instance.direction_labels = None

            cls._instance.intersection_ids = None
            cls._instance.road_segments = None
            cls._instance.road_segment_cnt = 0

            cls._instance.road_adj = None
            cls._instance.segment_lookup = None

        return cls._instance

   

    def load_observation_file(self):
        logging.info("Start reading observe data file.")
        traj_data = pd.read_pickle(global_vars.observe_file)
        required_fields = {
            "cam_x",
            "cam_x_high",
            "traj_y",
            "cam_tms",
        }
        missing_fields = required_fields - set(
            traj_data.keys()
        )

        if missing_fields:
            raise KeyError(
                f"Missing trajectory fields: "
                f"{sorted(missing_fields)}"
            )

        observed_ids_high = set(
            traj_data["cam_x_high"].keys()
        )
        observed_ids = set(
            traj_data["cam_x"].keys()
        )
        target_ids = set(
            traj_data["traj_y"].keys()
        )
        tms_ids = set(
            traj_data["cam_tms"].keys()
        )

        if observed_ids != target_ids:
            raise ValueError(
                "cam_x and traj_y have "
                "different sample IDs."
            )

        self.traj_data = traj_data
        self.sample_ids = sorted(target_ids)  # 对应的车辆ID/轨迹ID
        self.traj_cnt = len(self.sample_ids)

        logging.info(
            f"Observation data loaded. "
            f"Sample count: {self.traj_cnt}"
        )

    def load_road_topology(self):
        logging.info(
            f"Start reading KG entity file: "
            f"{global_vars.kg_entity_file}"
        )

        entity_df = pd.read_csv(  # 实体表
            global_vars.kg_entity_file,
            usecols=[
                "entity_id",
                "entity_type",
                "original_id",
                "properties_json",
            ],
        )

        intersection_df = entity_df[  # 路口表
            entity_df["entity_type"] == "Intersection"
            ]

        road_segment_df = entity_df[  # 道路段表
            entity_df["entity_type"] == "RoadSegment"
            ]

        self.intersection_ids = set(  # 路口ID（0~1838）
            intersection_df["original_id"]
            .astype(int)
            .tolist()
        )

        road_adj = {}
        segment_lookup = {}

        # 构建路网拓扑结构
        for row in road_segment_df.itertuples(index=False):
            properties = json.loads(
                row.properties_json
            )  # 解析JSON属性

            u = int(properties["u"])  # 起点
            v = int(properties["v"])  # 终点
            key = int(properties.get("key", 0))

            road_adj.setdefault(u, []).append(v)  # 邻接表，形成 {起点: [终点1, 终点2, ...]} 的邻接关系

            segment_lookup.setdefault(  # 路段查找表，按 (起点, 终点) 为键存储路段详情
                (u, v),
                [],
            ).append(
                {
                    "entity_id": row.entity_id,
                    "original_id": row.original_id,
                    "key": key,
                    "length": properties.get("length"),
                    "highway": properties.get(
                        "highway_raw"
                    ),
                    "name": properties.get(
                        "name_raw"
                    ),
                }
            )

        self.road_adj = {  # 邻接表去重
            node_id: tuple(dict.fromkeys(neighbors))
            for node_id, neighbors in road_adj.items()
        }

        self.segment_lookup = segment_lookup  # 路段查找表

        self.road_segments = road_segment_df  # 路段表
        self.road_segment_cnt = len(
            road_segment_df
        )

        self.intersection_cnt = len(self.intersection_ids)

        logging.info(
            f"Road topology loaded. "
            f"Intersections: {len(self.intersection_ids)}, "
            f"Road segments: {self.road_segment_cnt}"
        )

    def load_all(self, rank=0):
        if self.traj_data is None:
            self.load_observation_file()

        if self.road_adj is None:
            self.load_road_topology()

        logging.info(
            "All data loaded. "
            f"Trajectories: {self.traj_cnt}, "
            f"Intersections: {len(self.intersection_ids)}, "
            f"Road segments: {self.road_segment_cnt}"
        )

    def get_observed_nodes(self, sample_id):
        return [
            int(node_id)
            for node_id
            in self.traj_data[
                "cam_x"
            ][sample_id]
        ]

    def get_observed_nodes_high(self, sample_id):
        return [
            int(node_id)
            for node_id
            in self.traj_data[
                "cam_x_high"
            ][sample_id]
        ]

    def get_target_nodes(self, sample_id):
        return [
            int(node_id)
            for node_id
            in self.traj_data[
                "traj_y"
            ][sample_id]
        ]

    def get_tms(self, sample_id):
        return [
            int(node_id)
            for node_id
            in self.traj_data[
                "cam_tms"
            ][sample_id]
        ]

    def get_sample_ids(self):
        return self.sample_ids

    def get_intersection_cnt(self):
        return self.intersection_cnt


    def get_image_name(self, sample_id):
        """
        Return list of image filenames for observed nodes of a sample.
        """
        return [
            str(name)
            for name in self.traj_data["image_name"][sample_id]
        ]


    def get_image_path(self, image_name, captures_root=None):
        """
        Construct full path to an image file in the captures directory.
        """
        if captures_root is None:
            from config.global_vars import captures_dir as default_dir
            captures_root = default_dir
        import os
        return os.path.join(captures_root, str(image_name))

    def load_road_network(self, network_path=None):
        """
        Load the road network graph and extract node_xy lookup dict.
        Stores self.node_xy: {node_id: [longitude, latitude]}.
        """
        if network_path is None:
            from config.global_vars import road_network_file as default_path
            network_path = default_path

        import pickle
        with open(network_path, "rb") as f:
            G = pickle.load(f)

        self.node_xy = {}
        for node_id, attrs in G.nodes(data=True):
            if "xy" in attrs:
                self.node_xy[node_id] = attrs["xy"]
            elif "x" in attrs and "y" in attrs:
                self.node_xy[node_id] = [attrs["x"], attrs["y"]]

        import logging
        logging.info(f"Loaded road network: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return self.node_xy

    def get_node_xy(self, node_id):
        """
        Return [longitude, latitude] for a road node.
        """
        if self.node_xy is None:
            raise RuntimeError("node_xy not loaded. Call load_road_network() first.")
        return self.node_xy.get(int(node_id), None)

    def load_direction_labels(self, label_csv=None):
        """
        Load direction labels CSV and store as nested dict:
        self.direction_labels[sample_id][obs_idx] = {
            "heading_bin": int,
            "image_name": str,
            "obs_node_id": int,
            ...
        }
        """
        if label_csv is None:
            from config.global_vars import direction_label_train_file as default_csv
            label_csv = default_csv

        import pandas as pd
        df = pd.read_csv(label_csv)
        import logging
        logging.info(f"Loaded direction labels: {len(df)} entries from {label_csv}")

        self.direction_labels = {}
        for _, row in df.iterrows():
            sid = int(row["sample_id"])
            oidx = int(row["obs_index"])
            self.direction_labels.setdefault(sid, {})[oidx] = {
                "heading_bin": int(row["heading_bin"]),
                "image_name": str(row["image_name"]),
                "recd_token": int(row["recd_token"]),
                "obs_node_id": int(row["obs_node_id"]),
                "next_node_id": int(row["next_node_id"]),
                "theta_rad": float(row["theta_rad"]),
            }

        return self.direction_labels

    def get_direction_label(self, sample_id, obs_idx):
        """
        Return heading_bin for a specific observation.
        Returns -1 if label not found.
        """
        if self.direction_labels is None:
            raise RuntimeError("direction_labels not loaded. Call load_direction_labels() first.")
        sample_labels = self.direction_labels.get(int(sample_id), {})
        entry = sample_labels.get(int(obs_idx), {})
        return entry.get("heading_bin", -1)



file_loader = FileLoader()
