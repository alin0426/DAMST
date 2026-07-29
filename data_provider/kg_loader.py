import os
import json
import logging

import numpy as np
import pandas as pd
import torch

from config.args_config import args
from config import global_vars


def _read_csv_safe(filepath, **kwargs):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"KG data file not found: {filepath}")
    return pd.read_csv(filepath, **kwargs)


def load_kg_data(city_root_path, force_reload=False):

    cached_path = os.path.join(city_root_path, "cached_kg_data.pt")

    if not force_reload and os.path.exists(cached_path):
        logging.info(f"Loading cached KG data from {cached_path}")
        return torch.load(cached_path, map_location="cpu", weights_only=False)

    logging.info("Cached KG data not found, loading from CSV files.")

    entity_file = global_vars.kg_entity_file
    relation_file = global_vars.kg_relation_file

    entity_df = _read_csv_safe(entity_file)
    logging.info(f"Loaded entities: {len(entity_df)} rows, "
                 f"columns: {list(entity_df.columns)}")

    entity_df = entity_df.sort_values("entity_id").reset_index(drop=True)
    entity_id2idx = {
        row["entity_id"]: idx
        for idx, row in entity_df.iterrows()
    }
    num_entities = len(entity_df)

    entity_types = entity_df["entity_type"].tolist()

    intersection_df = entity_df[
        entity_df["entity_type"] == "Intersection"
    ]

    road2eidx = {}
    for _, row in intersection_df.iterrows():
        road_id = int(row["original_id"])          
        road2eidx[road_id] = entity_id2idx[row["entity_id"]]

    logging.info(
        f"Entities: {num_entities} total, "
        f"{len(intersection_df)} intersections, "
        f"road2eidx mapping built"
    )

    relation_df = _read_csv_safe(relation_file)
    logging.info(f"Loaded relations: {len(relation_df)} rows, "
                 f"columns: {list(relation_df.columns)}")

    unique_rel_types = sorted(relation_df["relation_type"].unique())
    rel_type2idx = {rt: i for i, rt in enumerate(unique_rel_types)}
    num_rel_types = len(unique_rel_types)  # 27

    logging.info(f"Distinct relation types: {num_rel_types}")

    mask = (
        relation_df["source_entity_id"].isin(entity_id2idx)
        & relation_df["target_entity_id"].isin(entity_id2idx)
    )
    valid_df = relation_df[mask].copy()
    skipped = len(relation_df) - len(valid_df)
    if skipped:
        logging.warning(f"Skipped {skipped} relations with unknown entity IDs")

    source_idx = valid_df["source_entity_id"].map(entity_id2idx).values
    target_idx = valid_df["target_entity_id"].map(entity_id2idx).values

    edge_index = torch.from_numpy(np.stack([source_idx, target_idx])).long()  # [2, E]

    edge_type = torch.from_numpy(valid_df["relation_type"].map(rel_type2idx).values).long()  # [E]

    num_edges = edge_index.size(1)
    logging.info(f"Edge index built: {num_edges} edges, "
                 f"shape {tuple(edge_index.shape)}")

    kg_data = {
        "entity_id2idx": entity_id2idx,
        "num_entities": num_entities,
        "edge_index": edge_index,
        "edge_type": edge_type,
        "num_edge": num_edges,
        "num_rel_types": num_rel_types,
        "road2eidx": road2eidx,
        "rel_type2idx": rel_type2idx,
        "entity_types": entity_types,
    }

    logging.info(f"Caching KG data to {cached_path}")
    torch.save(kg_data, cached_path)

    return kg_data


def load_kg_data_for_model(device="cpu", force_reload=False):
    city_root = os.path.join(args.dataset_path, args.city)
    kg_data = load_kg_data(city_root, force_reload=force_reload)

    kg_data["edge_index"] = kg_data["edge_index"].to(device)
    kg_data["edge_type"] = kg_data["edge_type"].to(device)

    return kg_data
