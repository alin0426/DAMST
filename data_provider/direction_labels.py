import os
import math
import pickle
import logging

import pandas as pd
import networkx as nx


def load_road_network(network_path):
    with open(network_path, 'rb') as f:
        G = pickle.load(f)

    node_xy = {}
    for node_id, attrs in G.nodes(data=True):
        if 'xy' in attrs:
            node_xy[node_id] = attrs['xy']
        elif 'x' in attrs and 'y' in attrs:
            node_xy[node_id] = [attrs['x'], attrs['y']]
        else:
            logging.warning(f"Node {node_id} has no coordinates")

    logging.info(f"Loaded road network: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return node_xy


def find_next_node_in_traj(obs_node, traj_y):
    for i, node in enumerate(traj_y):
        if node == obs_node and i + 1 < len(traj_y):
            return traj_y[i + 1]
    return None


def compute_heading_bin(x1, y1, x2, y2, num_bins=16):
    dx = x2 - x1
    dy = y2 - y1
    theta = math.atan2(dy, dx)

    bin_size = 2.0 * math.pi / num_bins
    heading_bin = round(theta / bin_size) % num_bins

    return heading_bin, theta


def generate_direction_labels(data_file, network_path, output_csv, num_bins=16):
    logging.info(f"Loading data from {data_file}")
    data = pd.read_pickle(data_file)

    node_xy = load_road_network(network_path)

    sample_ids = sorted(data['cam_x'].keys())
    logging.info(f"Processing {len(sample_ids)} samples")

    records = []
    missing_xy = 0
    missing_next = 0

    for sample_id in sample_ids:
        observed_nodes = [int(x) for x in data['cam_x'][sample_id]]
        traj_y = [int(x) for x in data['traj_y'][sample_id]]
        image_names = [str(x) for x in data['image_name'][sample_id]]
        recd_tokens = [int(x) for x in data['recd_token'][sample_id]]

        for obs_idx, obs_node in enumerate(observed_nodes):
            next_node = find_next_node_in_traj(obs_node, traj_y)

            if next_node is None:
                missing_next += 1
                continue

            if obs_node not in node_xy or next_node not in node_xy:
                missing_xy += 1
                continue

            x1, y1 = node_xy[obs_node]
            x2, y2 = node_xy[next_node]

            heading_bin, theta = compute_heading_bin(x1, y1, x2, y2, num_bins)

            records.append({
                'sample_id': sample_id,
                'obs_index': obs_idx,
                'image_name': image_names[obs_idx] if obs_idx < len(image_names) else '',
                'recd_token': recd_tokens[obs_idx] if obs_idx < len(recd_tokens) else -1,
                'obs_node_id': obs_node,
                'next_node_id': next_node,
                'heading_bin': heading_bin,
                'theta_rad': round(theta, 6),
            })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    logging.info(f"Saved {len(df)} direction labels to {output_csv}")
    logging.info(f"  Missing next node: {missing_next}")
    logging.info(f"  Missing xy coords: {missing_xy}")
    logging.info(f"  Bin distribution:\n{df['heading_bin'].value_counts().sort_index()}")

    return df


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset', 'sz')
    network_path = os.path.join(base_dir, 'longhua_1.8k.pkl')

    num_bins = 16

    # Generate training labels
    train_file = os.path.join(base_dir, 'data_sim_4k_train.pkl')
    train_output = os.path.join(base_dir, 'direction_labels_train.csv')
    generate_direction_labels(train_file, network_path, train_output, num_bins)

    # Generate test labels
    test_file = os.path.join(base_dir, 'data_sim_4k_test.pkl')
    test_output = os.path.join(base_dir, 'direction_labels_test.csv')
    generate_direction_labels(test_file, network_path, test_output, num_bins)


if __name__ == '__main__':
    main()
