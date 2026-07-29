# -*- coding: utf-8 -*-
import os
import torch
import torch.distributed as dist
import pandas as pd
from config.args_config import args

# 数据文件路径

device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
device_id = dist.get_rank() if dist.is_initialized() else -1

city_root_path = os.path.join(args.dataset_path, args.city)

save_root_path = os.path.join(city_root_path, 'test_cache' if args.test_cache else '')
os.makedirs(save_root_path, exist_ok=True)

#新的文件
observe_file = os.path.join(city_root_path, f'data_sim_4k_train.pkl')

# 增加知识图谱数据文件
kg_entity_file = os.path.join(city_root_path, f'kg_entity_{args.city}.csv')     # 实体表
kg_relation_file = os.path.join(city_root_path, f'kg_relation_{args.city}.csv')     # 关系表
kg_triple_file = os.path.join(city_root_path, f'kg_triple_{args.city}.csv')     # 三元组表
cached_kg_nodes = os.path.join(city_root_path, f'cached_kg_node_{args.city}.pt')
cached_kg_edges = os.path.join(city_root_path, f'cached_kg_edge_{args.city}.pt')

# Direction label and image data paths
direction_label_file = os.path.join(city_root_path, "direction_labels_train.csv")
direction_label_test_file = os.path.join(city_root_path, "direction_labels_test.csv")
captures_dir = os.path.join(city_root_path, "captures")
road_network_file = os.path.join(city_root_path, "longhua_1.8k.pkl")

if args.test_cache:
    observe_file = os.path.join(city_root_path, f'data_sim_4k_test.pkl')
    direction_label_file = os.path.join(city_root_path, "direction_labels_test.csv")

