import logging
import traceback
import numpy as np
from tqdm import tqdm
#import wandb

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, top_k_accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import label_binarize

from config.logging_config import init_logger, make_log_dir
from config.args_config import args

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetTrajectoryRecovery, DatasetTrajectoryRecoveryWithDirection
from data_provider.kg_loader import load_kg_data_for_model

from models.trajkstfinetune import TrajKSTFineTune

from utils.device import setup_device


from collections import defaultdict


import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import os
from config import global_vars


def _lcs_length(a, b):

    n, m = len(a), len(b)
    prev = [0] * (m + 1)

    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        ai = a[i - 1]

        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])

        prev = curr

    return prev[m]




def eval_traj_recover(preds, labels, ignore_ids=None):

    assert len(preds) == len(labels)

    if ignore_ids is None:
        ignore_ids = set()
    else:
        ignore_ids = set(ignore_ids)

    total = len(labels)

    exact_match = 0
    total_token_correct = 0
    total_token_count = 0

    total_pred_len = 0
    total_label_len = 0

    total_lcs_precision = 0.0
    total_lcs_recall = 0.0
    total_lcs_f1 = 0.0

    total_edit_similarity = 0.0

    # 新增：road node set-level 指标
    total_node_precision = 0.0
    total_node_recall = 0.0
    total_node_iou = 0.0

    for pred, label in zip(preds, labels):
        pred = [int(x) for x in pred if int(x) not in ignore_ids]
        label = [int(x) for x in label if int(x) not in ignore_ids]

        pred_len = len(pred)
        label_len = len(label)

        # 1. exact match
        if pred == label:
            exact_match += 1

        # 2. 逐位置 token_acc
        min_len = min(pred_len, label_len)

        for i in range(min_len):
            if pred[i] == label[i]:
                total_token_correct += 1

        total_token_count += label_len

        # 3. 长度统计
        total_pred_len += pred_len
        total_label_len += label_len

        # 4. LCS 指标：允许错位，只要求顺序一致
        if pred_len == 0 and label_len == 0:
            lcs_precision = 1.0
            lcs_recall = 1.0
            lcs_f1 = 1.0
        elif pred_len == 0 or label_len == 0:
            lcs_precision = 0.0
            lcs_recall = 0.0
            lcs_f1 = 0.0
        else:
            lcs_len = _lcs_length(pred, label)

            lcs_precision = lcs_len / pred_len
            lcs_recall = lcs_len / label_len

            if lcs_precision + lcs_recall > 0:
                lcs_f1 = (
                    2 * lcs_precision * lcs_recall
                    / (lcs_precision + lcs_recall)
                )
            else:
                lcs_f1 = 0.0

        total_lcs_precision += lcs_precision
        total_lcs_recall += lcs_recall
        total_lcs_f1 += lcs_f1

        # # 5. 编辑距离相似度
        # if pred_len == 0 and label_len == 0:
        #     edit_similarity = 1.0
        # else:
        #     edit_dist = _edit_distance(pred, label)
        #     edit_similarity = 1.0 - edit_dist / max(pred_len, label_len)

        # total_edit_similarity += edit_similarity

        # 6. 新增：road node set-level Precision / Recall / IoU
        pred_set = set(pred)
        label_set = set(label)

        intersection = pred_set & label_set
        union = pred_set | label_set

        if len(pred_set) == 0 and len(label_set) == 0:
            node_precision = 1.0
            node_recall = 1.0
            node_iou = 1.0
        elif len(pred_set) == 0 or len(label_set) == 0:
            node_precision = 0.0
            node_recall = 0.0
            node_iou = 0.0
        else:
            node_precision = len(intersection) / len(pred_set)
            node_recall = len(intersection) / len(label_set)
            node_iou = len(intersection) / len(union)

        total_node_precision += node_precision
        total_node_recall += node_recall
        total_node_iou += node_iou

    # exact_match_rate = exact_match / total if total > 0 else 0.0

    # token_acc = (
    #     total_token_correct / total_token_count
    #     if total_token_count > 0
    #     else 0.0
    # )

    avg_pred_len = total_pred_len / total if total > 0 else 0.0
    avg_label_len = total_label_len / total if total > 0 else 0.0

    lcs_precision = total_lcs_precision / total if total > 0 else 0.0
    lcs_recall = total_lcs_recall / total if total > 0 else 0.0
    lcs_f1 = total_lcs_f1 / total if total > 0 else 0.0

    # edit_similarity = total_edit_similarity / total if total > 0 else 0.0

    node_precision = total_node_precision / total if total > 0 else 0.0
    node_recall = total_node_recall / total if total > 0 else 0.0
    node_iou = total_node_iou / total if total > 0 else 0.0

    return {
        # "exact_match": exact_match_rate,
        # "token_acc": token_acc,

        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_iou": node_iou,

        "lcs_precision": lcs_precision,
        "lcs_recall": lcs_recall,
        "lcs_f1": lcs_f1,

        # "edit_similarity": edit_similarity,

        "avg_pred_len": avg_pred_len,
        "avg_label_len": avg_label_len,
    }

def _build_full_gap_paths(
        gap_start_nodes,
        gap_end_nodes,
        middle_nodes_list,
):
    full_gap_paths = []

    for start_node, end_node, middle_nodes in zip(
            gap_start_nodes,
            gap_end_nodes,
            middle_nodes_list,
    ):
        full_gap_paths.append(
            [int(start_node)]
            + [int(x) for x in middle_nodes]
            + [int(end_node)]
        )

    return full_gap_paths



def _stitch_gap_middle_nodes_by_sample(
        sample_ids,
        gap_ids,
        gap_start_nodes,
        gap_end_nodes,
        middle_nodes_list,
):
    """
    将多个 gap 结果按 sample_id 拼回轨迹片段。
    """
    grouped = defaultdict(list)

    for sample_id, gap_id, start_node, end_node, middle_nodes in zip(
            sample_ids,
            gap_ids,
            gap_start_nodes,
            gap_end_nodes,
            middle_nodes_list,
    ):
        grouped[int(sample_id)].append({
            "gap_id": int(gap_id),
            "start_node": int(start_node),
            "end_node": int(end_node),
            "middle_nodes": [int(x) for x in middle_nodes],
        })

    stitched_by_sample = {}

    for sample_id, items in grouped.items():
        items = sorted(
            items,
            key=lambda x: x["gap_id"],
        )

        recovered = []

        for item in items:
            start_node = item["start_node"]
            end_node = item["end_node"]
            middle_nodes = item["middle_nodes"]

            if len(recovered) == 0:
                recovered.append(start_node)
            else:
                if recovered[-1] != start_node:
                    recovered.append(start_node)

            recovered.extend(middle_nodes)
            recovered.append(end_node)

        stitched_by_sample[sample_id] = recovered

    return stitched_by_sample



def collate_trajectory_recovery_batch(batch):
    return {
        "sample_ids": torch.tensor(
            [item["sample_id"] for item in batch],
            dtype=torch.long,
        ),
        "gap_ids": [
            item["gap_id"]
            for item in batch
        ],
        "observed_nodes": [
            item["observed_nodes"]
            for item in batch
        ],
        "timestamps": [
            item["timestamps"]
            for item in batch
        ],
        "gap_observed_nodes": [
            item["gap_observed_nodes"]
            for item in batch
        ],
        "gap_timestamps": [
            item["gap_timestamps"]
            for item in batch
        ],
        # "kg_path_nodes": [
        #     item["kg_path_nodes"]
        #     for item in batch
        # ],
        "gap_start_nodes": [
            item["gap_start_node"]
            for item in batch
        ],
        "gap_end_nodes": [
            item["gap_end_node"]
            for item in batch
        ],
        "gap_start_time": [item["gap_start_time"] for item in batch],
        "gap_end_time": [item["gap_end_time"] for item in batch],
        "gap_delta_time": [item["gap_delta_time"] for item in batch],
        "gap_delta_times": [item["gap_delta_times"] for item in batch],
        "gap_delta_time_buckets": [item["gap_delta_time_buckets"] for item in batch],

        "target_nodes": [
            item["gap_target_nodes"]
            for item in batch
        ],

        "full_target_nodes": [
            item["full_target_nodes"]
            for item in batch
        ],
    }


def collate_trajectory_recovery_batch_with_direction(batch):
    """
    Collate function that includes direction label fields alongside
    the standard trajectory recovery fields.
    """
    return {
        "sample_ids": torch.tensor(
            [item["sample_id"] for item in batch],
            dtype=torch.long,
        ),
        "gap_ids": [item["gap_id"] for item in batch],
        "observed_nodes": [item["observed_nodes"] for item in batch],
        "timestamps": [item["timestamps"] for item in batch],
        "gap_observed_nodes": [item["gap_observed_nodes"] for item in batch],
        "gap_timestamps": [item["gap_timestamps"] for item in batch],
        "gap_start_nodes": [item["gap_start_node"] for item in batch],
        "gap_end_nodes": [item["gap_end_node"] for item in batch],
        "gap_start_time": [item["gap_start_time"] for item in batch],
        "gap_end_time": [item["gap_end_time"] for item in batch],
        "gap_delta_time": [item["gap_delta_time"] for item in batch],
        "gap_delta_times": [item["gap_delta_times"] for item in batch],
        "gap_delta_time_buckets": [item["gap_delta_time_buckets"] for item in batch],
        "target_nodes": [item["gap_target_nodes"] for item in batch],
        "full_target_nodes": [item["full_target_nodes"] for item in batch],
        # Direction fields
        "image_names": [item["image_names"] for item in batch],
        "direction_bins": [item["direction_bins"] for item in batch],
        "gap_image_names": [item["gap_image_names"] for item in batch],
        "gap_direction_bins": [item["gap_direction_bins"] for item in batch],
    }


def evaluate(device):

    file_loader.load_all()
    kg_data = load_kg_data_for_model(device=device)
    
    
    use_direction_file = global_vars.direction_label_file
    use_direction = os.path.exists(use_direction_file) and args.dir_loss_weight > 0
    print("use_direction:",use_direction)

    if use_direction:
        datasets = {
            "traj_recover": DatasetTrajectoryRecoveryWithDirection(),
        }
        collate_fn = collate_trajectory_recovery_batch_with_direction
    else:
        datasets = {
            "traj_recover": DatasetTrajectoryRecovery(),
        }
        collate_fn = collate_trajectory_recovery_batch

    dataloaders = {
        name: DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,num_workers=args.num_workers)
        for name, dataset in datasets.items()
    }

    # 评估指标
    eval_funcs = {
        "traj_recover": eval_traj_recover,
    }
    

    trajkst = TrajKSTFineTune(device, checkpoint=args.ckpt,
        freeze_st_tokenizer=True, kg_data=kg_data,
        use_direction=use_direction).to(device)
    trajkst.eval()  # 冻结参数

    print("missing_keys:", [k for k, v in trajkst.state_dict().items() if (v == 0).all() or v.numel() == 0])


    for task_name, dataloader in dataloaders.items():
        logging.info(f"Evaluating task: {task_name}")
        
        progress_bar = tqdm(
            enumerate(dataloader), 
            total=len(dataloader), 
            unit="batch"
        )

        all_preds, all_labels = [], []

        all_pred_middle_nodes = []
        all_label_middle_nodes = []

        all_pred_full_gap_nodes = []
        all_label_full_gap_nodes = []

        all_sample_ids = []
        all_gap_ids = []

        all_gap_start_nodes = []
        all_gap_end_nodes = []

        for batch_idx, (batch) in progress_bar:
            progress_bar.set_description(f"Task: {task_name: <18}")

            batch_sample_id = batch["sample_ids"].to(device)
            batch_gap_id = batch["gap_ids"]

            batch_observed_nodes = batch["observed_nodes"]
            batch_timestamps = batch["timestamps"]

            batch_target_nodes = batch["target_nodes"]
            batch_full_target_nodes = batch["full_target_nodes"]
            batch_gap_start_nodes = batch["gap_start_nodes"]
            batch_gap_end_nodes = batch["gap_end_nodes"]

            batch_gap_delta_time = batch["gap_delta_time"]
            batch_gap_delta_times = batch["gap_delta_times"]
            batch_gap_delta_time_buckets = batch["gap_delta_time_buckets"]

            batch_gap_observed_nodes = batch["gap_observed_nodes"]
            batch_gap_timestamps = batch["gap_timestamps"]



            with torch.no_grad():   
                dir_prob_for_gen=None
                gen_kwargs_eval = {}
                if getattr(trajkst, "use_direction", False):
                    gen_kwargs_eval["batch_image_names"] = batch.get("gap_image_names", None)
                    gen_kwargs_eval["batch_direction_bins"] = batch.get("gap_direction_bins", None)


                batch_pred_middle_nodes = trajkst.generate_trajectory(
                    observed_nodes=batch_observed_nodes,
                    timestamps=batch_timestamps,
                    gap_start_nodes=batch_gap_start_nodes,
                    gap_end_nodes=batch_gap_end_nodes,
                    gap_delta_time=batch_gap_delta_time,
                    max_gen_len=24,
                    min_gen_len=0,
                    dir_prob=dir_prob_for_gen,
                    **gen_kwargs_eval,
                )

                batch_pred_full_gap_nodes = _build_full_gap_paths(
                    gap_start_nodes=batch_gap_start_nodes,
                    gap_end_nodes=batch_gap_end_nodes,
                    middle_nodes_list=batch_pred_middle_nodes,
                )

                batch_label_full_gap_nodes = _build_full_gap_paths(
                    gap_start_nodes=batch_gap_start_nodes,
                    gap_end_nodes=batch_gap_end_nodes,
                    middle_nodes_list=batch_target_nodes,
                )

                all_pred_middle_nodes.extend(batch_pred_middle_nodes)
                all_label_middle_nodes.extend(batch_target_nodes)

                all_pred_full_gap_nodes.extend(batch_pred_full_gap_nodes)
                all_label_full_gap_nodes.extend(batch_label_full_gap_nodes)

                all_sample_ids.extend([int(x) for x in batch_sample_id])
                all_gap_ids.extend([int(x) for x in batch_gap_id])

                all_gap_start_nodes.extend([int(x) for x in batch_gap_start_nodes])
                all_gap_end_nodes.extend([int(x) for x in batch_gap_end_nodes])

        pred_traj_by_sample = _stitch_gap_middle_nodes_by_sample(
            sample_ids=all_sample_ids,
            gap_ids=all_gap_ids,
            gap_start_nodes=all_gap_start_nodes,
            gap_end_nodes=all_gap_end_nodes,
            middle_nodes_list=all_pred_middle_nodes,
        )

        label_traj_by_sample = _stitch_gap_middle_nodes_by_sample(
            sample_ids=all_sample_ids,
            gap_ids=all_gap_ids,
            gap_start_nodes=all_gap_start_nodes,
            gap_end_nodes=all_gap_end_nodes,
            middle_nodes_list=all_label_middle_nodes,
        )

        common_sample_ids = sorted(
            set(pred_traj_by_sample.keys())
            & set(label_traj_by_sample.keys())
        )

        traj_preds = [
            pred_traj_by_sample[sample_id]
            for sample_id in common_sample_ids
        ]

        traj_labels = [
            label_traj_by_sample[sample_id]
            for sample_id in common_sample_ids
        ]

        trajectory_metrics = eval_funcs[task_name](
            traj_preds,
            traj_labels,
        )

        print(f"pred None:{sum(1 for x in all_pred_middle_nodes if not x)}")  
        print(f"label None:{sum(1 for x in all_label_middle_nodes if not x)}")  

        logging.info(
            f"[{task_name}] trajectory metrics: {trajectory_metrics}"
        )


def main():
    
    log_dir = make_log_dir(args.log_path)
    init_logger(log_dir)
    
    device = setup_device(args)
    logging.info(f"Using device: {device}")
    
    try:
        evaluate(device)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        logging.info(f"Saving losses to {log_dir}.")
        
        logging.info(f"Finishing training.")
        

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())