# -*- coding: utf-8 -*-
import os
import logging
import traceback
import numpy as np
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from config.logging_config import init_logger, make_log_dir
from config.args_config import args

from data_provider.file_loader import file_loader
from data_provider.data_loader import DatasetTrajectoryRecovery, DatasetTrajectoryRecoveryWithDirection
from data_provider.kg_loader import load_kg_data_for_model

from models.trajkstfinetune import TrajKSTFineTune

from utils.tools import EarlyStopping
from utils.round_iterator import RoundRobinIterator
# from utils.plot_losses import save_loss_image, save_losses_to_csv

# from transformers import GPT2Tokenizer

import torch.nn.functional as F
from collections import defaultdict

from utils.device import setup_device
from config import global_vars




losses = {
    "traj_recover": [],
}

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



def collect_direction_image_paths(dataset, captures_root):
    """
    Collect all non-empty image paths used by direction module.
    Prefer dataset.gap_samples to avoid calling __getitem__ repeatedly.
    """
    image_paths = set()

    if hasattr(dataset, "gap_samples"):
        for gap in dataset.gap_samples:
            for fname in gap.get("image_names", []):
                if fname and str(fname).strip():
                    image_paths.add(os.path.join(captures_root, str(fname)))
    else:
        for i in range(len(dataset)):
            item = dataset[i]
            for fname in item.get("image_names", []):
                if fname and str(fname).strip():
                    image_paths.add(os.path.join(captures_root, str(fname)))

    return sorted(image_paths)


def train(device):
    file_loader.load_all()
    kg_data = load_kg_data_for_model(device=device)

    use_direction_file = global_vars.direction_label_file
    use_direction = os.path.exists(use_direction_file) and args.dir_loss_weight > 0

    if os.path.exists(use_direction_file) and args.dir_loss_weight > 0 :
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
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)
        for name, dataset in datasets.items()
    }

    
    trajkst = TrajKSTFineTune(
        device=device,
        checkpoint=None,
        freeze_st_tokenizer=False,
        kg_data=kg_data,
        use_direction=use_direction,
    ).to(device)

    if use_direction and args.llm_backbone == "qwen_vl":
        direction_dataset = datasets["traj_recover"]

        all_image_paths = collect_direction_image_paths(
            direction_dataset,
            trajkst.captures_root,
        )

        logging.info(
            f"Preloading Qwen image feature bank, images={len(all_image_paths)}"
        )

        trajkst.backbone.preload_image_feature_bank(
            all_image_paths,
            strict=True,
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, trajkst.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight,
    )

    steps_per_epoch = sum(len(loader) for loader in dataloaders.values())
    total_training_steps = steps_per_epoch * args.train_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.1),  # 10%  
        num_training_steps=total_training_steps  
    )

    early_stopping = EarlyStopping("finetune", patience=args.patience, verbose=True)

    losses = {
        name: []
        for name in datasets.keys()
    }

    global_step = 0
    for epoch in range(1, args.train_epochs + 1):
        trajkst.train()

        epoch_gap_cls_losses = []
        epoch_gap_cls_accs = []
        epoch_kg_losses = []
        epoch_losses = {
            name: []
            for name in datasets.keys()
        }

        iterator = RoundRobinIterator(dataloaders)

        progress_bar = tqdm(
            enumerate(iterator),
            total=len(iterator),
            unit="batch"
        )

        total_loss = 0.0
        num_batches = 0


        for batch_idx, (task_name, batch) in progress_bar:
            progress_bar.set_description(
                f"Epoch {epoch}/{args.train_epochs} - Task: {task_name: <18}"
            )
            global_step += 1

            batch_sample_id = batch["sample_ids"].to(device)
            batch_gap_id = batch["gap_ids"]

            batch_observed_nodes = batch["observed_nodes"]
            batch_timestamps = batch["timestamps"]

            batch_gap_observed_nodes = batch["gap_observed_nodes"]
            batch_gap_timestamps = batch["gap_timestamps"]

            batch_target_nodes = batch["target_nodes"]
            batch_full_target_nodes = batch["full_target_nodes"]
            batch_gap_start_nodes = batch["gap_start_nodes"]
            batch_gap_end_nodes = batch["gap_end_nodes"]

            batch_gap_delta_time = batch["gap_delta_time"]
            batch_gap_delta_times = batch["gap_delta_times"]
            batch_gap_delta_time_buckets = batch["gap_delta_time_buckets"]

            # Forward pass (loss has been calculated)
            optimizer.zero_grad(set_to_none=True)

            # Prepare direction kwargs if using direction module
            dir_kwargs = {}
            if hasattr(trajkst, 'use_direction') and trajkst.use_direction:
                dir_kwargs['batch_image_names'] = batch.get('image_names', None)
                dir_kwargs['batch_direction_bins'] = batch.get('direction_bins', None)
                dir_kwargs['batch_observed_nodes_for_dir'] = batch.get('observed_nodes', None)

            output = trajkst(
                task_name, batch_sample_id, batch_observed_nodes, batch_target_nodes,
                batch_timestamps, batch_gap_start_nodes, batch_gap_end_nodes, batch_gap_delta_time,
                **dir_kwargs,
            )


            loss = output['loss']

            # Record loss
            if torch.isnan(loss):
                raise RuntimeError(
                    f"NaN loss at epoch={epoch}, batch_idx={batch_idx}, task={task_name}"
                )

            loss.backward()
        
            # Extract gap classifier metrics
            gap_cls_loss_val = output['gap_cls_loss'].detach().item()
            gap_cls_logits = output['gap_cls_logits']
            gap_cls_labels = output['gap_cls_labels']
            gap_cls_preds = torch.argmax(gap_cls_logits, dim=-1)
            gap_cls_acc = (gap_cls_preds == gap_cls_labels).float().mean().item()

            kg_loss_val = output['kg_loss'].detach().item()

            
            optimizer.step()
      
            scheduler.step()
           

            loss_value = loss.detach().item()
            losses[task_name].append(loss_value)
            epoch_losses[task_name].append(loss_value)
            epoch_gap_cls_losses.append(gap_cls_loss_val)
            epoch_kg_losses.append(kg_loss_val)
            epoch_gap_cls_accs.append(gap_cls_acc)

            total_loss += loss.item()
            num_batches += 1

            dir_loss_val = output.get('dir_loss', torch.tensor(0.0)).detach().item()
            progress_bar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "gap_cls": f"{gap_cls_loss_val:.4f}",
                "kg":  f"{kg_loss_val:.4f}",
                "dir": f"{dir_loss_val:.4f}",
                "gap_acc": f"{gap_cls_acc:.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6g}",
            })
    
        average_losses = {
            f"{task_name}_epoch_average_loss": float(np.mean(task_losses))
            for task_name, task_losses in epoch_losses.items()
            if len(task_losses) > 0
        }
        epoch_loss = average_losses[f"{task_name}_epoch_average_loss"]
        epoch_average_loss = total_loss / num_batches

        average_losses["learning_rate"] = optimizer.param_groups[0]['lr']
        # Accumulate gap classifier metrics
        average_losses["gap_cls_loss"] = float(np.mean(epoch_gap_cls_losses))
        average_losses["kg_loss"] = float(np.mean(epoch_kg_losses))
        average_losses["gap_cls_acc"] = float(np.mean(epoch_gap_cls_accs))
        logging.info(f"Epoch:{epoch},{average_losses}")

        early_stopping(epoch_average_loss, trajkst, optimizer, epoch)

        if early_stopping.early_stop:
            logging.info("Early stopping triggered.")
            break

def main():
    log_dir = make_log_dir(args.log_path)
    init_logger(log_dir)
    
    device = setup_device(args)
    logging.info(f"Using device: {device}")
    
    try:
        train(device)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        
        logging.info(f"Finishing training.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("\n" + traceback.format_exc())

