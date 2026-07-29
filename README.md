# DAMST
Direction-aware Multi-modality Spatial-Temporal Vehicle Trajectory Recovery for Urban Public Safety


## Operating Environment
```bash
```

## Dataset


## Finetune
```bash
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

python finetune.py  --use_gpu  --dataset_path ./dataset/  --city sz  --num_workers 4  --train_epochs 20  --batch_size 16  --learning_rate 1e-4  --llm_backbone qwen_vl  --device 0
```

## Evaluate
```bash
python evaluate.py  --use_gpu  --dataset_path ./dataset/  --city sz  --llm_backbone qwen_vl  --device 0  --test_cache  --ckpt ./checkpoints/sz_finetune_best.pth
```
