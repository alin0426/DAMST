# DAMST
Direction-aware Multi-modality Spatial-Temporal Vehicle Trajectory Recovery for Urban Public Safety


## Runtime Environment
```bash
Hardware: Huawei Ascend 910B2
NPU Memory: 64 GB HBM
NPU Driver: 24.1.0.3
npu-smi: 24.1.0.3
Python: 3.10.16
PyTorch: 2.4.0
torch-npu: 2.4.0.post2
```

## Dataset
[Download](https://drive.google.com/file/d/1KBqFNAkh9T7S1dzyfYj-etXI-_bENl-e/view?usp=sharing)

Raw data from [github](https://github.com/bonaldli/VisionTraj), [dataset](https://drive.google.com/drive/folders/1e5clH_lgFEjJp9AtS8gJlzrRiBksZ896?usp=sharing)

## Finetune
```bash
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0

python finetune.py  --use_gpu  --dataset_path ./dataset/  --city sz  --num_workers 4  --train_epochs 20  --batch_size 16  --learning_rate 1e-4  --llm_backbone qwen_vl  --device 0  --qwen_model_path /root/work/model/Qwen3.5-4B
```
Replace --qwen_model_path with the actual path to the Qwen3.5-VL-4B model parameter files.


## Evaluate
```bash
python evaluate.py  --use_gpu  --dataset_path ./dataset/  --city sz  --llm_backbone qwen_vl  --device 0  --test_cache  --ckpt ./checkpoints/sz_finetune_best.pth  --qwen_model_path /root/work/model/Qwen3.5-4B
```
Replace --qwen_model_path with the actual path to the model parameter files.
