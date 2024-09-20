# DETR for grocery dataset item detection


## Installation Instructions

Create python environment (optional)
```sh
conda create -n grocery python=3.8 -y
conda activate grocery
```

Clone the repository
```sh
git clone https://github.com/dipanjyoti/DETR_grocery.git
cd DETR_grocery
```

Install python dependencies

```sh
pip install -r requirements.txt
```

## Training (fine-tuning)


```sh
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12341  --use_env main.py --coco_path "<path/to/datasets>" --finetune "path/to/detr-r50-e632da11.pth" --num_queries 600 --output_dir 'output' --batch_size 2
```

## Training (from scratch)

```sh
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 12341  --use_env main.py --coco_path "<path/to/datasets>" --num_queries 600 --output_dir 'output' --batch_size 2
```
