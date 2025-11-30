# COPE119 Finetuning on AG-Pose

This is a customized implementation for finetuning AG-Pose model on COPE119 dataset, based on the official implementation of "Instance-Adaptive and Geometric-Aware Keypoint Learning for Category-Level 6D Object Pose Estimation" (CVPR 2024).

## Original Paper

[[Arxiv](https://arxiv.org/abs/2403.19527)]

## Environment Setup

The code has been tested with:
- Python 3.9
- PyTorch 1.12
- CUDA 11.3

### Installation

```bash
pip install gorilla-core==0.2.5.3
pip install opencv-python

cd model/pointnet2
python setup.py install
```

## Dataset Structure

Prepare your COPE119 dataset with the following structure:

```
COPE119_Data/
├── bottle/
│   ├── bottle_00/
│   │   ├── color/
│   │   │   └── *_color.png
│   │   ├── depth/
│   │   │   └── *_depth.png
│   │   ├── mask/
│   │   │   └── *_mask.png
│   │   └── intrinsics.txt
│   ├── bottle_01/
│   └── ...
├── bowl/
├── camera/
├── can/
├── laptop/
└── mug/
```

Dataset configuration directory:
```
Splict_COPE119/
├── COPE119/
│   ├── COPE119_train_list.txt
│   └── intrinsics.txt
├── train_model_pkls/
│   └── COPE119_train_model.pkl
└── train_label_pkls/
    └── *.pkl
```

## Data Processing

### 1. Generate Model Point Clouds
unzip the train_label.zip in Splict_COPE119

```bash
python model_process.py \
    --data_dir /path/to/COPE119_Data \
    --train_list_dir Splict_COPE119
```

This will generate normalized NOCS model point clouds in `train_model_pkls/COPE119_train_model.pkl`.

## Training

### Finetune on COPE119 Dataset

```bash
python finetune.py \
    --config config/COPE119/COPE119.yaml \
    --pretrained_model checkpoints/pretrained_model.pt \
    --gpus 0
```

Configuration file: `config/COPE119/COPE119.yaml`
- Adjust `dataset_dir` to point to your data directory
- Modify hyperparameters as needed

## Evaluation

Evaluate on a specific category and scene:

```bash
python eval_on_COPE119.py \
    --checkpoint checkpoints/your_model.pt \
    --data_dir /path/to/COPE119_Data \
    --category_id bottle \
    --scene_id 00 \
    --save_path results/bottle_00
```

Or use the provided script:

```bash
bash run_COPE119.sh bottle 00
```

Note: Update the script with your checkpoint and data paths.

## Visualization

Visualize prediction results:

```bash
python vis_results.py \
    --sequence_path /path/to/COPE119_Data/bottle/bottle_00 \
    --result_dir results/bottle_00 \
    --vis_dir visualizations/bottle_00
```

## Configuration Files

### Main Configuration: `config/COPE119/COPE119.yaml`

Key parameters:
- `dataset_name`: COPE119
- `dataset_dir`: Path to COPE119 dataset
- `image_size`: 224
- `sample_num`: 1024 (number of points sampled per object)
- `outlier_th`: 0.14 (outlier threshold for filtering)
- `rgb_backbone`: dino (or resnet)

### Additional Configuration: `config/COPE119/additional_cope119.yaml`

Data loading patterns for different file types.

## Project Structure

```
FinetuneOnAGPose/
├── config/
│   └── COPE119/
│       ├── COPE119.yaml
│       └── additional_cope119.yaml
├── model/
│   ├── Net.py
│   └── pointnet2/
├── provider/
│   ├── nocs_dataset.py
│   └── create_dataloaders.py
├── utils/
│   ├── solver.py
│   └── evaluation_utils.py
├── finetune.py           # Training script
├── eval_on_COPE119.py    # Evaluation script
├── model_process.py      # Data preprocessing script
└── run_COPE119.sh        # Quick evaluation script
```

## Key Modifications from Original AG-Pose

1. **Dataset Adaptation**: Custom dataset loader for COPE119 format
2. **Dynamic Intrinsics Loading**: Reads camera intrinsics from dataset directory
3. **Flexible Path Configuration**: Supports configurable data paths via command line
4. **No Hardcoded Paths**: All paths are parameterized

## Citation

If you use this code, please cite the original AG-Pose paper:

```bibtex
@inproceedings{lin2024instance,
  title={Instance-adaptive and geometric-aware keypoint learning for category-level 6d object pose estimation},
  author={Lin, Xiao and Yang, Wenfei and Gao, Yuan and Zhang, Tianzhu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21040--21049},
  year={2024}
}
```

## Acknowledgements

This implementation is based on [AG-Pose](https://github.com/lly00412/AGPose) and leverages code from:
- [NOCS](https://github.com/hughw19/NOCS_CVPR2019)
- [DualPoseNet](https://github.com/Gorilla-Lab-SCUT/DualPoseNet)
- [DPDN](https://github.com/JiehongLin/Self-DPDN)

## License

This code is released under MIT License.