<h1 align="center">Dyn<span style="color: green;">OPETs</span>: A Versatile Benchmark for Dynamic <span style="color: green;">O</span>bject <span style="color: green;">P</span>ose <span style="color: green;">E</span>stimation and <span style="color: green;">T</span>racking in Moving Camera <span style="color: green;">S</span>enarios</h1>

<p align="center" style="font-size: larger;">
Xiangting Meng* · Jiaqi Yang* · Mingshu Chen · Chenxin Yan · Yujiao Shi · Wenchao Ding · Laurent Kneip
</p>
<p align="center" style="font-size: larger;">
* Equal Contribution
</p>

<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/pdf/2503.19625"><strong>Paper</strong></a> | 
  <a href="https://www.youtube.com/watch?v=hCOwqutWoLI"><strong>Video</strong></a> | 
  <a href="https://stay332.github.io/DynOPETs"><strong>Project Page</strong></a>
</p>
<p align="center">
  <img src="assets/demo.gif" alt="DynOPETs Demo GIF" width="85%">
</p>
<!-- 🚧 Dataset coming soon! We are actively organizing and refining the data for public release. Stay tuned! -->

## Updates
- [June 23, 2025] 🎉 DynOPETs is now open source! We also provide visualization tools for rendering object models to facilitate learning and usage.


---
## DynOPETs Datasets Overview

<p align="center">
  <img src="assets/cover.png" alt="DynOPETs Cover Image" width="95%">
</p>

**DynOPETs** DynOPETs is a real-world RGB-D dataset designed for object pose estimation and tracking in dynamic scenes with moving cameras.

DynOPETs is split into two complementary subsets:
### COPE119
119 sequences covering 6 common categories from the COPE benchmark: bottles, bowls, cameras, cans, laptops, mugs.
Designed for COPE (Category-level Pose Estimation) methods.

**[Download](https://drive.google.com/drive/folders/1v2gDBpawSnMnq5O3b12ePfEp1MyJ4TwK?usp=drive_link) (41.42GB)** [Training Set](https://drive.google.com/drive/folders/1v2gDBpawSnMnq5O3b12ePfEp1MyJ4TwK?usp=drive_link) / [Test Set](https://drive.google.com/drive/folders/1v2gDBpawSnMnq5O3b12ePfEp1MyJ4TwK?usp=drive_link)
### UOPE56
56 sequences of unconstrained household objects. Tailored for UOPE (Unseen Object Pose Estimation) methods.

**[Download](https://drive.google.com/drive/folders/1wYSwy-MKwDPFuEDXHD0GnnfS7Bg6Ft85?usp=drive_link) (18.24GB)**

---

## DynOPETs Toolbox

### Environment Preparation 
```
conda create -n dynopets python=3.10
conda activate dynopets
pip install -r requirements.txt
```

### Object Pose Visualization Tool
```
python visualize_pose.py 
```

### Point Cloud Generation Tool

```
python generate_point_cloud.py 
```



## Annotation Pipeline

<p align="center">
  <img src="assets/pipeline.png" alt="Annotation Pipeline" width="95%">
</p>

---

## Dataset Format

```
DynOPETs
    ├── COPE119
    │     ├── bottle (30 sequences)
    │     ├── bowl (21 sequences)
    │     ├── camera (9 sequences)
    │     ├── can (22 sequences)
    │     ├── laptop (10 sequences)
    │     └── mug (25 sequences)
    │     └── groundtruth
    └── UOPE56
          └── others (56 sequences)
          └── groundtruth

bottle (example)
  ├── bottle_05
  │     ├── cam_annotations 
  │     │         ├── gripper2base.npy
  │     │         └── gripper_tstamps.npy
  │     ├── color (0000_color.png ...)
  │     ├── depth (0000_depth.png ...)
  │     ├── mask  (0000_mask.png ...)
  │     └── model 
  │           ├── Model.jpg
  │           ├── Model.mtl
  │           └── Model.obj
 ...
```
### COPE119 Training Set
```
- 'bottle_00', 'bottle_05', 'bottle_06', 'bottle_08', 'bottle_09', 'bottle_10','bottle_12', 'bottle_14', 'bottle_15', 'bottle_16', 'bottle_17', 'bottle_18', 'bottle_19', 'bottle_20', 'bottle_21', 'bottle_22', 'bottle_24', 'bottle_25', 'bottle_26', 'bottle_27', 'bottle_28', 'bottle_29',

- 'bowl_00', 'bowl_01', 'bowl_02', 'bowl_04', 'bowl_05', 'bowl_07', 'bowl_08', 'bowl_09', 'bowl_12', 'bowl_14', 'bowl_15', 'bowl_16', 'bowl_17', 'bowl_19', 'bowl_20',

- 'camera_00', 'camera_02', 'camera_03', 'camera_06', 'camera_07', 'camera_08',

- 'can_00', 'can_02', 'can_04', 'can_05', 'can_06', 'can_07', 'can_08', 'can_10', 'can_11', 'can_12', 'can_16', 'can_17', 'can_18', 'can_19', 'can_21',

- 'laptop_01', 'laptop_02', 'laptop_03', 'laptop_05', 'laptop_07', 'laptop_08', 'laptop_10', 'laptop_11',

- 'mug_00', 'mug_02', 'mug_04', 'mug_05', 'mug_07', 'mug_09', 'mug_10', 'mug_13', 'mug_15', 'mug_16', 'mug_17', 'mug_18', 'mug_19', 'mug_20', 'mug_21', 'mug_22', 'mug_23', 'mug_24'
```

### COPE119 Test Set
```
- 'bottle_02', 'bottle_03', 'bottle_04', 'bottle_05', 'bottle_08', 'bottle_12', 'bottle_14', 'bottle_24',

- 'bowl_04', 'bowl_07', 'bowl_11', 'bowl_12', 'bowl_14', 'bowl_19',

- 'camera_02', 'camera_05', 'camera_06',

- 'can_02', 'can_04', 'can_10', 'can_14', 'can_15', 'can_16', 'can_21',

- 'laptop_01', 'laptop_05', 'laptop_07', 'laptop_10',

- 'mug_02', 'mug_04', 'mug_07', 'mug_09', 'mug_12', 'mug_13', 'mug_15'
```

## Devices & Softwares

- RGB-D Camera: [Structure Sensor](https://structure.io/) with iPad Pro

  We developed a **Real-time Visualization & Record App** based on the official SDK that displays and records **RGB, Depth, and Normal images** when connected, making it convenient for everyone to develop and use.

  
  For detailed usage instructions of this software, please refer to the tutorial in the repository below:

  [**Structure Sensor Record App**](https://github.com/Launch-on-Titania/Structure-Sensor-Record)


- Motion Capture System (for camera pose only): [OptiTrack](https://optitrack.com/) Motive

- Object Model Scanner: [Scanner-Structure SDK](https://apps.apple.com/us/app/scanner-structure-sdk/id891169722?platform=ipad) with iPad Pro
  
  We used the **Scanner-Structure SDK** that comes with Structure Sensors to achieve precise **3D Object Asset Scanning**.


## Contact
If you have any questions, please feel free to contact us:

[Xiangting Meng](https://github.com/Launch-on-Titania): [mengxt@shanghaitech.edu.cn](mailto:mengxt@shanghaitech.edu.cn), [arnoxtmann@gmail.com](mailto:arnoxtmann@gmail.com)

[Jiaqi Yang](https://github.com/Jiaqi-Yang): [yangjq1202@shanghaitech.edu.cn](mailto:yangjq1202@shanghaitech.edu.cn)

## Citation
```bibtex
@article{meng2025dynopets,
  title={DynOPETs: A Versatile Benchmark for Dynamic Object Pose Estimation and Tracking in Moving Camera Scenarios},
  author={Meng, Xiangting and Yang, Jiaqi and Chen, Mingshu and Yan, Chenxin and Shi, Yujiao and Ding, Wenchao and Kneip, Laurent},
  journal={arXiv preprint arXiv:2503.19625},
  year={2025}
}
```

## License
This project is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. See [LICENSE](LICENSE) for additional details.