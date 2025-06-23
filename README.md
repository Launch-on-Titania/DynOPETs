<h1 align="center">Dyn<span style="color: green;">OPETs</span>: A Versatile Benchmark for Dynamic <span style="color: green;">O</span>bject <span style="color: green;">P</span>ose <span style="color: green;">E</span>stimation and <span style="color: green;">T</span>racking in Moving Camera <span style="color: green;">S</span>enarios</h1>

<p align="center" style="font-size: larger;">
Xiangting Meng* · Jiaqi Yang* · Mingshu Chen · Chenxin Yan · Yujiao Shi · Wenchao Ding · Laurent Kneip
</p>
<p align="center" style="font-size: larger;">
* Equal Contribution
</p>

<p align="center" style="font-size: larger;">
  <a href="https://stay332.github.io/DynOPETs"><strong>Project Page</strong></a> |
  <a href="https://www.youtube.com/watch?v=hCOwqutWoLI"><strong>Video</strong></a> |
  <a href="https://arxiv.org/pdf/2503.19625"><strong>Paper</strong></a> | 
  <a href="https://stay332.github.io/DynOPETs/assets/pdf/supp.pdf"><strong>Supplementary</strong></a>
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
python visulization_tools/runner_rendering.py --show_axis --save_mode video --names "bottle_00, bottle_01"

# Parameters:
 --names, -ns: Comma-separated list of object names to render (default: "bottle_00, bottle_01")
 --output_dir, -od: Output directory for rendered results (default: "results")
 --device, -d: Device to use for rendering (default: "cuda")
 --show_axis: Show coordinate axis on rendered images (flag)
 --save_mode, -sm: Output format - "video" or "images" (default: "video")
 
```
### Visualization Results

<p align="center">
  <table>
    <tr>
      <td align="center">
        <video src="assets/objects/bottle.mp4" alt="Bottle" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Bottle</em>
      </td>
      <td align="center">
        <video src="assets/objects/bowl.mp4" alt="Bowl" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Bowl</em>
      </td>
      <td align="center">
        <video src="assets/objects/camera.mp4" alt="Camera" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Camera</em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <video src="assets/objects/can.mp4" alt="Can" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Can</em>
      </td>
      <td align="center">
        <video src="assets/objects/laptop.mp4" alt="Laptop" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Laptop</em>
      </td>
      <td align="center">
        <video src="assets/objects/mug.mp4" alt="Mug" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Mug</em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <video src="assets/objects/others0.mp4" alt="Others" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Others</em>
      </td>
      <td align="center">
        <video src="assets/objects/others1.mp4" alt="Others" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Others</em>
      </td>
      <td align="center">
        <video src="assets/objects/others2.mp4" alt="Others" width="250px" controls autoplay loop muted>
        </video>
        <br><em>Others</em>
      </td>
    </tr>  </table>
</p>

### Point Cloud Generation Tool

```
python cloud_generater.py --subset COPE119 --seqs bottle_00
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
  ├── bottle_00
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
### COPE119 [Trainset](COPE119_subset/COPE119_trainset_list.txt), [Testset](COPE119_subset/COPE119_testset_list.txt) 




<!-- ```
- bottle (00, 05, 06, 08, 09, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29)

- bowl (00, 01, 02, 04, 05, 07, 08, 09, 12, 14, 15, 16, 17, 19, 20)

- camera (00, 02, 03, 06, 07, 08)

- can (00, 02, 04, 05, 06, 07, 08, 10, 11, 12, 16, 17, 18, 19, 21)

- laptop (01, 02, 03, 05, 07, 08, 10, 11)

- mug (00, 02, 04, 05, 07, 09, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
``` -->

### 
<!-- ```
- bottle (02, 03, 04, 05, 08, 12, 14, 24)

- bowl (04, 07, 11, 12, 14, 19)

- camera (02, 05, 06)

- can (02, 04, 10, 14, 15, 16, 21)

- laptop (01, 05, 07, 10)

- mug (02, 04, 07, 09, 12, 13, 15)
``` -->

## Devices & Softwares

- RGB-D Camera: [Structure Sensor](https://structure.io/) with iPad Pro

  We developed a **Real-time Visualization & Record App** based on the official SDK that displays and records **RGB, Depth, and Normal images** when connected, making it convenient for everyone to develop and use.

  
  For detailed usage instructions of this software, please refer to the tutorial in the repository below:

  [**Structure Sensor Record App**](https://github.com/Launch-on-Titania/Structure-Sensor-Record)

  <p align="center">
    <img src="assets/ss_viewer.jpeg" alt="Structure Sensor Record App" width="90%">
  </p>

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