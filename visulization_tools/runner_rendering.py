import cv2
import bz2
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from render_utils import render_obj_to_image, load_obj_from_path, draw_axis_on_image



def main(args):

    names = [n.strip() for n in args.names.split(',') if n.strip()]

    for name in tqdm(names):
        if "others" in name:
            Subset_Name = GT_DATA_BASE_DIR.parent / GT_DATA_BASE_DIR.parts[-1].replace("COPE119", "UOPE56")
            Subset_Pose_Name = GT_POSE_BASE_DIR.parent / GT_POSE_BASE_DIR.parts[-1].replace("COPE119", "UOPE56")
        else:
            Subset_Name = GT_DATA_BASE_DIR
            Subset_Pose_Name = GT_POSE_BASE_DIR
        print(Subset_Name)
        print(Subset_Pose_Name)
        GT_Data_Path = Subset_Name / name.split('_')[0] / name
        GT_Pose_Path = Subset_Pose_Name / f"{name}.pkl.bz2"
        CAD_Model_Path = GT_Data_Path / "model" / "Model.obj"

        obj_mesh = load_obj_from_path(CAD_Model_Path)

        with bz2.BZ2File(GT_Pose_Path, 'rb') as f:
            gt_obj_pose = pickle.load(f)
        num_pose = len(gt_obj_pose)

        image_paths = sorted((GT_Data_Path / "color").glob("*.png"))
        intrinsics = np.loadtxt(GT_Data_Path / "intrinsics.txt")
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]

        intrinsics_mat = np.array([[fx,  0, cx], 
                                [0,  fy, cy], 
                                [0,   0,  1]])

        output_dir = Path(f"{args.output_dir}/{name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_mode = getattr(args, "save_mode", "images")
        video_writer = None
        if save_mode == "video":
            first_img = cv2.imread(str(image_paths[0]))
            height, width = first_img.shape[:2]
            video_path = output_dir / f"{name}_gt_rendering.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))

        for idx in tqdm(range(num_pose), desc=f"Processing frames for {name}"):
            gt_pose = gt_obj_pose[idx]['gt_pose_o2c']
            gt_mat = np.array(gt_pose)

            image = cv2.imread(str(image_paths[idx]))
            intrinsics = intrinsics_mat

            image_size = (image.shape[0], image.shape[1])  # (height, width)

            rendering = render_obj_to_image(
                obj_mesh,
                gt_mat[:3, :3],
                gt_mat[:3, 3],
                intrinsics,
                image_size,
                torch.device(args.device)
            )
            rendering_mask = rendering[...,3].cpu().numpy()
            rendering_mask = (rendering_mask*255.0).astype(np.uint8)
            rendering = rendering[...,:3].cpu().numpy()
            rendering = (rendering*255.0).astype(np.uint8)
            rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
            composite_image = image//2 + rendering//2
            contours, hierarchy = cv2.findContours(rendering_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            composite_image=cv2.drawContours(composite_image, contours, -1, (0, 255, 0), 2) # -1 means draw all contours
            if args.show_axis:
                composite_image = draw_axis_on_image(composite_image, intrinsics, gt_mat)
            
            if save_mode == "images":
                output_path = output_dir / f"{idx:04d}_gt.png"
                cv2.imwrite(str(output_path), composite_image)
            elif save_mode == "video":
                video_writer.write(composite_image)
        
        if video_writer is not None:
            video_writer.release()
        
if __name__ == "__main__":
    
    GT_POSE_BASE_DIR = Path("~/DynOPETs/COPE119/groundtruth/").expanduser()
    GT_DATA_BASE_DIR = Path("~/DynOPETs/COPE119/").expanduser() 
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--names",'-ns', type=str, help="comma(,) separated list of names", default="bottle_00, bottle_01")
    parser.add_argument("--output_dir",'-od', type=str, default="results")
    parser.add_argument("--device",'-d', type=str, default="cuda")
    parser.add_argument("--show_axis", action="store_true")
    parser.add_argument("--save_mode", '-sm' , type=str, help="video or images", default="video")
    args = parser.parse_args()
    
    main(args)