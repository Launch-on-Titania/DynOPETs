import os
import sys
import glob
import numpy as np
import _pickle as cPickle
import open3d as o3d
from pathlib import Path
sys.path.append('../lib')
from model_process_utils import sample_points_from_mesh
from tqdm import tqdm

def load_model(model_path):
    points = None
    if model_path.endswith(".obj"):
        mesh = o3d.io.read_triangle_mesh(model_path)
        points = np.asarray(mesh.vertices)
    elif model_path.endswith(".ply"):
        points = np.asarray(o3d.io.read_point_cloud(model_path).points)
    else:
        raise ValueError("Unsupported model format")
    return points

def get_model_scale(model_path):

    model_points = load_model(model_path)
    scales = np.max(np.abs(model_points), axis=0) * 2 
    return scales

def save_nocs_model_to_file(obj_model_dir):
    """ Sampling points from mesh model and normalize to NOCS.
        Models are centered at origin, i.e. NOCS-0.5

    """
    mug_meta = {}
    # used for re-align mug category
    # Real dataset
    for subset in ['real_train']:
        real = {}
        inst_list = glob.glob(os.path.join(obj_model_dir, subset, '*.obj'))
        for inst_path in tqdm(inst_list):
            instance = os.path.basename(inst_path).split('.')[0]
            bbox_file = inst_path.replace('.obj', '.txt')
            bbox_dims = np.loadtxt(bbox_file)
            scale = np.linalg.norm(bbox_dims)
            model_points = sample_points_from_mesh(inst_path, 1024, fps=True, ratio=3)
            model_points /= scale
            # relable mug category
            if 'mug' in instance:
                shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
                shift = np.array([shift_x, 0.0, 0.0])
                model_points += shift
                size = 2 * np.amax(np.abs(model_points), axis=0)
                scale = 1 / np.linalg.norm(size)
                model_points *= scale
                mug_meta[instance] = [shift, scale]
            real[instance] = model_points
        with open(os.path.join(obj_model_dir, '{}.pkl'.format(subset)), 'wb') as f:
            cPickle.dump(real, f)
    # save mug_meta information for re-labeling
    with open(os.path.join(obj_model_dir, 'mug_meta.pkl'), 'wb') as f:
        cPickle.dump(mug_meta, f)

def save_COPE119_model_to_file(data_dir, train_list_dir):
    """ Sampling points from mesh model and normalize to NOCS.
        Models are centered at origin, i.e. NOCS-0.5

    """
    with open(os.path.join(train_list_dir, 'COPE119_train_list.txt'), 'r') as f:
        train_lists = f.readlines()
    mug_meta = {}
    # used for re-align mug category
    # Real dataset
    inst_list = []
    for subset in tqdm(sorted(train_lists)):
        subset = subset.strip()  # filter \n
        inst_list.append(os.path.join(data_dir, subset, 'Model.obj'))

    real = {}
    for inst_path in tqdm(inst_list):
        instance = inst_path.split('/')[-2].split('_')[0]
        instance_id =  inst_path.split('/')[-2]

        scale = get_model_scale(inst_path)
        model_points = sample_points_from_mesh(inst_path, 1024, fps=True, ratio=3)
        model_points /= scale
        # relable mug category
        if 'mug' in instance:
            shift_x = (np.amin(model_points[:, 2]) - np.amax(model_points[:, 2])) / 2 - np.amin(model_points[:, 0])
            shift = np.array([shift_x, 0.0, 0.0])
            model_points += shift
            size = 2 * np.amax(np.abs(model_points), axis=0)
            scale = 1 / np.linalg.norm(size)
            model_points *= scale
            mug_meta[instance] = [shift, scale]
        real[instance_id] = model_points
        
    with open(os.path.join(os.path.join(train_list_dir, 'train_model_pkls'), 'COPE119_train_model.pkl'), 'wb') as f:
        cPickle.dump(real, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to COPE119_Data directory')
    parser.add_argument('--train_list_dir', type=str, required=True, help='Path to Splict_COPE119 directory')
    args = parser.parse_args()
    
    # Save ground truth models for training deform network
    save_COPE119_model_to_file(args.data_dir, args.train_list_dir)
  