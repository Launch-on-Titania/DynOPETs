import glob
import cv2
import numpy as np
import open3d as o3d
# import _pickle as cPickle
import compress_pickle as pickle
from tqdm import tqdm
from pathlib import Path



def create_img_list(root_dir, subset_list_dir):
    """ Create train/test data list for and COPE119. """
    # COPE119 dataset
    train_list_dir = subset_list_dir / 'COPE119_trainset_list.txt'
    test_list_dir = subset_list_dir / 'COPE119_testset_list.txt'

    # Create train & test subset list
    for subset_type, list_dir in [('train', train_list_dir), ('test', test_list_dir)]:
        if list_dir.exists():
            with open(list_dir, 'r') as f:
                subset_list = f.read().splitlines()
            img_list = []
            for item in subset_list:
                sequence = Path(root_dir) / item
                img_paths = glob.glob(str(sequence / 'color' / '*_color.png'))
                img_paths = sorted(img_paths)
                for img_full_path in img_paths:
                    img_name = Path(img_full_path).name.split('_')[0]
                    img_path = Path(root_dir) / item / 'substitution' / img_name
                    img_list.append(str(img_path))
            with open(list_dir, 'w') as f:
                for img_path in img_list:
                    f.write("%s\n" % img_path)
    print('Write all data paths to file done!')

def load_depth(img_path):
    """ Load depth image from img_path. """
    depth_path = img_path.replace('substitution', 'depth') + '_depth.png'
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth[:, :, 1]*256 + depth[:, :, 2]
        depth16 = np.where(depth16==32001, 0, depth16)
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16

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
    scales = np.max(np.abs(model_points), axis=0)
    return scales

def annotate_cope119(root_dir, gt_pose_dir):
    """ Generate gt labels for Real train. """
    real_train = open(Path('COPE119') / 'COPE119_trainset_list.txt').read().splitlines()
    test_train = open(Path('COPE119') / 'COPE119_testset_list.txt').read().splitlines()

    valid_img_list = []
    for img_path in tqdm(real_train):
        all_exist = Path(img_path.replace('substitution', 'color') + '_color.png').exists() and \
                    Path(img_path.replace('substitution', 'depth') + '_depth.png').exists() and \
                    Path(img_path.replace('substitution', 'mask') + '_mask.png').exists()
        if not all_exist:
            continue
        with open(Path(gt_pose_dir) / Path(img_path).parts[-3] / 'gt_pose.pkl.bz2', 'rb') as f:
            gt_pose = pickle.load(f)
        
        model_path = Path(root_dir) / Path(img_path).parts[-4] / Path(img_path).parts[-3] / 'Model.obj'
        size = get_model_scale(str(model_path)) # This is the size of the model
        scale = np.linalg.norm(size)
        
        mask = cv2.imread(img_path.replace('substitution', 'mask') + '_mask.png')
        mask_non_255 = np.where(mask != 255)
        y1, x1 = np.min(mask_non_255[0]), np.min(mask_non_255[1])
        y2, x2 = np.max(mask_non_255[0]), np.max(mask_non_255[1])
        class_id = mask[:, :, 2][mask[:, :, 2] != 255][0]
        bbox = [y1, x1, y2, x2]
        
        # TODO:direct load the gt poses
        # rotation = R
        # translation = T
        # write results
        gts = {}
        gts['class_ids'] = np.array([class_id])    # int list, 1 to 6
        gts['bboxes'] = np.array([bbox])  # np.array, [[y1, x1, y2, x2], ...]
        gts['scales'] = np.array([scale.astype(np.float32)])  # np.array, scale factor from NOCS model to depth observation
        gts['sizes'] = np.array([size.astype(np.float32)])
        gts['rotations'] = rotation.astype(np.float32)    # np.array, R
        gts['translations'] = translation.astype(np.float32)  # np.array, T
        gts['instance_ids'] = np.array([1])  # int list, start from 1
        gts['model_list'] = np.array(['Model'])  # str list, model id/name
        #TODO: not in this directory
        with open(img_full_path + '_label.pkl', 'wb') as f:
            cPickle.dump(gts, f)
        valid_img_list.append(img_path)
    # write valid img list to file
    with open(Path(data_dir) / 'real' / 'train_list.txt', 'w') as f:
        for img_path in valid_img_list:
            f.write("%s\n" % img_path)


if __name__ == '__main__':
    root_dir = Path("~/Datasets/DynOPETs").expanduser()
    gt_pose_dir = root_dir / "COPE119" / "groundtruth"
    subset_list_dir = Path("COPE119")

    # create list for all data
    # create_img_list(root_dir, subset_list_dir)
    # annotate dataset and re-write valid data to list
    annotate_cope119(root_dir, gt_pose_dir)
