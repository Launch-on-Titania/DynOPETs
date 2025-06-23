from pytorch3d.io import load_objs_as_meshes, load_obj
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, PerspectiveCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    HardPhongShader
)
import pytorch3d
from ipdb import set_trace as st
from PIL import Image
import cv2
import numpy as np
import torch
from pytorch3d.utils import cameras_from_opencv_projection

def load_obj_from_path(obj_path):
    obj_mesh = load_objs_as_meshes([obj_path], device=torch.device("cuda"))
    return obj_mesh

def render_obj_to_image(obj_mesh, R, T, K_mat, image_size, device):
    R = torch.tensor(R, dtype=torch.float32, device=device).unsqueeze(0)
    T = torch.tensor(T, dtype=torch.float32, device=device).unsqueeze(0)
    K_mat = torch.tensor(K_mat, dtype=torch.float32, device=device).unsqueeze(0)
    image_size = torch.tensor(image_size, dtype=torch.float32, device=device).unsqueeze(0)
    # Create camera
    # cameras = PerspectiveCameras
    cameras = create_pytorch3d_fov_perspective_camera(R, T, K_mat, image_size)
    # Setup rasterization settings
    raster_settings = RasterizationSettings(
        image_size=(int(image_size[0][0]), int(image_size[0][1])), 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=0,
    )
    
    # Setup lights
    lights = PointLights(
        ambient_color=((1, 1, 1),), 
        device=device, 
        location=[[0.0, 0.0, -1.0]]
    )
    
    # Create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        # shader=SoftPhongShader(
        #     device=device, 
        #     cameras=cameras,
        #     lights=lights
        # )
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            # lights=lights
        )
    )
    
    # Render the mesh
    images = renderer(obj_mesh)
    
    return images[0]  # Return the first (and only) image

def create_pytorch3d_fov_perspective_camera(R: torch.Tensor, T: torch.Tensor, K_mat: torch.Tensor, image_size: torch.Tensor) -> PerspectiveCameras:
    cameras = cameras_from_opencv_projection(R, T, K_mat, image_size)

    return cameras

def save_image_tensor_as_pil(image_tensor, filename="output.png"):
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu() # If the tensor is on GPU, move it to CPU

    # Convert tensor to NumPy array
    # If it's a float type (0.0-1.0), convert to 0-255 uint8
    if image_tensor.dtype == torch.float32 or image_tensor.dtype == torch.float:
        # Ensure values are in 0-1 range, then multiply by 255 and convert to uint8
        image_np = (image_tensor.clamp(0.0, 1.0) * 255).to(torch.uint8).numpy()
    elif image_tensor.dtype == torch.uint8:
        image_np = image_tensor.numpy()
    else:
        raise TypeError(f"Unsupported tensor data type: {image_tensor.dtype}. Expected float or uint8.")

    # Create PIL Image object
    # mode='RGBA' is for images with shape h x w x 4
    if image_np.shape[2] == 4:
        pil_image = Image.fromarray(image_np, mode='RGBA')
    else:
        pil_image = Image.fromarray(image_np, mode='RGB')

    # Save image
    pil_image.save(filename)
    print(f"saved to: {filename}")


def draw_axis_on_image(image, K, T_o2c, axis_length=0.05, thickness=2):
    """
    Draws 3D axis on the image given camera intrinsics and object-to-camera transform.

    Args:
        image: np.ndarray, the image to draw on (will not be modified in-place).
        K: np.ndarray, (3,3) camera intrinsic matrix.
        T_o2c: np.ndarray, (4,4) object-to-camera transformation matrix.
        axis_length: float, length of the axis in 3D (default 0.1).
        thickness: int, line thickness (default 2).

    Returns:
        np.ndarray: image with axis drawn.
    """

    # Define axis in homogeneous coordinates (origin + x, y, z)
    
    axis = np.array([
        [0.0, 0, 0, 1],
        [axis_length, 0, 0, 1],
        [0, axis_length, 0, 1],
        [0, 0, axis_length, 1],
    ]).T  # shape (4, 4)

    # Transform axis from object to camera coordinates
    transformed_axis = T_o2c @ axis  # shape (4, 4)

    # Project to 2D
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    x = transformed_axis[0, :] / transformed_axis[2, :]
    y = transformed_axis[1, :] / transformed_axis[2, :]

    axis_2d_x = fx * x + cx
    axis_2d_y = fy * y + cy

    # Draw lines: origin to x (red), y (green), z (blue)
    img = image.copy()
    origin = (int(axis_2d_x[0]), int(axis_2d_y[0]))
    x_end = (int(axis_2d_x[1]), int(axis_2d_y[1]))
    y_end = (int(axis_2d_x[2]), int(axis_2d_y[2]))
    z_end = (int(axis_2d_x[3]), int(axis_2d_y[3]))


    img = cv2.line(img, origin, x_end, (0, 0, 255), thickness=thickness, lineType=cv2.LINE_AA)   # Red (BGR)
    img = cv2.line(img, origin, y_end, (0, 255, 0), thickness=thickness, lineType=cv2.LINE_AA)   # Green (BGR)
    img = cv2.line(img, origin, z_end, (255, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)   # Blue (BGR)

    return img
