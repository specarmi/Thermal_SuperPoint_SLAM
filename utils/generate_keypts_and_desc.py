import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import sys

# Import a function from the pytorch-superpoint submodule
curr_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_path)
sys.path.append("../thirdparty/pytorch-superpoint")
from utils.loader import get_module

def load_model(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct the model config
    model_dict = {'name': 'SuperPointNet_gauss2',
                  'params': {},
                  'detection_threshold': opt.detection_threshold,
                  'nms': opt.nms_dist,
                  'nn_thresh': 1.0,
                  'pretrained': opt.superpoint_model_path,
                  'batch_size': 1}

    # Load the model
    model_module = get_module("", 'Val_model_heatmap')
    model = model_module(model_dict, device=device)
    model.loadModel()

    return model, device

def read_image(width, height, path):
    input_image = cv2.imread(path)
    if not width is None:
        input_image = cv2.resize(input_image, (width, height), interpolation=cv2.INTER_AREA)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    input_image_float = input_image.astype('float32') / 255.0
    H, W = input_image_float.shape[0], input_image_float.shape[1]
    return input_image, torch.tensor(input_image_float, dtype=torch.float32).reshape(1, 1, H, W)

def get_superpoint_features(model, device, img):
    model.run(img.to(device))

    # heatmap to pts
    pts = model.heatmap_to_pts()
    
    # subpixel estimation
    pts = model.soft_argmax_points(pts, patch_size=5)

    # heatmap, pts to desc
    desc_sparse = model.desc_to_sparseDesc()

    return np.asarray(pts[0], dtype=np.float32).T, np.asarray(desc_sparse[0], dtype=np.float32).T

if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser(description ='Applies a trained SuperPoint network to an image directory and '
        'outputs the resulting keypoints and descriptors in sequentially named YAML files.')
    parser.add_argument('superpoint_model_path', help = 'Filepath to the trained superpoint model file.', type=str)
    parser.add_argument('directory_path', type = str, help='Path to image directory')
    parser.add_argument('out_dir', type=str, 
        help='Output directory name (it will located in the same folder as the original image directory).')
    parser.add_argument('--max-features', type=int, default=100e3, 
        help='The maximum number of features to keep per image (default: 100e3).')
    parser.add_argument('--resize', type=int, default=None, nargs=2,
        help='The width and height to resize the image to (default: None, the original size is kept)')
    parser.add_argument('--detection-threshold', type=float, default=0.015, 
        help='Superpoint heatmap interest point detection threshold (default: 0.015)')
    parser.add_argument('--nms-dist', type=int, default=4, 
        help='SuperPoint Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--output-orb', action='store_true',
        help='Output ORB features instead of SuperPoint features. If True superpoint_model_path and the max-features '
        'are ignored (default: False)')
    opt = parser.parse_args()

    # Load the model
    if not opt.output_orb:
        print("Loading SuperPoint model...\n")
        model, device = load_model(opt)
        print("Model loaded.\n")

    # Create the ORB detector
    if opt.output_orb:
        orb = cv2.ORB_create()

    # Find paths to image file paths
    image_files = os.listdir(opt.directory_path)
    print('Found ' + str(len(image_files)) + ' files in ' + opt.directory_path + '\n')

    # Create output directory
    top_dir = os.path.split(opt.directory_path[:-1])[0]
    results_dir = top_dir + '/' + opt.out_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print('Using output directory: ' + results_dir + '\n')

    # Loop over images, generate keypoints and descriptors, and log them
    for index, image_file in enumerate(sorted(image_files)):
        image_path = opt.directory_path + image_file

        # Import the image
        if opt.resize is None:
            img_np, img = read_image(None, None, image_path)
        else:
            img_np, img = read_image(opt.resize[0], opt.resize[1], image_path)
    
        # Generate keypoints and descriptors
        if not opt.output_orb:
            kpts, desc = get_superpoint_features(model, device, img)
        else:
            kpts, desc = orb.detectAndCompute(img_np, None)
            kpts = np.asarray([[kp.pt[0], kp.pt[1], kp.response] for kp in kpts])

        # Keep only the top max points (300 in the original bag of binary words paper)
        if not opt.output_orb and opt.max_features < kpts.shape[0]:
            pts = np.hstack((kpts, desc))
            pts = pts[np.argsort(pts[:, 2])]
            kpts = pts[-opt.max_features:, :3]
            desc = pts[-opt.max_features:, 3:]
    
        # Write the results to a yaml file
        result_file = cv2.FileStorage(results_dir + '/' + str(index + 1) + '.yaml', 1)
        result_file.write('keypoints', kpts)
        result_file.write('descriptors', desc)
        result_file.release()