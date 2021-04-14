import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
from  skimage.metrics import structural_similarity


# Import a function from the pytorch-superpoint submodule
curr_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_path)
sys.path.append("../thirdparty/pytorch-superpoint")
from utils.loader import get_module

def read_image(path):
    input_image = cv2.imread(path)
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

    return np.asarray(pts[0]), np.asarray(desc_sparse[0])

def load_model(opt, modelpath):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Construct the model config
    model_dict = {'name': 'SuperPointNet_gauss2',
                  'params': {},
                  'detection_threshold': opt.detection_threshold,
                  'nms': opt.nms_dist,
                  'nn_thresh': 1.0,
                  'pretrained': modelpath,
                  'batch_size': 1}

    # Load the model
    model_module = get_module("", 'Val_model_heatmap')
    model = model_module(model_dict, device=device)
    model.loadModel()

    return model, device

def get_superpoint_matches(sp_model, device, img_1_tensor, img_2_tensor):
    # Get the SuperPoint features
    pts_1, desc_1 = get_superpoint_features(sp_model, device, img_1_tensor)
    kpts_1 = [cv2.KeyPoint(pt[0], pt[1], pt[2]) for pt in pts_1.T]
    pts_2, desc_2 = get_superpoint_features(sp_model, device, img_2_tensor)
    kpts_2 = [cv2.KeyPoint(pt[0], pt[1], pt[2]) for pt in pts_2.T]

    # Create the brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors.
    matches = bf.match(desc_1.T, desc_2.T)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches, kpts_1, kpts_2

def get_orb_matches(orb, img_1, img_2):
    # Get the ORB features
    kpts_1, desc_1 = orb.detectAndCompute(img_1, None)
    kpts_2, desc_2 = orb.detectAndCompute(img_2, None)

    # Create the brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(desc_1, desc_2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches, kpts_1, kpts_2


def main():
    # Handle arguments
    parser = argparse.ArgumentParser(description ='Draw matches between two images using various features')
    parser.add_argument('superpoint_thermal_model_path', type=str, help = 'Filepath to the trained superpoint thermal model file.')
    parser.add_argument('superpoint_rgb_model_path', type=str, help = 'Filepath to the trained superpoint rgb model file.')
    parser.add_argument('--min-matches', type=int, default=50, 
        help='The number of matches/correspondences to plot (default: 100).')
    parser.add_argument('--detection-threshold', type=float, default=0.015, 
        help='Superpoint heatmap interest point detection threshold (default: 0.015)')
    parser.add_argument('--nms-dist', type=int, default=4, 
        help='SuperPoint Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('img_1', type=str, help = 'filepath of the first image')
    parser.add_argument('img_2', type=str, help = 'filepath of the second image')
    opt = parser.parse_args()

    MIN_MATCH_COUNT = opt.min_matches

    # Import the images
    img_1_path = opt.img_1
    img_2_path = opt.img_2
    img1, img_1_tensor = read_image(img_1_path) 
    img2, img_2_tensor = read_image(img_2_path)

    # Load the superpoint models
    print("Loading SuperPoint models...\n")
    thermalModel, deviceThermal = load_model(opt, opt.superpoint_thermal_model_path)
    rgbModel, deviceRGB = load_model(opt, opt.superpoint_rgb_model_path)
    print("Models loaded.\n")

    # Initialize ORB Detector
    orb = cv2.ORB_create()

    # Initialize SIFT Detector
    sift = cv2.SIFT_create()

    # Get the SuperPoint matches for Thermal
    spMatchesThermal, spKeyPtsThermal1, spKeyPtsThermal2 = get_superpoint_matches(thermalModel, deviceThermal, img_1_tensor, img_2_tensor)
    print("Total number of SuperPoint-Thermal matches: " + str(len(spMatchesThermal)))

    # Get the SuperPoint matches for RGB
    spMatchesRGB, spKeyPtsRGB1, spKeyPtsRGB2 = get_superpoint_matches(rgbModel, deviceRGB, img_1_tensor, img_2_tensor)
    print("Total number of SuperPoint-RGB matches: " + str(len(spMatchesRGB)))

    # Get the ORB matches
    orbMatches, orbKeyPts1, orbKeyPts2 = get_orb_matches(orb, img1, img2)
    print("Total number of ORB matches: " + str(len(orbMatches)))

    # Get SIFT Matches
    siftMatches, siftKeyPts1, siftKeyPts2 = get_orb_matches(orb, img1, img2)
    print("Total number of SIFT matches: " + str(len(siftMatches)))

    print("Evaluation:")
    im_out_thermal = None
    im_out_rgb = None
    im_out_orb = None
    im_out_sift = None
    if len(spMatchesThermal) > MIN_MATCH_COUNT:
        src_pts = np.float32([spKeyPtsThermal1[m.queryIdx].pt for m in spMatchesThermal]).reshape(-1,1,2)
        dst_pts = np.float32([spKeyPtsThermal2[m.trainIdx].pt for m in spMatchesThermal]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w, = img1.shape
        im_out_thermal = cv2.warpPerspective(img1, H, (w, h))
        print("SuperPoint-Thermal Reconstruction Error (MSE, SSIM): ", np.square(np.subtract(img2,im_out_thermal)).mean(), ",", structural_similarity(img2, im_out_thermal))
   
    if len(spMatchesRGB) > MIN_MATCH_COUNT:
        src_pts = np.float32([spKeyPtsRGB1[m.queryIdx].pt for m in spMatchesRGB]).reshape(-1,1,2)
        dst_pts = np.float32([spKeyPtsRGB2[m.trainIdx].pt for m in spMatchesRGB]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w, = img1.shape
        im_out_rgb = cv2.warpPerspective(img1, H, (w, h))
        print("SuperPoint-RGB Reconstruction Error (MSE, SSIM): ", np.square(np.subtract(img2,im_out_rgb)).mean(), ",", structural_similarity(img2, im_out_rgb))
    
    if len(orbMatches) > MIN_MATCH_COUNT:
        src_pts = np.float32([orbKeyPts1[m.queryIdx].pt for m in orbMatches]).reshape(-1,1,2)
        dst_pts = np.float32([orbKeyPts2[m.trainIdx].pt for m in orbMatches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w, = img1.shape
        im_out_orb = cv2.warpPerspective(img1, H, (w, h))
        print("ORB Reconstruction Error (MSE, SSIM): ", np.square(np.subtract(img2,im_out_orb)).mean(), ",", structural_similarity(img2, im_out_orb))
    
    if len(siftMatches) > MIN_MATCH_COUNT:
        src_pts = np.float32([siftKeyPts1[m.queryIdx].pt for m in siftMatches]).reshape(-1,1,2)
        dst_pts = np.float32([siftKeyPts2[m.trainIdx].pt for m in siftMatches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w, = img1.shape
        im_out_sift = cv2.warpPerspective(img1, H, (w, h))
        print("SIFT Reconstruction Error (MSE, SSIM): ", np.square(np.subtract(img2,im_out_sift)).mean(), ",", structural_similarity(img2, im_out_sift))


    fig, axes = plt.subplots(1, 5)
    axes[0].imshow(img2, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Ground Truth')

    axes[1].imshow(im_out_thermal, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('SuperPoint-Thermal Homography')

    axes[2].imshow(im_out_rgb, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('SuperPoint-RGB Homography')

    axes[3].imshow(im_out_orb, cmap='gray', vmin=0, vmax=255)
    axes[3].set_title('ORB Homography')


    axes[4].imshow(im_out_sift, cmap='gray', vmin=0, vmax=255)
    axes[4].set_title('SIFT Homography')
    plt.show()
    

if __name__ == "__main__":
    main()
    