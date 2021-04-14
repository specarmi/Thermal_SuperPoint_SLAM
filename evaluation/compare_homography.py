import argparse
import numpy as np
import torch
import tensorflow as tf
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

def sample_homography(
        shape, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=5, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.5, max_angle=np.pi/2,
        allow_artifacts=False, translation_overflow=0.):
    """
    Computes the homography transformation between a random patch in the original image
    and a warped projection with the same image size.
    As in `tf.contrib.image.transform`, it maps the output point (warped patch) to a
    transformed input point (original patch).
    The original patch, which is initialized with a simple half-size centered crop, is
    iteratively projected, scaled, rotated and translated.

    Arguments:
        shape: A rank-2 `Tensor` specifying the height and width of the original image.
        perspective: A boolean that enables the perspective and affine transformations.
        scaling: A boolean that enables the random scaling of the patch.
        rotation: A boolean that enables the random rotation of the patch.
        translation: A boolean that enables the random translation of the patch.
        n_scales: The number of tentative scales that are sampled when scaling.
        n_angles: The number of tentatives angles that are sampled when rotating.
        scaling_amplitude: Controls the amount of scale.
        perspective_amplitude_x: Controls the perspective effect in x direction.
        perspective_amplitude_y: Controls the perspective effect in y direction.
        patch_ratio: Controls the size of the patches used to create the homography.
        max_angle: Maximum angle used in rotations.
        allow_artifacts: A boolean that enables artifacts when applying the homography.
        translation_overflow: Amount of border artifacts caused by translation.

    Returns:
        A `Tensor` of shape `[1, 8]` corresponding to the flattened homography transform.
    """

    # Corners of the output image
    pts1 = tf.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + tf.constant([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]],
                                tf.float32)

    # Random perspective and affine perturbations
    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)
        perspective_displacement = tf.truncated_normal([1], 0., perspective_amplitude_y/2)
        h_displacement_left = tf.truncated_normal([1], 0., perspective_amplitude_x/2)
        h_displacement_right = tf.truncated_normal([1], 0., perspective_amplitude_x/2)
        pts2 += tf.stack([tf.concat([h_displacement_left, perspective_displacement], 0),
                          tf.concat([h_displacement_left, -perspective_displacement], 0),
                          tf.concat([h_displacement_right, perspective_displacement], 0),
                          tf.concat([h_displacement_right, -perspective_displacement],
                                    0)])

    # Random scaling
    # sample several scales, check collision with borders, randomly pick a valid one
    if scaling:
        scales = tf.concat(
                [[1.], tf.random.truncated_normal([n_scales], 1, scaling_amplitude/2)], 0)
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        scaled = tf.expand_dims(pts2 - center, axis=0) * tf.expand_dims(
                tf.expand_dims(scales, 1), 1) + center
        if allow_artifacts:
            valid = tf.range(n_scales)  # all scales are valid except scale=1
        else:
            valid = tf.where(tf.reduce_all((scaled >= 0.) & (scaled < 1.), [1, 2]))[:, 0]
        idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = scaled[idx]

    # Random translation
    if translation:
        t_min, t_max = tf.reduce_min(pts2, axis=0), tf.reduce_min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += tf.expand_dims(tf.stack([tf.random.uniform((), -t_min[0], t_max[0]),
                                         tf.random.uniform((), -t_min[1], t_max[1])]),
                               axis=0)

    # Random rotation
    # sample several rotations, check collision with borders, randomly pick a valid one
    if rotation:
        angles = tf.linspace(tf.constant(-max_angle), tf.constant(max_angle), n_angles)
        angles = tf.concat([angles, [0.]], axis=0)  # in case no rotation is valid
        center = tf.reduce_mean(pts2, axis=0, keepdims=True)
        rot_mat = tf.reshape(tf.stack([tf.cos(angles), -tf.sin(angles), tf.sin(angles),
                                       tf.cos(angles)], axis=1), [-1, 2, 2])
        rotated = tf.matmul(
                tf.tile(tf.expand_dims(pts2 - center, axis=0), [n_angles+1, 1, 1]),
                rot_mat) + center
        if allow_artifacts:
            valid = tf.range(n_angles)  # all angles are valid, except angle=0
        else:
            valid = tf.where(tf.reduce_all((rotated >= 0.) & (rotated < 1.),
                                           axis=[1, 2]))[:, 0]
        idx = valid[tf.random_uniform((), maxval=tf.shape(valid)[0], dtype=tf.int32)]
        pts2 = rotated[idx]

    # Rescale to actual size
    shape = tf.to_float(shape[::-1])  # different convention [y, x]
    pts1 *= tf.expand_dims(shape, axis=0)
    pts2 *= tf.expand_dims(shape, axis=0)

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    a_mat = tf.stack([f(pts1[i], pts2[i]) for i in range(4) for f in (ax, ay)], axis=0)
    p_mat = tf.transpose(tf.stack(
        [[pts2[i][j] for i in range(4) for j in range(2)]], axis=0))
    homography = tf.transpose(tf.linalg.lstsq(a_mat, p_mat, fast=True)) # [1x8]

    homography = tf.reshape(tf.concat([homography, tf.ones([tf.shape(homography)[0], 1])], axis=1), [3, 3]) # [1x3x3]
    return homography


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

def get_sift_matches(sift, img_1, img_2):
    # Get the ORB features
    kpts_1, desc_1 = sift.detectAndCompute(img_1, None)
    kpts_2, desc_2 = sift.detectAndCompute(img_2, None)

    # Create the brute force matcher
    bf = cv2.BFMatcher()

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
        help='The number of matches/correspondences needed to run test (default: 50).')
    parser.add_argument('--max-matches', type=int, default=100, 
        help='The number of matches/correspondences to plot (default: 100).')
    parser.add_argument('--detection-threshold', type=float, default=0.015, 
        help='Superpoint heatmap interest point detection threshold (default: 0.015)')
    parser.add_argument('--nms-dist', type=int, default=4, 
        help='SuperPoint Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--num-loops', type=int, default=1, 
        help='Number of loops to test on. If default or 1 display will generate (default: 1).')
    parser.add_argument('img_1', type=str, help = 'filepath of the image')
    opt = parser.parse_args()

    MIN_MATCH_COUNT = opt.min_matches
    MAX_MATCHES = opt.max_matches
    LOOPS = opt.num_loops

    # Import the image
    img_1_path = opt.img_1
    img1, img_1_tensor = read_image(img_1_path)


    # Load the superpoint models
    print("Loading SuperPoint models...\n")
    thermalModel, deviceThermal = load_model(opt, opt.superpoint_thermal_model_path)
    rgbModel, deviceRGB = load_model(opt, opt.superpoint_rgb_model_path)
    print("Models loaded.\n")

    # Initialize ORB Detector
    orb = cv2.ORB_create()

    # Initialize SIFT Detector
    sift = cv2.SIFT_create()

    MSE = np.zeros(4)
    SSIM = np.zeros(4)
    with tf.Session() as sess:
        for i in range(LOOPS):
            h,w = img1.shape
            img2 = None
            homog = sample_homography((h,w))
            homog_numpy = homog.eval()
            img2 = cv2.warpPerspective(img1, homog_numpy, (w, h))

            img_2_tensor = torch.tensor(img2, dtype=torch.float32).reshape(1, 1, h, w) 
            print("Iteration #",i)

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
            siftMatches, siftKeyPts1, siftKeyPts2 = get_sift_matches(sift, img1, img2)
            print("Total number of SIFT matches: " + str(len(siftMatches)))

            im_out_thermal = None
            im_out_rgb = None
            im_out_orb = None
            im_out_sift = None
            if len(spMatchesThermal) > MIN_MATCH_COUNT:
                src_pts = np.float32([spKeyPtsThermal1[m.queryIdx].pt for m in spMatchesThermal[:MAX_MATCHES]]).reshape(-1,1,2)
                dst_pts = np.float32([spKeyPtsThermal2[m.trainIdx].pt for m in spMatchesThermal[:MAX_MATCHES]]).reshape(-1,1,2)
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                h,w, = img1.shape
                im_out_thermal = cv2.warpPerspective(img2, H, (w, h))
                MSE[0] += np.square(np.subtract(img1,im_out_thermal)).mean()
                SSIM[0] += structural_similarity(img1, im_out_thermal)
        
            if len(spMatchesRGB) > MIN_MATCH_COUNT:
                src_pts = np.float32([spKeyPtsRGB1[m.queryIdx].pt for m in spMatchesRGB[:MAX_MATCHES]]).reshape(-1,1,2)
                dst_pts = np.float32([spKeyPtsRGB2[m.trainIdx].pt for m in spMatchesRGB[:MAX_MATCHES]]).reshape(-1,1,2)
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                h,w, = img1.shape
                im_out_rgb = cv2.warpPerspective(img2, H, (w, h))
                MSE[1] += np.square(np.subtract(img1,im_out_rgb)).mean()
                SSIM[1] += structural_similarity(img1, im_out_rgb)
            
            if len(orbMatches) > MIN_MATCH_COUNT:
                src_pts = np.float32([orbKeyPts1[m.queryIdx].pt for m in orbMatches[:MAX_MATCHES]]).reshape(-1,1,2)
                dst_pts = np.float32([orbKeyPts2[m.trainIdx].pt for m in orbMatches[:MAX_MATCHES]]).reshape(-1,1,2)
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                h,w, = img1.shape
                im_out_orb = cv2.warpPerspective(img2, H, (w, h))
                MSE[2] += np.square(np.subtract(img1,im_out_orb)).mean()
                SSIM[2] += structural_similarity(img1, im_out_orb)
            
            if len(siftMatches) > MIN_MATCH_COUNT:
                src_pts = np.float32([siftKeyPts1[m.queryIdx].pt for m in siftMatches[:MAX_MATCHES]]).reshape(-1,1,2)
                dst_pts = np.float32([siftKeyPts2[m.trainIdx].pt for m in siftMatches[:MAX_MATCHES]]).reshape(-1,1,2)
                H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
                h,w, = img1.shape
                im_out_sift = cv2.warpPerspective(img2, H, (w, h))
                MSE[3] += np.square(np.subtract(img1,im_out_sift)).mean()
                SSIM[3] += structural_similarity(img1, im_out_sift)

            if LOOPS == 1:
                fig, axes = plt.subplots(1, 5)
                axes[0].imshow(img1, cmap='gray', vmin=0, vmax=255)
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
            
        print("SuperPoint-Thermal Reconstruction Error (MSE, SSIM): ", MSE[0] / LOOPS , ",", SSIM[0] / LOOPS)
        print("SuperPoint-RGB Reconstruction Error (MSE, SSIM): ", MSE[1] / LOOPS , ",", SSIM[1] / LOOPS)
        print("ORB Reconstruction Error (MSE, SSIM): ", MSE[2] / LOOPS , ",", SSIM[2] / LOOPS)
        print("SIFT Reconstruction Error (MSE, SSIM): ", MSE[3] / LOOPS , ",", SSIM[3] / LOOPS)



if __name__ == "__main__":
    main()
    
