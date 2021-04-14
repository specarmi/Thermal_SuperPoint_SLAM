import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
import sys
from PIL import Image

# Import a function from the pytorch-superpoint submodule
curr_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_path)
sys.path.append("../thirdparty/pytorch-superpoint")
from utils.loader import get_module

# Jet colormap for visualization.
# Code from: https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class PointTracker(object):
  """ Modified Code from: https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py
  
  Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh, matcher):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999
    self.matcher = matcher

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    cv_matches = self.matcher.match(desc1.T, desc2.T)
    cv_matches = [match for match in cv_matches if match.distance < nn_thresh]
    matches = np.asarray([[match.queryIdx, match.trainIdx, match.distance] for match in cv_matches]).T
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)

def load_model(opt, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model == 'thermal':
        pretrained = opt.thermal_superpoint_model_path
    else:
        pretrained = opt.rgb_superpoint_model_path

    # Construct the model config
    model_dict = {'name': 'SuperPointNet_gauss2',
                  'params': {},
                  'detection_threshold': opt.detection_threshold,
                  'nms': opt.nms_dist,
                  'nn_thresh': 1.0,
                  'pretrained': pretrained,
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
    parser = argparse.ArgumentParser(description ='Generates a feature tracking GIF from an image directory')
    parser.add_argument('thermal_superpoint_model_path', help = 'Filepath to the trained thermal superpoint model '
        'file.', type=str)
    parser.add_argument('rgb_superpoint_model_path', help = 'Filepath to the trained RGB superpoint model '
        'file.', type=str)
    parser.add_argument('directory_path', type = str, help='Path to image directory')
    parser.add_argument('out_file', type=str, help='Output filename.')
    parser.add_argument('frame_rate', type=float, help='The framerate of the images in the directory.')
    parser.add_argument('--speed-multiplier', type=float, default=2.0, help='Speed increase factor for GIF.')
    parser.add_argument('--resize', type=int, default=None, nargs=2,
        help='The width and height to resize the image to (default: None, the original size is kept)')
    parser.add_argument('--detection-threshold', type=float, default=0.015, 
        help='Superpoint heatmap interest point detection threshold (default: 0.015)')
    parser.add_argument('--nms-dist', type=int, default=4, 
        help='SuperPoint Non Maximum Suppression (NMS) distance (default: 4).')
    parser.add_argument('--thermal_nn_thresh', type=float, default=0.7, 
        help='Thermal SuperPoint descriptor L2 distance matching threshold (default: 0.7).')
    parser.add_argument('--rgb_nn_thresh', type=float, default=0.7, 
        help='RGB SuperPoint descriptor L2 distance matching threshold (default: 0.7).')
    parser.add_argument('--orb_nn_thresh', type=float, default=50, 
        help='ORB descriptor Hamming matching threshold (default: 100).')
    parser.add_argument('--sift_nn_thresh', type=float, default=100, 
        help='SIFT descriptor L2 distance matching threshold (default: 100).')
    opt = parser.parse_args()

    # Load the thermal model
    print("Loading thermal SuperPoint model...")
    thermal_model, thermal_device = load_model(opt, 'thermal')
    print("Thermal Model loaded.\n")

    # Load the RGB model
    print("Loading RGB SuperPoint model...")
    rgb_model, rgb_device = load_model(opt, 'rgb')
    print("RGB Model loaded.\n")

    # Initiate the ORB detector
    orb = cv2.ORB_create(fastThreshold=5)

    # Initiate the SIFT detector
    sift = cv2.xfeatures2d_SIFT.create()

    # Find paths to image file paths
    image_files = os.listdir(opt.directory_path)
    print('Found ' + str(len(image_files)) + ' files in ' + opt.directory_path + '\n')

    # Font parameters for visualizaton.
    font = cv2.FONT_HERSHEY_DUPLEX
    font_clr = (255, 255, 255)
    font_pt = (10, 40)
    font_sc = 1

    # Create trackers
    bf_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    bf_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    thermal_tracker = PointTracker(5, nn_thresh=opt.thermal_nn_thresh, matcher=bf_l2)
    rgb_tracker = PointTracker(5, nn_thresh=opt.rgb_nn_thresh, matcher=bf_l2)
    orb_tracker = PointTracker(5, nn_thresh=opt.orb_nn_thresh, matcher=bf_hamming)
    sift_tracker = PointTracker(5, nn_thresh=opt.sift_nn_thresh, matcher=bf_l2)

    # Loop over images, generate tracks, and save images
    win = 'Tracking'
    cv2.namedWindow(win)
    pil_images = []
    for index, image_file in enumerate(sorted(image_files)):
        image_path = opt.directory_path + image_file

        # Import the image
        if opt.resize is None:
            img_np, img = read_image(None, None, image_path)
        else:
            img_np, img = read_image(opt.resize[0], opt.resize[1], image_path)
    
        # Update thermal tracker
        pts, desc = get_superpoint_features(thermal_model, thermal_device, img)
        thermal_tracker.update(pts.T, desc.T)
        tracks = thermal_tracker.get_tracks(2)
        thermal_img = (np.dstack((img_np, img_np, img_np))).astype('uint8')
        tracks[:, 1] /= float(opt.thermal_nn_thresh) # Normalize track scores to [0,1].
        thermal_tracker.draw_tracks(thermal_img, tracks)
        cv2.putText(thermal_img, 'Thermal SuperPoint', font_pt, font, font_sc, font_clr, lineType=16)

        # Update RGB tracker
        pts, desc = get_superpoint_features(rgb_model, rgb_device, img)
        rgb_tracker.update(pts.T, desc.T)
        tracks = rgb_tracker.get_tracks(2)
        rgb_img = (np.dstack((img_np, img_np, img_np))).astype('uint8')
        tracks[:, 1] /= float(opt.rgb_nn_thresh) # Normalize track scores to [0,1].
        rgb_tracker.draw_tracks(rgb_img, tracks)
        cv2.putText(rgb_img, 'RGB SuperPoint', font_pt, font, font_sc, font_clr, lineType=16)

        # Update ORB tracker
        kpts, desc = orb.detectAndCompute(img_np, None)
        pts = np.asarray([[kp.pt[0], kp.pt[1], kp.response] for kp in kpts])
        orb_tracker.update(pts.T, desc.T)
        tracks = orb_tracker.get_tracks(2)
        orb_img = (np.dstack((img_np, img_np, img_np))).astype('uint8')
        tracks[:, 1] /= float(opt.orb_nn_thresh) # Normalize track scores to [0,1].
        orb_tracker.draw_tracks(orb_img, tracks)
        cv2.putText(orb_img, 'ORB', font_pt, font, font_sc, font_clr, lineType=16)

        # Update SIFT tracker
        kpts, desc = sift.detectAndCompute(img_np, None)
        pts = np.asarray([[kp.pt[0], kp.pt[1], kp.response] for kp in kpts])
        sift_tracker.update(pts.T, desc.T)
        tracks = sift_tracker.get_tracks(2)
        sift_img = (np.dstack((img_np, img_np, img_np))).astype('uint8')
        tracks[:, 1] /= float(opt.sift_nn_thresh) # Normalize track scores to [0,1].
        sift_tracker.draw_tracks(sift_img, tracks)
        cv2.putText(sift_img, 'SIFT', font_pt, font, font_sc, font_clr, lineType=16)

        # Construct the final image 
        final_img = np.hstack((thermal_img, rgb_img, orb_img, sift_img))

        # Show the final image
        cv2.imshow(win, final_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print('Quitting, \'q\' pressed.')
            break

        # Save the final image
        pil_images.append(Image.fromarray(cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)))

pil_images[0].save(fp=opt.out_file, format='GIF', append_images=pil_images[1:], save_all=True, 
    duration= (1 / opt.frame_rate) * 1000.0 / opt.speed_multiplier, loop=0)