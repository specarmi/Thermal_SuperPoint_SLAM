import os
import cv2
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Applies CLAHE to an image directory and writes the results to a new '
                                             'directory')
parser.add_argument('directory_path', type=str, help='Path to image directory.')
parser.add_argument('out_dir', type=str, 
    help='Output directory name (it will located in the same folder as the original image directory).')
parser.add_argument('--clip-limit', type=float, default=100, help='CLAHE clip limit (default: 100).')
parser.add_argument('--tile-grid-size', type=int, default=8, help='CLAHE square tile grid sidelength (defualt: 8).')
opt = parser.parse_args()

# Find paths to image file paths
img_dir = Path(opt.directory_path)
image_paths = list(img_dir.iterdir())
image_paths = [str(path) for path in image_paths]
print('Found ' + str(len(image_paths)) + ' files in ' + opt.directory_path + '\n')

# Create output directory
top_dir = os.path.split(opt.directory_path[:-1])[0]
results_dir = top_dir + '/' + opt.out_dir
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print('Using output directory: ' + results_dir)

# Apply CLAHE to images and write the results to the output directory
print('\nApplying CLAHE to images...')
clahe = cv2.createCLAHE(opt.clip_limit, (opt.tile_grid_size, opt.tile_grid_size))
for path in image_paths:
    # Import the image
    input_image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    
    # Apply CLAHE to the 16 bit image and then convert it to a 8 bit image
    clahe_image = clahe.apply(input_image)
    output_image = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Save the resulting image
    path, filename = os.path.split(path)
    path_no_filetype = os.path.splitext(path)[0]
    cv2.imwrite(results_dir + '/' + os.path.splitext(filename)[0] + '.png', output_image)