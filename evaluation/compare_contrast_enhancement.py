import os
import cv2
import rosbag
import argparse
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Reads in ROS image messages and displays a comparison of contrast '
    'enhancement techniques.')
parser.add_argument('bag_path', type=str, help='Path to ROS bag.')
parser.add_argument('image_topic', type=str, help='Image topic.')
parser.add_argument('-f', type=int, default=0, 
    help='The frame to process (default: 0).')
opt = parser.parse_args()

# Import the ROS bag
print('Loading bag...\n')
bag = rosbag.Bag(opt.bag_path)

# Confirm the ROS topic is valid
num_messages = bag.get_message_count(opt.image_topic)
print('Found ' + str(num_messages) + ' messages in ' + opt.image_topic + ' topic.\n')
if num_messages == 0:
    raise ImportError('No messages under requested ROS topic.')

# Process messages and write data
print('Processing messages...')
bridge = CvBridge()
frame_count = 0
is_cooled = opt.image_topic == '/t2sls/image_raw'
clip_limits = [100]
tile_grid_sizes = [8]
img_fig, img_axes = plt.subplots(2, len(clip_limits) + 3)
f1_processed = f2_processed = False
for topic, msg, t in bag.read_messages(opt.image_topic):
    # Determine if the current message is the one to process
    process = False
    if frame_count == opt.f:
        process = True
    
    if process:
        # Convert the image message to an OpenCV image
        cv_image = bridge.imgmsg_to_cv2(msg)

        # If the image is a raw cooled camera image, remove the first row that contains unknown extra info
        if is_cooled:
            cv_image = cv_image[1:, :]

        # Direct conversion to 8 bit with no normalization
        output_image = (cv_image / 257).astype('uint8')
        img_axes[0, 0].imshow(output_image, cmap='gray', vmin=0, vmax=255)
        img_axes[0, 0].set_title('Direct Conversion')
        img_axes[0, 0].set_xticks([], [])
        img_axes[0, 0].set_yticks([], [])
        output_image[0, 0] = 200 # Dummy value to procude proper plot
        img_axes[1, 0].hist(output_image.flatten(), 32)
        img_axes[1, 0].set_xlim(0, 255)
        img_axes[1, 0].set_yticks([], [])

        # Apply normalization alone
        output_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        img_axes[0, 1].imshow(output_image, cmap='gray', vmin=0, vmax=255)
        img_axes[0, 1].set_title('Normalization')
        img_axes[0, 1].set_xticks([], [])
        img_axes[0, 1].set_yticks([], [])
        img_axes[1, 1].hist(output_image.flatten(), 32)
        img_axes[1, 1].set_xlim(0, 255)
        img_axes[1, 1].set_yticks([], [])

        # Apply global histogram equalization
        norm_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        output_image = cv2.equalizeHist(norm_image)
        img_axes[0, 2].imshow(output_image, cmap='gray', vmin=0, vmax=255)
        img_axes[0, 2].set_title('Global Hist. Equalization')
        img_axes[0, 2].set_xticks([], [])
        img_axes[0, 2].set_yticks([], [])
        img_axes[1, 2].hist(output_image.flatten(), 32)
        img_axes[1, 2].set_xlim(0, 255)
        img_axes[1, 2].set_yticks([], [])
        
        # Apply CLAHE with various settings to the image
        for i in range(len(clip_limits)):
            clahe = cv2.createCLAHE(clip_limits[i], (tile_grid_sizes[i], tile_grid_sizes[i]))
            clahe_image = clahe.apply(cv_image)
            output_image = cv2.normalize(clahe_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
            img_axes[0, i + 3].imshow(output_image, cmap='gray', vmin=0, vmax=255)
            img_axes[0, i + 3].set_title('CLAHE: CL ' + str(clip_limits[i]) + 
                ', TGS ' + str(tile_grid_sizes[i]) + 'x' + str(tile_grid_sizes[i]))
            img_axes[0, i + 3].set_xticks([], [])
            img_axes[0, i + 3].set_yticks([], [])
            img_axes[1, i + 3].hist(output_image.flatten(), 32)
            img_axes[1, i + 3].set_xlim(0, 255)
            img_axes[1, i + 3].set_yticks([], [])

        break
    
    # Update frame counter
    frame_count += 1

plt.show()