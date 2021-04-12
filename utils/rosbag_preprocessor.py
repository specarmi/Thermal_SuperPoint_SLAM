import os
import cv2
import rosbag
import argparse
from cv_bridge import CvBridge, CvBridgeError

parser = argparse.ArgumentParser(description='Writes ROS image messages to image folder and timestamp text file.')
parser.add_argument('bag_path', type=str, help='Path to ROS bag.')
parser.add_argument('image_topic', type=str, help='Image topic.')
parser.add_argument('out_dir', type=str, 
    help='Output directory name (the directory will be generated in same directory as the .bag file).')
parser.add_argument('--frame-rate-divisor', type=int, default=1,
    help='Integer to divide the frame rate by (default: 1).')
parser.add_argument('--time-start', type=float, default=0.0, 
    help='Time (in seconds) since the first frame to begin saving frames (default: 0.0).')
parser.add_argument('--time-stop', type=float, default=float('inf'), 
    help='Time (in seconds) since the first frame to stop saving frames (default: inf).')
parser.add_argument('--timestamps-only', action='store_true',
    help='Only save the timestamps file (default: False). Used to generate a new timestamps file in an existing output '
         'directory.')
parser.add_argument('--apply-clahe', action='store_true',
    help='Apply CLAHE, if False the raw 16 bit image is saved, if True CLAHE is applied and an 8 bit image is saved '
         '(default: False)')
parser.add_argument('--apply-median-blur', action='store_true',
    help='Apply median blur to the image (default: False)')
parser.add_argument('--clip-limit', type=float, default=100, help='CLAHE clip limit (default: 100).')
parser.add_argument('--tile-grid-size', type=int, default=8, help='CLAHE square tile grid sidelength (defualt: 8).')
opt = parser.parse_args()

# Import the ROS bag
print('Loading bag...\n')
bag = rosbag.Bag(opt.bag_path)

# Determine the frame rate of the desired image topic
start = bag.get_start_time()
end = bag.get_end_time()
num_messages = bag.get_message_count(opt.image_topic)
original_frame_rate = round(float(num_messages) / (end - start))
new_frame_rate = float(original_frame_rate) / opt.frame_rate_divisor
print('Found ' + str(num_messages) + ' messages in ' + opt.image_topic + ' topic.')
print('Original frame rate: ' + str(original_frame_rate))
print('New frame rate: ' + str(new_frame_rate) + '\n')

# Create output directory
path, filename = os.path.split(opt.bag_path)
results_dir = path + '/' + opt.out_dir
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print('Using output directory: ' + results_dir)

# Create the image directory and timestamp file
time_stop_str = 'tstop_inf'
if not opt.time_stop == float('inf'):
    time_stop_str = 'tstop_' + str(int(opt.time_stop))
param_info = str(int(new_frame_rate)) + 'hz_tstart_' + str(int(opt.time_start)) + '_' + time_stop_str
if not opt.timestamps_only:
    clahe_info = ''
    if opt.apply_clahe:
        clahe_info = 'clahe_'
    image_dir = results_dir + '/images_' + clahe_info + param_info
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    print('Using image subdirectory: ' + image_dir)
timestamp_dir = results_dir + '/timestamps'
if not os.path.exists(timestamp_dir):
    os.makedirs(timestamp_dir)
timestamp_filename = timestamp_dir + '/' + 'timestamps_' + param_info + '.txt'
timestamp_file = open(timestamp_filename, 'w')
print('Using timestamp subdirectory: ' + timestamp_dir)
print('Writing to timestamp file: ' + timestamp_filename)

# Process messages and write data
print('\nProcessing messages and writing data...')
bridge = CvBridge()
timestamp_list = []
count = 0
is_cooled = opt.image_topic == '/t2sls/image_raw'
clahe = cv2.createCLAHE(opt.clip_limit, (opt.tile_grid_size, opt.tile_grid_size))
initial_msg = True
for topic, msg, t in bag.read_messages(opt.image_topic):
    # Read the timestamp (in nanoseconds)
    timestamp = int(msg.header.stamp.secs * 1e9) + int(msg.header.stamp.nsecs)

    # Record the initial timestamp
    if initial_msg:
        initial_time = timestamp
        initial_msg = False

    # Stop if the time has reached time stop
    if float(timestamp - initial_time) / 1e9 >= opt.time_stop:
        break

    # Begin saving data if the time has reached time start
    if float(timestamp - initial_time) / 1e9 >= opt.time_start:
        # Skip frames to achieve desire frame rate
        if count % opt.frame_rate_divisor == 0:
            try:
                # Save the timestamp
                timestamp_list.append(timestamp)

                # Handle the image
                if not opt.timestamps_only:
                    # Convert the image message to an OpenCV image
                    cv_image = bridge.imgmsg_to_cv2(msg)

                    # If the image is a raw cooled camera image, remove the first row that contains unknown extra info
                    if is_cooled:
                        cv_image = cv_image[1:, :]

                    # Apply median blur if enabled
                    if opt.apply_median_blur:
                        cv_image = cv2.medianBlur(cv_image, 5)

                    # Apply CLAHE if enabled
                    if opt.apply_clahe:
                        cv_image = clahe.apply(cv_image)
                        cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

                    # Save the image using the timestamp as a filename
                    image_filename = str(timestamp) + '.png'
                    cv2.imwrite(image_dir + '/' + image_filename, cv_image)

            except Exception as e:
                print(e)
        count += 1

timestamp_list = list(set(timestamp_list))
timestamp_list.sort()
timestamp_file.write('\n'.join(str(time) for time in timestamp_list))
timestamp_file.close()