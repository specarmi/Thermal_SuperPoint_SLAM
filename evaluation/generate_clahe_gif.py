import os
import cv2
import rosbag
import argparse
from cv_bridge import CvBridge, CvBridgeError
from PIL import Image

parser = argparse.ArgumentParser(description='Generates a GIF of ROS image messages with CLAHE applied.')
parser.add_argument('bag_path', type=str, help='Path to ROS bag.')
parser.add_argument('image_topic', type=str, help='Image topic.')
parser.add_argument('out_file', type=str, 
    help='Output filename (the GIF will be generated in same directory as the .bag file).')
parser.add_argument('--frame-rate-divisor', type=int, default=1,
    help='Integer to divide the frame rate by (default: 1).')
parser.add_argument('--time-start', type=float, default=0.0, 
    help='Time (in seconds) since the first frame to begin using frames (default: 0.0).')
parser.add_argument('--time-stop', type=float, default=float('inf'), 
    help='Time (in seconds) since the first frame to stop using frames (default: inf).')
parser.add_argument('--clip-limit', type=float, default=100, help='CLAHE clip limit (default: 100).')
parser.add_argument('--tile-grid-size', type=int, default=8, help='CLAHE square tile grid sidelength (defualt: 8).')
parser.add_argument('--speed-multiplier', type=float, default=2.0, help='Speed increase factor for GIF.')
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

# Output filename
path, filename = os.path.split(opt.bag_path)
result_file = path + '/' + opt.out_file + '.gif'
print('GIF will be saved as: ' + result_file)

# Process messages and write data
print('\nProcessing messages...')
bridge = CvBridge()
pil_images = []
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
                # Convert the image message to an OpenCV image
                cv_image = bridge.imgmsg_to_cv2(msg)

                # If the image is a raw cooled camera image, remove the first row that contains unknown extra info
                if is_cooled:
                    cv_image = cv_image[1:, :]

                cv_image = clahe.apply(cv_image)
                cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

                pil_images.append(Image.fromarray(cv_image))

            except Exception as e:
                print(e)
        count += 1

pil_images[0].save(fp=result_file, format='GIF', append_images=pil_images[1:], save_all=True, 
    duration=(opt.time_stop - opt.time_start)/float(len(pil_images)) * 1000.0 / opt.speed_multiplier, loop=0)