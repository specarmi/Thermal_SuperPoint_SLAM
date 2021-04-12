# Thermal SuperPoint SLAM

Thermal SuperPoint SLAM is a project completed for ROB 530 at the University of Michigan in the winter 2021 semester. This project aimed to create an indirect SLAM algorithm that can successfully perform on thermal imagery. Specifically, we trained a SuperPoint feature detection and description network on thermal imagery and integrated the network with ORB_SLAM2 in place of the ORB feature detector and descriptor. Our combined algorithm runs offline on precomputed keypoints and descriptors. See our report for the details of the process and our results. Although the project was completed with thermal imagery in mind, the steps described here can be followed with any set of images to yield a SuperPoint network and corresponding vocabulary and use them within a modified version of ORB_SLAM2.

This project utilizes four existing codebases:
- SuperPoint Training: https://github.com/eric-yyjau/pytorch-superpoint
- Vocabulary Training: https://github.com/dorian3d/DBoW2
- Integration of SuperPoint and ORB_SLAM2: https://github.com/KinglittleQ/SuperPoint_SLAM
- Original ORB_SLAM2 (for comparison): https://github.com/raulmur/ORB_SLAM2

Each codebase required modifications and the modified forks are included in the *thirdparty* folder as submodules.

# 1. Setup

This library was tested on **Ubuntu 18.04**. After downloading the submodules the setup is divided into three task specific sections: SuperPoint training, vocabulary creation, and SLAM. Each task can be done independent of the others. Finally there are python requirements for our preprocessing and evaluation scripts.

## Downloading Submodules

This repository uses submodules, after cloning download the submodules by running:
```
cd Thermal_SuperPoint_SLAM
git submodule update --init --recursive
```

## SuperPoint Training (pytorch-superpoint)

The pytorch-superpoint repository provides a requirements file for installing dependencies. An example of using it to create an environment with Anaconda is as follows:
```
cd thirdparty/pytorch-superpoint/
conda create --name py36-sp python=3.6
conda activate py36-sp
pip install -r requirements.txt
pip install -r requirements_torch.txt
```

## Vocabulary Generation (DBoW2)

For training a SuperPoint vocabulary DBoW2 only requires [OpenCV](http://opencv.org)(C++). Download and install instructions can be found at: http://opencv.org.

After installing OpenCV the DBoW2 code can be built by running the provided shell script:
```
chmod +x build_vocab_code.sh
./build_vocab_code.sh
```

## SLAM (ORB_SLAM2 and SuperPoint_SLAM)

To run the modified versions of ORB_SLAM2 and SuperPoint_SLAM the following is required:

- C++11 or C++0x Compiler
- [Pangolin](https://github.com/stevenlovegrove/Pangolin): used for visualization and user interface. Download and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.
- [OpenCV](http://opencv.org): used to manipulate images and features. Download and install instructions can be found at: http://opencv.org. **Required at least 2.4.3.**.
- Eigen: required by g2o (an included third party optimization library). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

After installing all of the above the ORB_SLAM2 and SuperPoint_SLAM code can be built by running the provided shell script:
```
chmod +x build_slam_code.sh
./build_slam_code.sh
```

## Preprocessing and Evaluation (*utils/* and *evaluation/*)

**TODO**: create a requirements.txt file for these python scripts. Can they be used without installing ROS?

# 2. Image Directory Preprocessing

This section explains how to apply contrast limited adaptive histogram equalization (CLAHE) to an image directory containing 16 bit thermal images. This is a step we took for training our thermal SuperPoint model but it is not necessary for training on RGB images. See our report for more details.

The script `utils/image_directory_preprocessor.py` is provided to apply CLAHE to an image directory and write the results to a new directory. See the script's help message for the full details.

An example of how to use this script on an image directory located at *../../datasets/FLIR_ADAS/train/Data/* relative to the *utils* folder is as follows:
```
python image_directory_preprocessor.py ../../datasets/FLIR_ADAS/train/Data/ Data_CLAHE
```
This will apply CLAHE to each image in the source directory and output them to *../../datasets/FLIR_ADAS/train/Data_CLAHE/* as PNG images with the same filename.

# 3. ROS Bag Preprocessing

This section explains how to preprocess image messages in a ROS bag into a format suitable for ORB_SLAM2 and SuperPoint_SLAM (the direct ORB_SLAM2 ROS support has not been maintained here). The end result is a text file of timestamps and a folder of images with filenames corresponding to the timestamps. This is the same format ORB_SLAM2 uses for the EuRoC dataset.

The script `utils/rosbag_preprocessor.py` is provided for this purpose. See the script's help message for the full details.

An example of how to use this script with a rosbag located in *../../datasets/vivid/outdoor_robust_day1.bag* is as follows:
```
python rosbag_preprocessor.py ../../datasets/vivid/outdoor_robust_day1.bag /rgb/image outdoor_rgb
```
This will output all images under the topic `/rgb/image` to the directory *../../datasets/vivid/outdoor_rgb/images_30hz_tstart_0_tstop_inf* and will output a text file containing the timestamps of each image to *../../datasets/vivid/outdoor_rgb/timestamps/timestamps_30hz_tstart_0_tstop_inf.txt*. Note that the framerate, start time, and stop time are denoted in the image folder name and the timestamp filename (in this example the original framerate has been assumed to be 30 Hz).

When using ORB_SLAM2 or SuperPoint_SLAM (as described further down) only the images corresponding to the timestamps listed in the given timestamps file will be imported. Therefore, a different subset of images can be used without creating a new folder of images. Continuing with the previous example, if we want to use the same sequence but start it 30 seconds in and reduce it from 30 Hz to 10 Hz, we can generate a new timestamps file by running:
```
python rosbag_preprocessor.py ../../datasets/vivid/outdoor_robust_day1.bag /rgb/image outdoor_rgb --frame-rate-divisor 3 --time-start 30 --timestamps-only
```
The result is a new timestamps file *../../datasets/vivid/rgb_outdoor/timestamps/timestamps_10hz_tstart_30_tstop_inf.txt*.

As mentioned in the previous section we trained our thermal SuperPoint model on images that we had applied CLAHE to and therefore CLAHE should be applied to images before they are used with the model. If the topic passed to `utils/rosbag_preprocessor.py` is 16 bit thermal imagery the `--apply-clahe` flag can be used to apply CLAHE and output 8 bit images. 

# 4. SuperPoint Training

**TODO** Write this section

# 5. Vocabulary Generation

**TODO** Write this section

# 6. Running SuperPoint SLAM

Our modified version of SuperPoint SLAM runs offline on precomputed keypoints and descriptors. The original SuperPoint SLAM could be run online but utilized the pretrained SuperPoint model provided by the original SuperPoint authors [here](https://github.com/magicleap/SuperPointPretrainedNetwork). The third party implementation we use for training employs different layers in the neural network and our trained models are incompatible with the original SuperPoint_SLAM as a result. Our quick fix is to generate keypoints and descriptors offline and import them into SuperPoint_SLAM at runtime.  

## SuperPoint Keypoint and Descriptor Generation

The script `utils/generate_keypts_and_desc.py` is provided to apply a SuperPoint network to an image directory and output the resulting keypoints and descriptors in sequentially named YAML files. See the script's help message for the full details.

An example of how to use this script with the thermal SuperPoint model we trained is as follows:
```
python generate_keypts_and_desc.py ../trained_models/superpoint_thermal/thermal.pth.tar ../../datasets/vivid/outdoor_thermal/images_clahe_10hz_tstart_0_tstop_inf/ features
```
The result is a folder *../../datasets/vivid/outdoor_thermal/features/* containing sequentially named YAML files.

## Running SuperPoint SLAM

Assuming the data is in the format described in the ROS Bag Preprocessing section it can be imported using the EuRoC example. The executable can be run with the following arguments:
```
./thirdparty/SuperPoint_SLAM/Examples/Monocular/mono_euroc <PATH_TO_VOCABULARY> <PATH_TO_CONFIG> <PATH_TO_IMAGE_FOLDER> <PATH_TO_TIMESTAMP_FILE> <PATH_TO_SUPERPOINT_FEATURES>
```
For example:
```
./thirdparty/SuperPoint_SLAM/Examples/Monocular/mono_euroc vocabularies/superpt_thermal.yml.gz configs/ORB_SLAM2/ViViD_Thermal.yaml ../datasets/vivid/outdoor_thermal/images_clahe_10hz_tstart_0_tstop_inf/ ../datasets/vivid/outdoor_thermal/timestamps/timestamps_10hz_tstart_0_tstop_inf.txt ../datasets/vivid/outdoor_thermal/features/
```

# 7. Evaluation and Results
**TODO** Write section
## Comparing Contrast Enhancement Techniques
## Feature Matching 
## Vocabulary Image Similarity Scoring
## Localization
**TODO**: Mention running original ORB_SLAM2, running SuperPoint_SLAM on RGB images with RGB vocab, and (if added) how to quantitatively evaluate performance.