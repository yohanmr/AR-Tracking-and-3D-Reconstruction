# Implementing AR Tracking and 3D Reconstruction

## Dependencies and Code Base

Dependencies: Python 3.5.2, openCV 3.4.1, Glob, open3d.

calibration_data: A folder with at least 10 pictures of the checkboard in various orientations to calculate camera intrinsics.

CalibrationHelpers.py: Calculates and stores calibration data using calibration images.

ARImagePoseTracker.py: tracks a known image and computing the camera pose relative to this image in the scene. We use the BRISK feature detector on the reference image and the frame. The Rotational and translational matrix are computed by homography. Using this we render a cube on the image.

Example Image:

![AR Render](/images/ARRender.png)

cameradata: Holds calibration necessary images to get mobile camera intrinsics

pose_data: Images of known reference places in the background we wish to reconstruct.

q3.py: Using the Images from pose_data , we calculate the Rotation and Translation matrix WRT the reference and each other. We also find the common feature points between all images that follow the Epipolar Constraint. We than compute the depth of feature points using the reference and relative pose data. It is then plotted using open3d.

Example Image:

![3d point1](/images/3D1.png)

![3d point2](/images/3D2.png)
