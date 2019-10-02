# Implementing AR Tracking and 3D Reconstruction

## AR Tracking

###Dependencies and Code Base

Python 3.5.2 and openCV 3.4.1

calibration_data: A folder with at least 10 pictures of the checkboard in various orientations to calculate camera intrinsics.

CalibrationHelpers.py: Calculates and stores calibration data using calibration images.

ARImagePoseTracker.py: tracks a known image and computing the camera pose relative to this image in the scene. We use the BRISK feature detector on the reference image and the frame. The Rotational and translational matrix are computed by homography. Using this we render a cube on the image.

Example Image:
![AR Render](/images/ARRender.png)
