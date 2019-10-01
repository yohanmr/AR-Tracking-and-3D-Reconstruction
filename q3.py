import cv2
import numpy as np
import CalibrationHelpers as calib
import glob
RES = 480
#IMAGES GO IN ORDER 4 3 5 1 2
def FilterByEpipolarConstraint(intrinsics, matches, points1, points2, Rx1, Tx1,threshold = 0.01):
    inlier_mask = []
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        u1 = points1[img1_idx].pt[0]
        v1 = points1[img1_idx].pt[1]
        u2 = points2[img2_idx].pt[0]
        v2 = points2[img2_idx].pt[1]
        x1 = (u1 - intrinsics[0][2])/intrinsics[0][0]
        y1 = (v1 - intrinsics[1][2])/intrinsics[1][1]
        x2 = (u2 - intrinsics[0][2])/intrinsics[0][0]
        y2 = (v2 - intrinsics[1][2])/intrinsics[1][1]
        E = np.cross(Tx1,Rx1,axisa=0,axisb=0)
        p1 = [x1,y1,1]
        p2 = [x2,y2,1]
        if(np.matmul(np.transpose(p2),np.matmul(E,p1)) < threshold):
            inlier_mask.append(1)
        else:
            inlier_mask.append(0)
    # your code here
    return inlier_mask
def relativepose(R2,T2,R1,T1):
    R21 = np.matmul(R2,np.transpose(R1))
    T21 = np.subtract(T2,np.matmul(R21,T1))
    return R21,T21
def ProjectPoints(points3d, new_intrinsics, R, T):
    points2d = np.empty([4,2])
    i=0
    for pts in points3d:
        tmp = np.matmul(R,pts)+T
        uv = np.matmul(new_intrinsics,tmp)
        points2d[i] = [uv[0]/uv[2],uv[1]/uv[2]]
        i=i+1
    # your code here!

    return points2d

def renderCube(img_in, new_intrinsics, R, T):
    # Setup output image
    img = np.copy(img_in)

    # We can define a 10cm cube by 4 sets of 3d points
    # these points are in the reference coordinate frame
    scale = 0.1
    face1 = np.array([[0,0,0],[0,0,scale],[0,scale,scale],[0,scale,0]],
                     np.float32)
    face2 = np.array([[0,0,0],[0,scale,0],[scale,scale,0],[scale,0,0]],
                     np.float32)
    face3 = np.array([[0,0,scale],[0,scale,scale],[scale,scale,scale],
                      [scale,0,scale]],np.float32)
    face4 = np.array([[scale,0,0],[scale,0,scale],[scale,scale,scale],
                      [scale,scale,0]],np.float32)
    # using the function you write above we will get the 2d projected
    # position of these points
    face1_proj = ProjectPoints(face1, new_intrinsics, R, T)
    # this function simply draws a line connecting the 4 points
    img = cv2.polylines(img, [np.int32(face1_proj)], True,
                              tuple([255,0,0]), 3, cv2.LINE_AA)
    # repeat for the remaining faces
    face2_proj = ProjectPoints(face2, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face2_proj)], True,
                              tuple([0,255,0]), 3, cv2.LINE_AA)

    face3_proj = ProjectPoints(face3, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face3_proj)], True,
                              tuple([0,0,255]), 3, cv2.LINE_AA)

    face4_proj = ProjectPoints(face4, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face4_proj)], True,
                              tuple([125,125,0]), 3, cv2.LINE_AA)
    return img

def ComputePoseFromHomography(new_intrinsics, referencePoints, imagePoints):
    # compute homography using RANSAC, this allows us to compute
    # the homography even when some matches are incorrect
    homography, mask = cv2.findHomography(referencePoints, imagePoints,
                                          cv2.RANSAC, 5.0)
    # check that enough matches are correct for a reasonable estimate
    # correct matches are typically called inliers
    MIN_INLIERS = 30
    if(sum(mask)>MIN_INLIERS):
        # given that we have a good estimate
        # decompose the homography into Rotation and translation
        # you are not required to know how to do this for this class
        # but if you are interested please refer to:
        # https://docs.opencv.org/master/d9/dab/tutorial_homography.html
        RT = np.matmul(np.linalg.inv(new_intrinsics), homography)
        norm = np.sqrt(np.linalg.norm(RT[:,0])*np.linalg.norm(RT[:,1]))
        RT = -1*RT/norm
        c1 = RT[:,0]
        c2 = RT[:,1]
        c3 = np.cross(c1,c2)
        T = RT[:,2]
        R = np.vstack((c1,c2,c3)).T
        W,U,Vt = cv2.SVDecomp(R)
        R = np.matmul(U,Vt)
        return True, R, T
    # return false if we could not compute a good estimate
    return False, None, None


intrinsics, distortion, new_intrinsics, roi = calib.LoadCalibrationData('cameradata')
reference = cv2.imread('ARTrackerImage.jpg',0)
reference = cv2.resize(reference,(RES,RES))
feature_detector = cv2.BRISK_create(octaves=5)
reference_keypoints, reference_descriptors = feature_detector.detectAndCompute(reference, None)
keypoint_visualization = cv2.drawKeypoints(
        reference,reference_keypoints,outImage=np.array([]),
        flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imshow("Keypoints",keypoint_visualization)
# wait for user to press a key before proceeding
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# intrinsics, distortion, roi, new_intrinsics = calib.CalibrateCamera('cameradata',True)
# calib.SaveCalibrationData('cameradata', intrinsics, distortion, new_intrinsics, roi)
images = glob.glob('pose_data'+'/*.jpg')
Rref = np.empty([5,3,3])
Tref = np.empty([5,3])
refcount = 0
for fname in images:

    print (fname)
    img = cv2.imread(fname)
    img = cv2.resize(img, None, fx = 0.19, fy = 0.19, interpolation = cv2.INTER_AREA)

    img = cv2.undistort(img, intrinsics, distortion, None,new_intrinsics)
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    current_keypoints, current_descriptors = feature_detector.detectAndCompute(img, None)
    matches = matcher.match(reference_descriptors, current_descriptors)
    #match_visualization = cv2.drawMatches(reference, reference_keypoints, img,current_keypoints, matches, 0,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    #cv2.imshow('matches',match_visualization)
    #cv2.waitKey(0)
    referencePoints = np.float32([reference_keypoints[m.queryIdx].pt \
                                  for m in matches])
    SCALE = 0.1 # this is the scale of our reference image: 0.1m x 0.1m
    referencePoints = SCALE*referencePoints/RES

    imagePoints = np.float32([current_keypoints[m.trainIdx].pt \
                                  for m in matches])
    # compute homography
    ret, R, T = ComputePoseFromHomography(new_intrinsics,referencePoints,
                                          imagePoints)
    render_frame = img
    #if(ret):
    #    render_frame = renderCube(img,new_intrinsics,R,T)
    Rref[refcount] = R
    Tref[refcount] = T
    refcount = refcount + 1
    # display the current image frame
    #cv2.imshow('frame', render_frame)
    #cv2.waitKey(0)
    #input("press enter to continue")
Rrel = np.empty([4,3,3])
Trel = np.empty([4,3])
Rrel[0],Trel[0] = relativepose(Rref[1],Tref[1],Rref[0],Tref[0])
Rrel[1],Trel[1] = relativepose(Rref[2],Tref[2],Rref[0],Tref[0])
Rrel[2],Trel[2] = relativepose(Rref[3],Tref[3],Rref[0],Tref[0])
Rrel[3],Trel[3] = relativepose(Rref[4],Tref[4],Rref[0],Tref[0])
img1 = cv2.imread('pose_data/4.jpg')
img1 = cv2.resize(img1, None, fx = 0.19, fy = 0.19, interpolation = cv2.INTER_AREA)
img1 = cv2.undistort(img1, intrinsics, distortion, None,new_intrinsics)
x, y, w, h = roi
img1 = img1[y:y+h, x:x+w]
img1_keypoints, img1_descriptors = feature_detector.detectAndCompute(img1, None)
refcount = 0
for fname in images:
    if(fname == 'pose_data/4.jpg'):
        continue
    img = cv2.imread(fname)
    img = cv2.resize(img, None, fx = 0.19, fy = 0.19, interpolation = cv2.INTER_AREA)
    img = cv2.undistort(img, intrinsics, distortion, None,new_intrinsics)
    x, y, w, h = roi
    img = img[y:y+h, x:x+w]
    current_keypoints, current_descriptors = feature_detector.detectAndCompute(img, None)
    matches = matcher.match(img1_descriptors, current_descriptors)
    inlier_mask = FilterByEpipolarConstraint(new_intrinsics,matches,img1_keypoints,current_keypoints,Rrel[refcount],Trel[refcount])
    refcount = refcount + 1
    match_visualization = cv2.drawMatches(img1, img1_keypoints, img,current_keypoints, matches, 0, matchesMask =inlier_mask,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('matches',match_visualization)
    cv2.waitKey(0)

cv2.destroyAllWindows()
