from __future__ import print_function

import glob
import os
import cv2
import numpy as np

MAX_FEATURES = 5000
GOOD_MATCH_PERCENT = 0.15

list = os.listdir("prvi_sken2") # dir is your directory path
#number_files =len(list)-1
#number_files = number_files - 1
br = 1
print("Reorientating .jgps")
for i in list:

    if "PDF" in i:
        continue
    refFilename = "ScannedForms\\form.jpg"

    print("Reading reference image : ", refFilename)

    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)



    # Read image to be aligned


    imFilename = "prvi_sken2\\" + i

    print("Reading image to align : ", imFilename);

    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)


    print("Aligning images ...")

    # Registered image will be resotred in imReg.

    # The estimated homography will be stored in h.



    im1Gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    im2Gray = cv2.cvtColor(imReference, cv2.COLOR_BGR2GRAY)



    orb = cv2.ORB_create(MAX_FEATURES)

    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)

    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)



    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

    matches = matcher.match(descriptors1, descriptors2, None)


    matches = sorted(matches, key=lambda x:x.distance)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)

    matches = matches[:numGoodMatches]

    # Draw top matches

    imMatches = cv2.drawMatches(im, keypoints1, imReference, keypoints2, matches, None)

    cv2.imwrite("matches.jpg", imMatches)


    # Extract location of good matches

    points1 = np.zeros((len(matches), 2), dtype=np.float32)

    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for j, match in enumerate(matches):
        points1[j, :] = keypoints1[match.queryIdx].pt
        points2[j, :] = keypoints2[match.trainIdx].pt

    # Find homography

    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography

    height, width = imReference.shape[:2]

    imReg = cv2.warpPerspective(im, h, (width, height))



    # Write aligned image to disk.

    #outFilename = str(br) + ".jpg"
    outFilename=i
    br = br + 1
    outPath = "orijentirani2\\" + outFilename
    print("Saving aligned image : ", outFilename);

    cv2.imwrite(outPath, imReg)
    # Print estimated homography
    print("Estimated homography : \n", h)
