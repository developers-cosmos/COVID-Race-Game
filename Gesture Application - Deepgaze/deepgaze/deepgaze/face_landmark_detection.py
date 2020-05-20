#!/usr/bin/env python

## @package face_landmark_detection.py
#
# Massimiliano Patacchiola, Plymouth University 2016
#
# This module requires dlib >= 18.10 because of the use
# of the shape predictor object.

import numpy
import sys
import cv2
import dlib
import os.path

RIGHT_SIDE = 0
MENTON = 8
LEFT_SIDE = 16
SELLION = 27
NOSE = 30
SUB_NOSE = 33
RIGHT_EYE = 36
RIGHT_TEAR = 39
LEFT_TEAR = 42
LEFT_EYE = 45
STOMION = 62


class faceLandmarkDetection:


    def __init__(self, landmarkPath):
        #Check if the file provided exist
        if(os.path.isfile(landmarkPath)==False):
            raise ValueError('haarCascade: the files specified do not exist.')

        self._predictor = dlib.shape_predictor(landmarkPath)


    ##
    # Find landmarks in the image provided
    # @param inputImg the image where the algorithm will be called
    #
    def returnLandmarks(self, inputImg, roiX, roiY, roiW, roiH, points_to_return=range(0,68)):
            #Creating a dlib rectangle and finding the landmarks
            dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
            dlib_landmarks = self._predictor(inputImg, dlib_rectangle)

            #It selects only the landmarks that
            # have been indicated in the input parameter "points_to_return".
            #It can be used in solvePnP() to estimate the 3D pose.
            self._landmarks = numpy.zeros((len(points_to_return),2), dtype=numpy.float32)
            counter = 0
            for point in points_to_return:
                self._landmarks[counter] = [dlib_landmarks.parts()[point].x, dlib_landmarks.parts()[point].y]
                counter += 1


            #Estimation of the eye dimesnion
            #self._right_eye_w = self._landmark_matrix[RIGHT_TEAR].item((0,0)) - self._landmark_matrix[RIGHT_EYE].item((0,0)) 
            #self._left_eye_w = self._landmark_matrix[LEFT_EYE].item((0,0)) - self._landmark_matrix[LEFT_TEAR].item((0,0))


            return self._landmarks







