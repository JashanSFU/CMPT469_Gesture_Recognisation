#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import math
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    pyautogui.PAUSE = 0
    lastGestures = None
    # timeBetweenGestures = float('inf')
    # currentGestureTime = None
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    # with open(
    #         'model/point_history_classifier/point_history_classifier_label.csv',
    #         encoding='utf-8-sig') as f:
    #     point_history_classifier_labels = csv.reader(f)
    #     point_history_classifier_labels = [
    #         row[0] for row in point_history_classifier_labels
    #     ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    # palms facing detection details
    arrayOfGestureDetails = []
    for i in range(60):
        twoHands = {}
        twoHands['detected'] = False
        twoHands['leftHandGesture'] = None
        twoHands['rightHandGesture'] = None
        twoHands['leftHandLocation'] = None
        twoHands['rightHandLocation'] = None
        twoHands['leftHandLandmarks'] = None
        twoHands['rightHandLandmarks'] = None
        twoHands['leftHandSize'] = 0.0
        twoHands['rightHandSize'] = 0.0
        twoHands['handDistanceSame'] = False
        twoHands['leftProcessed'] = False
        twoHands['rightProcessed'] = False
        arrayOfGestureDetails.append(twoHands)
    frame = 0
    lastRotationStartIndex = -1
    lastRotationEndIndex = -1
    while True:
        frame += 1
        for i in range(59):
            
            arrayOfGestureDetails[i]['detected'] = arrayOfGestureDetails[i+1]['detected']
            arrayOfGestureDetails[i]['leftHandGesture'] = arrayOfGestureDetails[i+1]['leftHandGesture'] 
            arrayOfGestureDetails[i]['rightHandGesture'] = arrayOfGestureDetails[i+1]['rightHandGesture']
            arrayOfGestureDetails[i]['leftHandLocation'] = arrayOfGestureDetails[i+1]['leftHandLocation']
            arrayOfGestureDetails[i]['rightHandLocation'] = arrayOfGestureDetails[i+1]['rightHandLocation']            
            arrayOfGestureDetails[i]['leftHandLandmarks'] = arrayOfGestureDetails[i+1]['leftHandLandmarks']
            arrayOfGestureDetails[i]['rightHandLandmarks'] = arrayOfGestureDetails[i+1]['rightHandLandmarks']
            arrayOfGestureDetails[i]['leftHandSize'] = arrayOfGestureDetails[i+1]['leftHandSize']
            arrayOfGestureDetails[i]['rightHandSize'] = arrayOfGestureDetails[i+1]['rightHandSize']
            arrayOfGestureDetails[i]['handDistanceSame'] = arrayOfGestureDetails[i+1]['handDistanceSame'] 
            arrayOfGestureDetails[i]['leftProcessed'] = arrayOfGestureDetails[i]['leftProcessed']
            arrayOfGestureDetails[i]['rightProcessed'] = arrayOfGestureDetails[i]['rightProcessed']
      
        i = 59
        arrayOfGestureDetails[i]['detected'] = False
        arrayOfGestureDetails[i]['leftHandGesture'] = None
        arrayOfGestureDetails[i]['rightHandGesture'] = None
        arrayOfGestureDetails[i]['leftHandLocation'] = None
        arrayOfGestureDetails[i]['rightHandLocation'] = None
        arrayOfGestureDetails[i]['leftHandLandmarks'] = None
        arrayOfGestureDetails[i]['rightHandLandmarks'] = None
        arrayOfGestureDetails[i]['leftHandSize'] = 0.0
        arrayOfGestureDetails[i]['rightHandSize'] = 0.0
        arrayOfGestureDetails[i]['handDistanceSame'] = False
        arrayOfGestureDetails[i]['leftProcessed'] = False
        arrayOfGestureDetails[i]['rightProcessed'] = False
        
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            if len(results.multi_hand_landmarks) == 3:
                arrayOfGestureDetails[i]['detected'] = True 
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                # pre_processed_point_history_list = pre_process_point_history(
                #     debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                
                # feature play detection
                # Hand Location
                # Hand Size
                # print(handedness)
                if handedness.classification[0].label[0:] == 'Left':
                    arrayOfGestureDetails[i]['leftHandGesture'] = hand_sign_id
                    arrayOfGestureDetails[i]['leftHandLocation'] = landmark_list[0] #calc_location(landmark_list)
                    arrayOfGestureDetails[i]['leftHandSize'] = distance(landmark_list)
                    arrayOfGestureDetails[i]['leftHandLandmarks'] = landmark_list
                else:
                    arrayOfGestureDetails[i]['rightHandLocation'] = landmark_list[0] #calc_location(landmark_list)
                    arrayOfGestureDetails[i]['rightHandSize'] = distance(landmark_list)
                    arrayOfGestureDetails[i]['rightHandGesture'] = hand_sign_id
                    arrayOfGestureDetails[i]['rightHandLandmarks'] = landmark_list
 

                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id]
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        if arrayOfGestureDetails[i]['leftHandSize'] != 0 and arrayOfGestureDetails[i]['rightHandSize'] != 0:
            ratioFirstBySecond = arrayOfGestureDetails[i]['leftHandSize']/arrayOfGestureDetails[i]['rightHandSize']
            ratioSecondByFirst = arrayOfGestureDetails[i]['rightHandSize']/arrayOfGestureDetails[i]['leftHandSize']
            if (ratioFirstBySecond > 0.95 and ratioFirstBySecond <= 1.1) or (ratioSecondByFirst > 0.95 and ratioSecondByFirst <= 1.1):
                arrayOfGestureDetails[i]['handDistanceSame'] = True
        array = arrayOfGestureDetails
        
        countOfPause = 0
        countOfRotation = 0
        indexFirstRotationDetected = -1
        indexLastRotationDetected = -1
        noRotationBetweenFirstLastRotation = 0
        for index in range(30):
            gesture = arrayOfGestureDetails[index + 30]
            if gesture['leftHandGesture'] == 4:
                countOfPause += 1

        for index, gesture in enumerate(arrayOfGestureDetails):  
            if gesture['leftHandGesture'] == 6:
                countOfRotation += 1
                if gesture['leftProcessed'] == False:
                    indexLastRotationDetected = index
                if indexFirstRotationDetected == -1:
                    indexFirstRotationDetected = index
            elif index < indexLastRotationDetected and index > indexFirstRotationDetected:
                noRotationBetweenFirstLastRotation += 1
       
        if indexFirstRotationDetected == -1:
            lastRotationStartIndex = -1
        
        pauseDetectionThresh = fps/2 if fps/2 <= 7 else 7
        playDetectionThresh = fps/4 if fps/4 <= 10  else 10
        swipeDetectionThresh = fps/2 if fps/4 <= 10 else 20
        rotationDetectionThresh = fps/4 if fps/4 <= 7 else 7
       
        if countOfPause >= pauseDetectionThresh and lastGestures != 'Pause':
            # lastGestures = 'Pause'
            pyautogui.press('x', _pause = False)
            for i in range(30):
                        arrayOfGestureDetails[i + 30]['detected'] = False
                        arrayOfGestureDetails[i + 30]['leftHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['rightHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['leftHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['rightHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['handDistanceSame'] = False
                        arrayOfGestureDetails[i + 30]['leftProcessed'] = True
                        arrayOfGestureDetails[i + 30]['rightProcessed'] = True        

        elif array[30]['leftHandGesture'] == 5 and array[30]['rightHandGesture'] == 5:
            leftHandCountOfSwipe = 0
            rightHandCountOfSwipe = 0
            lastHandIndexOfSwipe = [0,0]
            for i in range(29):
                if array[i+31]['leftHandGesture'] == 5:
                    leftHandCountOfSwipe += 1
                    lastHandIndexOfSwipe[0] = i+31
                if array[i+31]['rightHandGesture'] == 5: 
                    rightHandCountOfSwipe += 1
                    lastHandIndexOfSwipe[1] = i+31
            
            if leftHandCountOfSwipe + rightHandCountOfSwipe >= swipeDetectionThresh:
                leftStartLocation_X = array[30]['leftHandLandmarks'][0][0]
                rightStartLocation_X = array[30]['rightHandLandmarks'][0][0]
                leftEndLocation_X = array[lastHandIndexOfSwipe[0]]['leftHandLandmarks'][0][0]
                rightEndLocation_X = array[lastHandIndexOfSwipe[1]]['rightHandLandmarks'][0][0]
                if leftEndLocation_X > leftStartLocation_X + 50 and rightEndLocation_X > rightStartLocation_X + 50:
                    pyautogui.press('n', _pause = False)
                    lastGestures = 'swipe back'
                    for i in range(30):
                        arrayOfGestureDetails[i + 30]['detected'] = False
                        arrayOfGestureDetails[i + 30]['leftHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['rightHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['leftHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['rightHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['handDistanceSame'] = False
                        arrayOfGestureDetails[i + 30]['leftProcessed'] = False
                        arrayOfGestureDetails[i + 30]['rightProcessed'] = False
                elif leftEndLocation_X < leftStartLocation_X - 50 and rightEndLocation_X < rightStartLocation_X - 50:
                    pyautogui.press('m', _pause = False)
                    lastGestures = 'swipe next'
                    for i in range(30):
                        arrayOfGestureDetails[i + 30]['detected'] = False
                        arrayOfGestureDetails[i + 30]['leftHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['rightHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['leftHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['rightHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['handDistanceSame'] = False
                        arrayOfGestureDetails[i + 30]['leftProcessed'] = False
                        arrayOfGestureDetails[i + 30]['rightProcessed'] = False

        elif array[30]['leftHandGesture'] == 5:
            startLocation_X , startLocation_Y = array[30]['leftHandLocation']
            key = 'leftHandGesture'
            count = 0
            # countForDown = 1
            # handNotGoingUp = 0
            handLocationstatic = 0
            for i in range(29):
                if array[i+31][key] == 1:
                    location_X, location_Y = array[i+31]['leftHandLocation']
                    if location_Y >= startLocation_Y - 15:
                        handLocationstatic += 1
                    count += 1
            if count >= playDetectionThresh and handLocationstatic >= playDetectionThresh/2:
                # if lastGestures != 'Play':
                lastGestures = 'Play'
                pyautogui.press('z', _pause = False)
                
                for i in range(30):
                        arrayOfGestureDetails[i + 30]['detected'] = False
                        arrayOfGestureDetails[i + 30]['leftHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['rightHandGesture'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLocation'] = None
                        arrayOfGestureDetails[i + 30]['leftHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['rightHandLandmarks'] = None
                        arrayOfGestureDetails[i + 30]['leftHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['rightHandSize'] = 0.0
                        arrayOfGestureDetails[i + 30]['handDistanceSame'] = False
                        arrayOfGestureDetails[i + 30]['leftProcessed'] = True
                        arrayOfGestureDetails[i + 30]['rightProcessed'] = True
        
        elif indexFirstRotationDetected != -1 and indexLastRotationDetected != -1 and noRotationBetweenFirstLastRotation < (indexLastRotationDetected - indexFirstRotationDetected)/3: #and countOfRotation <= rotationDetectionThresh:
            #calculations!
            if indexFirstRotationDetected != 0: 
                landmarksForStart = array[indexFirstRotationDetected]['leftHandLandmarks']
                landmarksForLast = array[indexLastRotationDetected]['leftHandLandmarks']
                landmarksForStart = pre_process_landmark(
                    landmarksForStart)
                landmarksForLast = pre_process_landmark(
                    landmarksForLast)
              
                vector1 = (landmarksForStart[17] - landmarksForStart[1], landmarksForStart[16] - landmarksForStart[0])
                vector2 = (landmarksForLast[17] - landmarksForLast[1], landmarksForLast[16] - landmarksForLast[0])
                adjustment = landmarksForLast[1] - landmarksForStart[1]

                if (vector1 != vector2):
                    if landmarksForStart[3] + adjustment <= landmarksForLast[3]: 
                        skip = angle_between(vector1,vector2)
                        degreeMeasure = math.degrees(skip) // 18
                
                        #implementing key stroke
                        if degreeMeasure < 10:
                            lastRotationStartIndex = indexFirstRotationDetected
                            lastRotationEndIndex = indexLastRotationDetected
                            for i in range(indexLastRotationDetected - indexFirstRotationDetected + 1):
                                arrayOfGestureDetails[indexFirstRotationDetected + i]['leftProcessed'] = True
                                arrayOfGestureDetails[indexFirstRotationDetected + i]['rightProcessed'] = True
                                pyautogui.press(str(int(degreeMeasure)), _pause = False)  
            else:
              for i in range(60):
                        arrayOfGestureDetails[i]['detected'] = False
                        arrayOfGestureDetails[i]['leftHandGesture'] = None
                        arrayOfGestureDetails[i]['rightHandGesture'] = None
                        arrayOfGestureDetails[i]['leftHandLocation'] = None
                        arrayOfGestureDetails[i]['rightHandLocation'] = None
                        arrayOfGestureDetails[i]['leftHandLandmarks'] = None
                        arrayOfGestureDetails[i]['rightHandLandmarks'] = None
                        arrayOfGestureDetails[i]['leftHandSize'] = 0.0
                        arrayOfGestureDetails[i]['rightHandSize'] = 0.0
                        arrayOfGestureDetails[i]['handDistanceSame'] = False
                        arrayOfGestureDetails[i]['leftProcessed'] = True
                        arrayOfGestureDetails[i]['rightProcessed'] = True
            
        cv.imshow('Hand Gesture Recognition', debug_image)
    cap.release()
    cv.destroyAllWindows()

def calc_location(landmark_list):
    midpointOfPartA = ((landmark_list[0][0] - landmark_list[5][0])/2, (landmark_list[0][1] - landmark_list[5][1])/2)
    midpointOfPartB = ((landmark_list[1][0] - landmark_list[17][0])/2, (landmark_list[1][1] - landmark_list[17][1])/2)
    partA_X, partA_Y = midpointOfPartA
    partB_X, partB_Y = midpointOfPartB
    return ((partA_X-partB_X)/2 , (partA_Y - partB_Y)/2)

def checkForMotion(array):
    if checkForPause(array) != None:
        return 'Pause'
    elif checkForPlay(array) != None:
        return 'Play'
    return None

def checkForPause(array):
    countOfPauseGesture = 0
    for i in range(30):
        if array[i]['leftHandGesture'] == 4:
            countOfPauseGesture += 1
    if countOfPauseGesture > 0: 
        return 'Pause'
    return None

def checkForPlay(array):
        if array[0]['leftHandGesture'] == 1 and array[0]['rightHandGesture'] == 1:
            # print('here we go')
            count = 0
            countForDown = 1
            handNotGoingUp = 0
             
            for i in range(29):
                if array[i+1]['leftHandGesture'] == 2 and array[i+1]['rightHandGesture'] == 2 :
                    count += 1
                if array[i+1]['leftHandGesture'] == 1 and array[i+1]['rightHandGesture'] == 1 :
                    countForDown += 1
            if count >= 25 and countForDown == 1 and handNotGoingUp <= 2:
                return 'Play'
        return None
def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image

def distance(pointlist):
        distance = 0
        for index in range(4):
            distance += math.dist(pointlist[index], pointlist[index+1])
        for index in range(4):
            if index == 0:
                distance += math.dist(pointlist[0], pointlist[index+5])
            else: 
                distance += math.dist(pointlist[index+4], pointlist[index+5])
        for index in range(4):
            if index == 0: 
                distance += math.dist(pointlist[0], pointlist[index+17])
            else: 
                distance += math.dist(pointlist[index+16],pointlist[index+17])
        for index in range(3):
            distance += math.dist(pointlist[index+9], pointlist[index+10])
        for index in range(3):
            distance += math.dist(pointlist[index+13], pointlist[index+14])
        distance = math.dist(pointlist[5], pointlist[9]) + math.dist(pointlist[9], pointlist[13]) + math.dist(pointlist[13], pointlist[17])
        return distance

def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

if __name__ == '__main__':
    main()
