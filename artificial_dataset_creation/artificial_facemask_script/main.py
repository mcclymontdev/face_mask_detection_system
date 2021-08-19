# Usage: python main.py FOLDER_OF_FACES FOLDER_TO_SAVE_NEW_IMAGES
# Modified with explicit written permission from the author Prajna Bhandary 
# Source: https://github.com/prajnasb/observations/blob/master/mask_classifier/Data_Generator/mask.py

import os
import sys
import random
import numpy as np
from PIL import Image, ImageFile
import face_recognition
import cv2
import argparse

MAIN_DIR = os.getcwd()

MASK_FOLDER = MAIN_DIR + "\masks\\"

parser = argparse.ArgumentParser(description='Adds a mask to a given image.')
parser.add_argument('UNMASKED_FOLDER', help='Folder containing images of unmasked faces.')
parser.add_argument('MASKED_FOLDER', help='Folder to save new masked images too.')
parser.add_argument('--debug', action='store_true', help='Show images with points ontop and console logs')
arguments = parser.parse_args()

UNMASKED_FOLDER = MAIN_DIR + "\\" + arguments.UNMASKED_FOLDER + "\\"
MASKED_FOLDER = MAIN_DIR + "\\" + arguments.MASKED_FOLDER + "\\"
DEBUG = arguments.debug

KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

unmasked_faces = [f for f in os.listdir(UNMASKED_FOLDER) if os.path.isfile(os.path.join(UNMASKED_FOLDER, f))]

def create_mask(face_path, mask_path, i):
    face_image_np = face_recognition.load_image_file(face_path)
    face_locations = face_recognition.face_locations(face_image_np, model="hog")
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
    face_img = Image.fromarray(face_image_np)
    mask_img = Image.open(mask_path)

    for face_landmark in face_landmarks:
        for facial_feature in KEY_FACIAL_FEATURES:
            if facial_feature in face_landmark:
                face_img = mask_face(face_img, face_image_np, mask_img, face_landmark)
                face_img.save(MASKED_FOLDER + str(i) + ".jpg")
                print("Face found - image saved (" + str(i) + ".jpg)")
                return
    print("Face not found - " + str(i))

def mask_face(face_img, face_image_np, mask_img, face_landmark):
        nose_bridge = face_landmark['nose_bridge']
        nose_v = np.mean(nose_bridge, axis=0)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]

        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[3]
        chin_right_point = chin[-4]

        # split mask and resize
        width = mask_img.width
        height = mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = mask_img.crop((0, 0, width // 2, height))
        mask_left_width = distance_perpendicular(chin_bottom_v, nose_v, chin_left_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = mask_img.crop((width // 2, 0, width, height))
        mask_right_width = distance_perpendicular(chin_bottom_v, nose_v, chin_right_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_v[1] - nose_v[1], chin_bottom_v[0] - nose_v[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_v[0] + chin_bottom_v[0]) // 2
        center_y = (nose_v[1] + chin_bottom_v[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = round(center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2)
        box_y = round(center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2)

        if DEBUG:
            print("nose_bridge: ", nose_bridge)
            print("nose_v: ", nose_v)
            print("new_height: ", new_height)
            RGB_img = cv2.cvtColor(face_image_np, cv2.COLOR_BGR2RGB)
            cv2.circle(img=RGB_img, center=(box_x, box_y), radius=5, color=(255, 20, 0), thickness=-1)
            cv2.circle(img=RGB_img, center=(int(nose_v[0]), int(nose_v[1])), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(img=RGB_img, center=(int(chin_bottom_v[0]), int(chin_bottom_v[1])), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.circle(img=RGB_img, center=(int(chin_left_point[0]), int(chin_left_point[1])), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.circle(img=RGB_img, center=(int(chin_right_point[0]), int(chin_right_point[1])), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.imshow(winname="Face", mat=RGB_img)
            cv2.waitKey(0)

        # add mask
        face_img.paste(mask_img, (box_x, box_y), mask_img)
        return face_img

# Distance from point P3 perpendicular to line P1P2
def distance_perpendicular(P1, P2, P3):
    return np.linalg.norm(np.cross(P2-P1, P1-P3))/np.linalg.norm(P2-P1)

# Get array of masks to be overlayed on images
masks = [f for f in os.listdir(MASK_FOLDER) if os.path.isfile(os.path.join(MASK_FOLDER, f))]
print("Masks found:")
print(masks)

for i,face in enumerate(unmasked_faces):
    mask_path = MASK_FOLDER + masks[i % len(masks)]
    face_path = UNMASKED_FOLDER + face
    create_mask(face_path, mask_path, i)