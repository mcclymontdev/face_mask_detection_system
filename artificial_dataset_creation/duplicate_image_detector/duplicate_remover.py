import numpy as np
import argparse
import cv2
import os

MAIN_DIR = os.getcwd()
MAIN_DATASET = "dataset"
IMAGE_TYPES = (".jpg", ".jpeg", ".gif", ".png", ".tiff", ".tif", ".bmp")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, default="dataset")
parser.add_argument("-r", "--remove", type=int, default=-1)
parser.add_argument("-s", "--size", type=int, default=-1)
arguments = vars(parser.parse_args())

MAIN_DATASET = arguments["dataset"]
DATASET_PATH = MAIN_DIR + "\\" + MAIN_DATASET

if os.path.exists(DATASET_PATH) == False:
    print("The dataset location entered does not exist!")
    exit(0)

if arguments["remove"] < 1:
    print("Removal mode is NOT ACTIVE.\n------")
    REMOVE = False
else:
    print("Removal mode is ACTIVE.\n------")
    REMOVE = True

if arguments["size"] < 1:
    print("Size checking (500x500) is NOT ACTIVE.\n------")
    SIZE = False
else:
    print("Size checking (500x500) is ACTIVE.\n------")
    SIZE = True

# https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/
def dhash(image, hashSize=16):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

image_paths = []
for r, d, f in os.walk(DATASET_PATH):
    for file in f:
        if file.endswith(IMAGE_TYPES):
            image_paths.append(os.path.join(r, file))

hashes = {}
for count, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)

    # Catching corrupted images
    try:
        image_hash = dhash(image)
    except:
        print("CORRUPTED IMAGE DETECTED: ", image_path)
        print("Exiting...")
        exit(0)

    if SIZE & (image.shape[0] < 500 or image.shape[1] < 500):
        if REMOVE == False:
            print("Image does not match size specification: " + image_path, end="\n")
        else:
            print("Removing image that does not match size specification: " + image_path, end="\n")
            os.remove(image_path)

    else:
        # Adding the hash to our dictionary
        paths = hashes.get(image_hash, [])
        paths.append(image_path)
        hashes[image_hash] = paths

    # Only update percentage every 10 images
    if (count % 10 == 0):
        print("Images processed: " + str(round(((count+1)/len(image_paths)*100),2)) + "% ", end='\r')

# Loop over our hash dictionary to check for duplicates
duplicates_detected = 0
for (h, hashedPaths) in hashes.items():
    if len(hashedPaths) > 1:
        duplicates_detected += 1
        # If we are not removing the images we can just print the paths for manual inspection
        if REMOVE == False:
            print("DUPLICATE IMAGES DETECTED:  ")
            for paths in hashedPaths:
                print(paths)
            print("------")
        else:
            # Removes all instances of the duplicate other than the first instance of the image
            print("DUPLICATE IMAGES DETECTED:  ", hashedPaths[0])
            for p in hashedPaths[1:]:
                os.remove(p)
                print("DUPLICATE REMOVED: ", p)
                print("------")

if duplicates_detected > 0:
    print("DUPLICATES DETECTED: " + str(duplicates_detected))
else:
    print("NO DUPLICATES DETECTED  ")