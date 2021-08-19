# CS19 in collaboration with CENSIS

University project completed as of March 2021

## Face mask detection system with fever checking  

This project is designed to help with workplace/public safety in response to COVID-19.  

We aim to ease the issue of non-compliance/forgetfulness with mask wearing before 
entering buildings by prompting users to wear a mask based on our face 
mask detection model.  

We have automated the process of temperature measurement; no human intervention 
is required. If the user has a temperature greater than 38°C the user is alerted 
of their elevated temperature and directed to not enter the building.  

Part of the project aim was to allow future development/expansion of our model by 
providing scripts to create artificial dataset images (face masks overlaid on the 
user's choice of images) and making it easy for users to train a model using our 
algorithm, this process is described in detail within the documentation.  

## Usage

Instructions on how to use each part of this project is described in the documentation.

### mask_detection

Directory Layout:

<pre>
-mask_detection
|---Build_Up_Algo
|   |--detect_image.py
|   |--firstdetector.py
|---face_detector
|   |--deploy.prototxt
|   |--res10_300x300_ssd_iter_140000.caffemodel
|---thermal_grabber
|   |--grabber_lib
|   |--thermal_grabber
|   |--CMakeLists.txt
|   |--README.md
|---|calibrator.py
|---|detector.py
|---|learning_algo.py
|---|mask_detector.model
|---|README.md
</pre>

The above directory contains our face mask detection system, thermal data
grabber, pretrained face mask detection model, and our training algorithm.  

### artificial_dataset_creation

Directory Layout:

<pre>
-artificial_dataset_creation
|---advanced_artificial_facemask_overlay
|   |--DATASET
|   |--face
|   |--mask
|   |--main.py
|   |--README.md
|   |--shape_predictor_68_face_landmarks.dat
|---artificial_facemask_script
|   |--faces
|   |--maskedfaces
|   |--masks
|   |--main.py
|   |--mask-sources.txt
|   |--README.md
|---duplicate_image_detector
|   |--duplicate_remover.py
|   |--libpng_fix.bat
|   |README.md
</pre>

The above directory contains our artificial face mask overlay scripts.  
These are designed to be used in the creation of a dataset for training a face
mask detection model.
