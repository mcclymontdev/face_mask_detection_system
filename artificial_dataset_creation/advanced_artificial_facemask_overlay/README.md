# Artificial Facemask Overlay  

## Workspace layout  
```
├── advanced_artificial_facemask_overlay:  
│  │   main.py  
│  │   shape_predictor_68_face_landmarks.dat  
│  ├── DATASET  
│  │   ├── cover_both  
│  │   ├── slice_face  
│  │   ├── uncover_both  
│  │   ├── uncover_nose  
│  ├── face   
│  ├── mask  
```

  
The “face” and “mask” folders are the default locations for unprocessed face and mask images, as well as mask landmark CSV files. 
The “DATASET” directory is the parent folder for saving different categories of output images images, 
which includes sliced faces removed from the rest of the image (slice_face), face with mask overlaid by only covering the chin (uncover_nose), 
uncovering nose (uncover_nose), and correct covering face (cover_both).  
  
## Parameters  
  
- ``-mp`` or ``--maskfolder``: Absolute path to mask folder 
- ``-fp`` or ``--facefolder``: Absolute path to face folder 
- ``-dp`` or ``--datasetfolder``: Absolute path to face folder 
- ``-p`` or ``--PATTERN`` Choices: ``cover_both`` ``uncover_both`` ``uncover_nose`` ``slice_face``. Overlay mask with choice option.  

## Facial landmark detection

By using the face landmark detector, mask images can be accurately and naturally overlaid on the target face image. 
“shape_predictor_68_face_landmarks.dat” is the necessary dependency file for the dlib face landmark detector. 
It detects 68 feature points on a person’s face, an example is shown below.  

## Mask images and landmarks  

The mask landmarks are marked by a webapp called [``Make Sense``](https://www.makesense.ai/), and should consists of 12 points with landmark numbers 1, 4-12, 15 and 30. 
These numbers are matched to the face landmark points. 
This landmark data needs to be exported as a csv file with the same file name as the mask.  
**Be aware that the mask must be a PNG with a transparent background/alpha channel for the optimal overlay effect.**  

It is possible to generate the mask landmark points with any marking tool as long as it is in the following format:  

landmark_number,x_coordinate,y_coordinate,file_name,image_width,image_height  
1,14,67,1.png,489,415  
4,0,220,1.png,489,415  
…  
15,471,67,1.png,489,415  
30,246,0,1.png,489,415  

## Things to note  

For performance reasons, only images with one face can be processed, otherwise it will be ignored. 
To create a versatile mask detector model, a range of different face attributes should be used when generating different types of artificial dataset images with a variety of ``PATTERN`` choices. 
Please be aware that as the resolution of the image increases, the longer the processing time.  
**Be aware that only JPG images are accepted.**  
**For more information, please refer to the user documentation.**  
