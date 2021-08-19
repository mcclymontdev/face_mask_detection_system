# Duplicate Image Removal Script
## duplicate_remover.py
### Usage
The main command `python duplicate_remover.py` will run a non destructive search for duplicates in the folder `dataset`.  
The layout of the folders within the dataset folder is not relevant as it searches across the whole set.  
To change the dataset folder run the main command with the additional argument `-d FOLDER_NAME`.  
To run the script in removal mode (destructive) run the main command with the additional argument `-r 1`.  

## libpng_fix.bat  
### Note
In some instances a warning will appear when running duplicate_remover.py on specific PNG's:  
`libpng warning: iCCP: known incorrect sRGB profile`  
Running libpng_fix.bat will remove the invalid iCCP chunks which cause this warning.  

### Usage  
*Requires [ImageMagick](https://imagemagick.org/script/download.php)*  
**Do not close the script while it is running or images WILL be corrupted.**