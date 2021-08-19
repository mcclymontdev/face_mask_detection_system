# How to use the artificial face mask script:  

Examples of appropriate pictures, masks and their outputs are available in the appropriate folders:
- ``faces`` - Unprocessed images 
- ``maskedfaces`` - Processed images 
- ``masks`` - Mask PNGs for overlaying on images within ``faces``  

Mask PNGs should be placed within ``masks``.

``main.py`` takes in 2 arguments: 
- `UNMASKED_FOLDER` - a relative directory containing unprocessed portrait images.
- `MASKED_FOLDER` - a relative directory to save processed images.

Example:  
``main.py faces maskedfaces``