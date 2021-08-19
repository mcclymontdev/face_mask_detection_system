# How to train a machine learning algorithm:

- In order to train an existing model use the ``--model`` or ``-m`` flag and provide the path to the model to train. If producing a new model then avoid this flag.
- Use the ``--dataset`` or ``-d`` and provide the path to the folder that contains the dataset folder for masked and unmasked images.
- Use the following command to run the code ``python3 learning_algo.py --dataset dataset_path``.

``learning_algo.py`` takes in 4 arguments:  
- ``-d`` or ``--dataset``, this is used to provide the path to the directory containing the dataset. For the above example, the argument would be -d .\dataset.
- ``-p`` or ``--plot`` can be used to generate the plot with regards to the accuracy of the model on the train and the test set. For example, the argument would be -p or -p “Graph.png”.
- ``-m`` or ``--model`` argument to provide the make for the model to be saved as, the file extension is .model. for example, the argument would be -m or --model “first_model.model”.
- ``-tm`` or ``--trainmodel`` is used to provide a path to a pre-trained model to be re-trained. For example, for the above workspace layout let us say we want to train ``model1.model`` then the argument would be ``-tp`` or ``--trainmodel .\model1.model``.  

Example:   
- the command to re-train ``model1.model`` on ``newDS`` and save it as ``DSmodel.model`` is as follows: 
``python learning_algo.py -d .\dataset2\newDS -tp .\model1.model -m “DSmodel.model”``

# How to run the face mask detector:

- If the name of the model is not ``mask_detector.model`` then use the ``--model`` or ``-m`` and provide the path to the .model file.
- Use the following command to run the detector with only face mask detection ``python3 detector.py``.

### Arguments

- To specify a different face detector model use: ``-f`` or ``--face NEW_DIRECTORY`` - default: face_detector
- To specify a different face mask detector model use: ``-m`` or ``--model NEW_MODEL_PATH`` - default: mask_detector.model
- To specify a minimum confidence for face detection use: ``-c`` or ``--confidence (0.0-1.0)`` - default: 0.5 

### Hotkeys:  
- `q` - Quit/exit the facemask detector completely.

## How to run the face mask detector with thermal:  

**Note: the thermal calibrator should only be used if the temperature varies by a significant amount i.e. the thermal camera’s factory calibration is off.**  

### Arguments:
- To activate thermal mode use: ``-t`` or ``--thermal``
- To activate thermal overlay (for checking alignment) use: ``-to`` or ``--thermaloverlay``.
- To activate demo/debug mode (displays ambient temperature and thermal offset value) use: ``-d`` or ``--debug``.
- To flip the thermal image/data vertically (i.e. you have the thermal camera setup with the pins facing vertically) use: ``-ft`` or ``--flipthermal``.
- To use the offset created by the ``calibrator.py`` use: ``-uo`` or ``--useoffset``.
- To launch the detector in fullscreen mode use: ``-fs`` or ``--fullscreen``.

### Hotkeys:  
- `q` - Quit/exit the facemask detector completely.`
- `o` - Toggle the thermal overlay for adjusting the thermal camera aligment in use.
- `d` - Toggle the demo/debug mode.
- `u` - Increment the thermal offset by 0.25C.
- `j` - Decrement the thermal offset by 0.25C.
- `r` - Reset temperature offset to original value.

# How to run the thermal calibrator:  
- The ``thermal_grabber`` must be provided, and it will run as a subprogram to offer the thermal video stream.
- Use the following command to run the code ``python3 calibrator.py -argument ...``.

### Arguments:  
- ``-p`` or ``--path``, this is the relative path of thermal frame grabber to script working directory. The default value is ``thermal_grabber/build/thermal_grabber``
- ``-d`` or ``--debug`` is the argument to activate the debug mode, which is ``False`` as default.
- ``ft`` or ``flipthermal``, is used to flip the thermal video stream 180 degrees. The default is ``False``.

### Using instruction  
- The program requires the user to manually get a known target point temperature with their own devices, then select it on the thermal video stream on the top video panel with the targeting cursor to get the default temperature (uncalibrated). Next, user should input the correct temperature of the point to let program works out the offset value. By clicking on the ``Set`` button, the offset value will show up and saved in the program working folder as a ``.dat`` file, which will read by ``detector`` later. 