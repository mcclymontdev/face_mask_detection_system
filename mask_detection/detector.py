"""
The detector.py file is used to determine whether a person is wearing a mask or not.
It uses detect_and_predict_mask function that takes in a single frame from the live
stream, a face_net used to determine faces in the frame and mask_net used to determine
whether the faces detected are wearing masks or not. mask_net is a pre-trained model,
that has been trained using the learning_algo.py. When the aglorithm is run it starts
a live stream from which it uses every frame to determine whether the person in that
frame is wearing a mask. It displays a green box around the face if the person is wearing
the mask and a red one if they are not wearing a mask.
THe -f or --face flag can be used to provide the path to the face detector model. The -f
only needs to be used if another model is to be used to detect faces in a frame. The -m
or --model flag can be used to provide a path to the pre-trained mask detection model.
The -c or --confidence flag can be used to provide an optional probability threshold
that would override the default 50% to filter weak face detections.
"""

# import the necessary packages
import argparse
import os
import subprocess
import signal
import multiprocessing
import sys
from sys import platform

import cv2
import tensorflow as tf
import numpy as np
from screeninfo import get_monitors

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Returns visible faces and their facemask predicitons.
def detect_and_predict_mask(frame, face_net, mask_net, confidence_arg, shared_dict):
    # Grab the dimensions of the frame and then construct a blob from it
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))

    # Pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()

    # Initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence (i.e., probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than the minimum confidence
        if confidence > confidence_arg:
            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Wider margin for face
            (start_x, start_y, end_x, end_y) = (int(start_x*0.95), int(start_y*0.95), int(end_x*1.05), int(end_y*1.05))

            # Ensure the bounding boxes fall within the dimensions of the frame
            (start_x, start_y) = (max(0, start_x), max(0, start_y))
            (end_x, end_y) = (min(W - 1, end_x), min(H - 1, end_y))

            # Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[start_y:end_y, start_x:end_x]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((start_x, start_y, end_x, end_y))

    # Only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    shared_dict['facemask_detector_status'] = True

    # Return a 2-tuple of the face locations and their corresponding predections
    return (locs, preds)

# Parses the thermal grabber program's STDOUT as thermal data and debug information.
def thermal_grabber_worker(shared_dict, thermal_program_path, FLIP_THERMAL):
    # Opens a subprocess to the thermal grabber
    thermal_grabber = subprocess.Popen(["./thermal_grabber"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=thermal_program_path, bufsize=1)

    concat_data = ""
    for line in iter(thermal_grabber.stdout.readline, b''):
        # Allow the C++ program to handle its termination
        if shared_dict['thermal_process_terminate']:
            thermal_grabber.send_signal(signal.SIGINT)
            break

        # The following statements parse the raw STDOUT data as a numpy array.
        data = line.decode("utf-8").rstrip()
        if data == "":
            continue
        if data[0] == "[" and data[-1] == ";":
            shared_dict['resync_count'] = 0
            concat_data = data[1:-1] + ", "
        elif data[-1] == ";":
            shared_dict['resync_count'] = 0
            concat_data = concat_data + data[:-1] + ", "
        elif data[-1] == "]" and concat_data != "":
            shared_dict['resync_count'] = 0
            concat_data = concat_data + data[:-1]
            try:
                data_array = np.fromstring(concat_data, np.uint16, sep=',')
            except:
                if debug:
                    print("[WARNING] Received invalid thermal array (np.fromstring)")
                concat_data = ""
                continue
            if data_array.size != 19200:
                if debug:
                    print("[WARNING] Received invalid size of thermal array: " + str(data_array.size) + " != 19200")
                concat_data = ""
                continue
            thermal_data = np.reshape(data_array, (120,160))

            if FLIP_THERMAL:
                thermal_data = cv2.rotate(thermal_data, cv2.ROTATE_180)

            shared_dict['thermal_data'] = thermal_data

            # Create a copy of the thermal data to process as a thermal image frame
            thermal_frame = thermal_data.copy()

            # Resize thermal image for output
            cv2.normalize(thermal_frame, thermal_frame, 0, 255, cv2.NORM_MINMAX)
            thermal_width = int(thermal_frame.shape[1] * THERMAL_SCALE_FACTOR)
            thermal_height = int(thermal_frame.shape[0] * THERMAL_SCALE_FACTOR)
            thermal_dim = (thermal_width, thermal_height)
            thermal_frame = cv2.resize(thermal_frame, thermal_dim, interpolation = cv2.INTER_AREA)
            thermal_frame = cv2.cvtColor(thermal_frame,cv2.COLOR_GRAY2RGB)
            thermal_frame = np.uint8(thermal_frame)

            shared_dict['thermal_frame'] = thermal_frame

            concat_data = ""
        elif "," in data:
            shared_dict['resync_count'] = 0
            if data[-1] != ",":
                concat_data = concat_data + data + ","
            else:
                concat_data = concat_data + data
        elif "RESYNC" in data:
            concat_data = ""
            shared_dict['resync_count'] += 1
            print(data)
        else:
            concat_data = ""
            shared_dict['resync_count'] = 0
            print(data)

    print("[INFO] Thermal subprocess closed.")

def facemask_worker(shared_dict, face_arg, mask_arg, confidence_arg):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Allow GPU memory usage to change automatically
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxt_path = os.path.sep.join([face_arg, "deploy.prototxt"])
    weights_path = os.path.sep.join([face_arg, "res10_300x300_ssd_iter_140000.caffemodel"])
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    # Load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    mask_net = load_model(mask_arg)
    print("[INFO] face mask detector model loaded.")

    while True:
        if shared_dict['frame'] is not None:
            shared_dict['locs'], shared_dict['preds'] = detect_and_predict_mask(shared_dict['frame'], face_net, mask_net, confidence_arg, shared_dict)


if __name__ == '__main__':
    # Construct the argument parser and parse the arguments

    MAIN_DIR = os.getcwd()

    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
                    default="face_detector",
                    help="Path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
                    default="mask_detector.model",
                    help="Path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="Minimum probability to filter weak detections")

    # Thermal mode switch
    ap.add_argument("-t", "--thermal", dest="thermal", action="store_true", help="Activate thermal mode")
    # Thermal overlay switch
    ap.add_argument("-to", "--thermaloverlay", dest="thermaloverlay", action="store_true", help ="Display thermal overlay")
    # Debug mode switch
    ap.add_argument("-d", "--debug", dest="debug", action="store_true", help ="Activate debug mode")
    # Flip thermal switch
    ap.add_argument("-ft", "--flipthermal", dest="flipthermal", action="store_true", help ="Flip thermal image 108 degrees")
    # Use temperature offset config file
    ap.add_argument("-uo", "--useoffset", dest="useoffset", action="store_true", help ="Use offset configuration file")
    # Fullscreen switch
    ap.add_argument("-fs", "--fullscreen", dest="fullscreen", action="store_true", help ="Use fullscreen mode")

    ap.set_defaults(thermal=False, debug=False, flipthermal=False, useoffset=False, fullscreen=False)

    # Thermal program path setup
    thermal_program_path =  os.path.join(MAIN_DIR, "thermal_grabber/build/thermal_grabber")
    ap.add_argument("-tp", "--thermalprogram", type=str, default=thermal_program_path, help="Thermal program path")

    args = vars(ap.parse_args())

    debug = args["debug"]

    if platform == "linux":
        # FR:30Hz dFoV:78° Logitech C920
        webcam_cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        W, H = 800, 600
        webcam_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        webcam_cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        webcam_cap = cv2.VideoCapture(0)
        W, H = 1920, 1080
        webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

    # Fullscreen setup
    fullscreen_mode = args["fullscreen"]
    MONITOR_INFO = get_monitors()[0]
    print("[INFO] Fullscreen mode: " + str(fullscreen_mode))

    # Display video stream info
    print("[INFO] starting video stream...")
    print("[INFO] Video stream active: " + str(webcam_cap.isOpened()))

    # GUI constants
    # Facemask confidence level minimum
    FACEMASK_CONFIDENCE = 0.80

    GUI_FONT = cv2.FONT_HERSHEY_DUPLEX
    TEXT_SCALE = 1

    SUCCESS_COLOUR = (0, 255, 0) # Green
    WARNING_COLOUR = (0, 0, 255) # Red
    COLD_COLOUR = (255, 0, 9) # Blue
    UNKNOWN_COLOUR = (128, 128, 128) # Grey

    # Multiprocessing setup
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    shared_dict['facemask_detector_status'] = False
    shared_dict['frame'] = None
    shared_dict['locs'] = []
    shared_dict['preds'] = []
    shared_dict['thermal_process_terminate'] = False
    shared_dict['resync_count'] = 0

    # Thermal program setup
    THERMAL_MODE = args['thermal']
    thermal_program_path = args["thermalprogram"]

    # Thermal constant values
    # Based on temperature range from Lepton on HIGH gain mode [0-150°C]
    THERMAL_CONVERSION = 0.0092
    TEMP_MINIMUM = 32
    TEMP_MAXIMUM = 42
    TEMP_FEVER = 38
    THERMAL_SCALE_FACTOR = 5
    FLIP_THERMAL = args['flipthermal']

    temp_offset = 0
    # Copy of the original offset so we can reset it using the 'r' key later.
    TEMP_OFFSET_ORG = temp_offset

    # Get thermal settings from arguments
    thermal_overlay = args['thermaloverlay']
    USE_OFFSET = args['useoffset']

    # Check that thermal grabber program is present in the specified directory.
    # Check that the temperature offset configuration file is available if it is required.
    if THERMAL_MODE:
        print("[INFO] Thermal mode: ON")
        print("[INFO] Checking thermal program path...")
        if not os.path.exists(thermal_program_path):
            print("[ERROR] Provided thermal program path does not exist: " + thermal_program_path)
            sys.exit(1)
        else:
            print("[SUCCESS] Provided thermal program path exists.")

        if USE_OFFSET:
            print("[INFO] Getting temperature offset...")
            try:
                with open("TEMP_OFFSET.dat", "r") as offset_file:
                    temp_offset = float(offset_file.readline().strip())
                    TEMP_OFFSET_ORG = temp_offset
                    print("[SUCCESS] Thermal offset set: " + str(temp_offset))
            except Exception as e:
                print("[WARNING] There was an error retrieving your offset from TEMP_OFFSET", e)
    else:
        print("[INFO] Thermal mode: OFF")

    # Start the thermal subprocess
    if THERMAL_MODE:
        shared_dict['thermal_data'] = None
        shared_dict['thermal_frame'] = None
        thermal_grabber_process = multiprocessing.Process(target=thermal_grabber_worker, args=(shared_dict, thermal_program_path, FLIP_THERMAL))
        thermal_grabber_process.start()
        output_window = 'Mask Detecting Stream (Thermal)'
    else:
        output_window = 'Mask Detecting Stream'

    # Start the facemask subprocess
    facemask_process = multiprocessing.Process(target=facemask_worker, args=(shared_dict, args["face"], args["model"], args["confidence"]))
    facemask_process.start()

    # Prcoess the thermal data and take an average temperature from the forehead
    def process_thermal_data(thermal_data, start_point, end_point):
        measure_point_x, measure_point_y = (start_point[0] + ((end_point[0] - start_point[0]) // 2)), (start_point[1] + ((end_point[1] - start_point[1]) // 6))

        # Create a margin for a larger sample size.
        x_margin = ((end_x - start_x)/5) 
        y_margin = ((end_y - start_y)/20)

        # Scale the margin for use on the thermal data.
        x_margin_scaled = x_margin // THERMAL_SCALE_FACTOR
        y_margin_scaled = y_margin // THERMAL_SCALE_FACTOR 

        # Scale the measuring points for use on the thermal data.
        measure_point_x_scaled = measure_point_x // THERMAL_SCALE_FACTOR
        measure_point_y_scaled = measure_point_y // THERMAL_SCALE_FACTOR

        # Get all thermal data from within our margin box.
        measure_point_data = thermal_data[int(measure_point_y_scaled-y_margin_scaled):int(measure_point_y_scaled+y_margin_scaled), int(measure_point_x_scaled-x_margin_scaled):int(measure_point_x_scaled+x_margin_scaled)]

        avg_temp = np.average(measure_point_data)*THERMAL_CONVERSION+temp_offset
        label_avg_temp = str(round(avg_temp, 1)) + " C"

        temperature_bound = ((int((measure_point_x-x_margin)*frame_scale), int((measure_point_y-y_margin)*frame_scale)),
                             (int((measure_point_x+x_margin)*frame_scale), int((measure_point_y+y_margin)*frame_scale)))

        return avg_temp, label_avg_temp, temperature_bound

    # Resize frame to fit fullscreen, keeping aspect ratio
    def fullscreen_resize(frame):
        frame_height, frame_width = frame.shape[:2]
        scale_width = float(MONITOR_INFO.width)/float(frame_width)
        scale_height = float(MONITOR_INFO.height)/float(frame_height)

        if scale_height>scale_width:
            frame_scale = scale_width
        else:
            frame_scale = scale_height

        new_x, new_y = frame.shape[1]*frame_scale, frame.shape[0]*frame_scale
        frame = cv2.resize(frame,(int(new_x),int(new_y)), interpolation=cv2.INTER_NEAREST)

        # Allows us to pad the frame later to centre align it.
        frame_width_diff = MONITOR_INFO.width - new_x
        frame_height_diff = MONITOR_INFO.height - new_y

        return frame, frame_scale, (frame_width_diff, frame_height_diff)


    # MAIN DRIVER LOOP
    while True:
        # Read frame from webcam.
        _, webcam_frame = webcam_cap.read()

        # Get the thermal data/frame from the thermal grabber subprocess.
        thermal_status = False
        if THERMAL_MODE:
            # Retrieve thermal info from the subprocess
            thermal_data, thermal_frame, resync_count = shared_dict['thermal_data'], shared_dict['thermal_frame'], shared_dict['resync_count']

            # If there is no thermal data available or the thermal camera is in a resync state, turn thermal mode off temporarily
            if thermal_data is None or thermal_frame is None or resync_count > 6:
                thermal_status = False
            else:
                thermal_status = True
        # If thermal mode is not active set default values.
        else:
            thermal_data, thermal_frame, resync_count = None, None, 0

        # Pass frame for processing
        shared_dict['frame'] = webcam_frame
        
        # Show the thermal frame overlayed ontop of the webcam frame.
        if thermal_status and thermal_overlay:
            alpha = 0.35
            beta = (1.0 - alpha)
            output_frame = cv2.addWeighted(thermal_frame, alpha, webcam_frame, beta, 0.0)
        else:
            output_frame = webcam_frame

        # Resize fullscreen output, keeping aspect ratio intact.
        frame_scale = 1
        if fullscreen_mode:
            output_frame, frame_scale, frame_diff = fullscreen_resize(output_frame)
        #TEXT_SCALE = 0.5 + (0.5 * frame_scale)
        TEXT_SCALE = frame_scale

        # If in debug mode show the ambient/room temperature.
        if thermal_status and debug:
            average_temperature = np.average(thermal_data)*THERMAL_CONVERSION+temp_offset
            cv2.putText(output_frame, "Ambient: " + str(round(average_temperature,1)) + " C", (int(35 * TEXT_SCALE), int(35 * TEXT_SCALE)),
            GUI_FONT, TEXT_SCALE, (255,255,255), 2, cv2.LINE_AA)

        # If in debug mode show the thermal offset value.
        if debug and temp_offset != 0:
            cv2.putText(output_frame, "Offset: " + str(temp_offset) + " C", (int(35 * TEXT_SCALE), int(70 * TEXT_SCALE)),
            GUI_FONT, TEXT_SCALE, (255,255,255), 2, cv2.LINE_AA)

        # Detect faces in the frame and determine if they are wearing a face mask or not.
        (locs, preds) = shared_dict['locs'], shared_dict['preds']

        # Loop over the detected face locations and their corresponding locations.
        for (box, pred) in zip(locs, preds):
            # Unpack the face bounding box and facemask predictions.
            (start_x, start_y, end_x, end_y) = box
            (withoutMask, mask) = pred

            # If there is thermal data available, get the forehead temperature.
            avg_temp = None
            if thermal_status:
                avg_temp, label_avg_temp, temperature_bound = process_thermal_data(thermal_data, (start_x, start_y), (end_x, end_y))
            # If there is no thermal data available display a message to the user prompting them to wait.
            elif THERMAL_MODE:
                cv2.rectangle(output_frame, (0, 0), (output_frame.shape[1], output_frame.shape[0]//5), (0,0,0), -1, cv2.LINE_AA)
                cv2.putText(output_frame, "Waiting for thermal camera...", (output_frame.shape[0]//10, output_frame.shape[0]//10),
                GUI_FONT, 1, (255,255,255), 2, cv2.LINE_AA)

            # Scale bounding box by the fullscreen scaling
            if fullscreen_mode:
                start_x, start_y, end_x, end_y = int(start_x * frame_scale), int(start_y * frame_scale), int(end_x * frame_scale), int(end_y * frame_scale)
            
            # Determine the class label and color we'll use to draw the bounding box and text
            mask_label = "Mask" if mask > withoutMask else "No Mask"
            mask_colour = SUCCESS_COLOUR if mask_label == "Mask" else WARNING_COLOUR

            # Confidence interval for the predictions.
            if mask < FACEMASK_CONFIDENCE and withoutMask < FACEMASK_CONFIDENCE:
                mask_label = "Look at the camera please!"
                mask_colour = UNKNOWN_COLOUR

            # Display appropriate messages to the user
            elif thermal_status:
                # If wearing mask and normal body temperature
                if mask > FACEMASK_CONFIDENCE and (avg_temp > TEMP_MINIMUM and avg_temp < TEMP_FEVER):
                    message_label = "You may enter!"
                    temperature_colour = SUCCESS_COLOUR
                    message_colour = SUCCESS_COLOUR
                # If not wearing a mask and normal body temperature
                elif withoutMask > FACEMASK_CONFIDENCE and (avg_temp > TEMP_MINIMUM and avg_temp < TEMP_FEVER):
                    message_label = "Please wear a mask!"
                    temperature_colour = SUCCESS_COLOUR
                    message_colour = WARNING_COLOUR
                # Fever alert (outside of normal body temperature)
                elif (avg_temp >= TEMP_FEVER):
                    message_label = "FEVER WARNING!"
                    secondary_label = "DO NOT ENTER"
                    temperature_colour = WARNING_COLOUR
                    message_colour = WARNING_COLOUR
                    mask_colour = WARNING_COLOUR
                    # Warning outline to differentiate from background
                    cv2.putText(output_frame, secondary_label, (start_x, int(end_y + (30 * TEXT_SCALE))),
                    cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, (0,), int(4*TEXT_SCALE), cv2.LINE_AA)
                    # Large warning
                    cv2.putText(output_frame, secondary_label, (start_x, int(end_y + (30 * TEXT_SCALE))),
                    cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, message_colour, int(2*TEXT_SCALE), cv2.LINE_AA)
                # User is too cold to get accurate temperature
                else:
                    message_label = "Heat up and try again!"
                    temperature_colour = COLD_COLOUR
                    message_colour = COLD_COLOUR
                
                # Display temperature box
                cv2.rectangle(output_frame, temperature_bound[1], temperature_bound[0], temperature_colour, 1, cv2.LINE_AA)

                # Message outline to differentiate from background
                cv2.putText(output_frame, message_label, (start_x, start_y - int(70 * TEXT_SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, (0,), int(4*TEXT_SCALE), cv2.LINE_AA)
                # Display message assigned above
                cv2.putText(output_frame, message_label, (start_x, start_y - int(70 * TEXT_SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, message_colour, int(2*TEXT_SCALE), cv2.LINE_AA)

                # Temperature outline to differentiate from background
                cv2.putText(output_frame, label_avg_temp, (start_x, start_y - int(40 * TEXT_SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, (0,), int(4*TEXT_SCALE), cv2.LINE_AA)
                # Display body temperature
                cv2.putText(output_frame, label_avg_temp, (start_x, start_y - int(40 * TEXT_SCALE)),
                cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, temperature_colour, int(2*TEXT_SCALE), cv2.LINE_AA)

            # Include the probability in the label
            mask_label = "{}: {:.2f}%".format(mask_label, max(mask, withoutMask) * 100)

            # Label outline to differentiate from background
            cv2.putText(output_frame, mask_label, (start_x, start_y - int(10 * TEXT_SCALE)), cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, (0,), int(4*TEXT_SCALE), cv2.LINE_AA)
            # Display the label and bounding box rectangle on the output
            cv2.putText(output_frame, mask_label, (start_x, start_y - int(10 * TEXT_SCALE)), cv2.FONT_HERSHEY_DUPLEX, TEXT_SCALE, mask_colour, int(2*TEXT_SCALE), cv2.LINE_AA)
            cv2.rectangle(output_frame, (start_x, start_y), (end_x, end_y), mask_colour, 2, cv2.LINE_AA)

        # Facemask detection loading screen
        if not shared_dict['facemask_detector_status']:
            cv2.rectangle(output_frame, (0, 0), (output_frame.shape[1], output_frame.shape[0]//5), (0,0,0), -1, cv2.LINE_AA)
            cv2.putText(output_frame, "Face mask detection is loading...", (output_frame.shape[0]//10, output_frame.shape[0]//10),
            GUI_FONT, TEXT_SCALE, (255,255,255), 2, cv2.LINE_AA)

        # Draw fullscreen window or standard window depending on mode
        if fullscreen_mode:
            # Centre align frame
            frame_h_padding = int(frame_diff[1]//2)
            frame_w_padding = int(frame_diff[0]//2)
            output_frame = cv2.copyMakeBorder(output_frame, frame_h_padding, frame_h_padding, frame_w_padding, frame_w_padding, cv2.BORDER_CONSTANT, value=(0,))
            cv2.namedWindow(output_window, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(output_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(output_window, cv2.WND_PROP_AUTOSIZE)
            cv2.setWindowProperty(output_window, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_NORMAL)
        cv2.imshow(output_window, output_frame)

        # Get key press
        key = cv2.waitKey(1)

        # Quit key
        if key == ord('q'):
            break

        # Toggle debug mode (ambient temperature and offset shown)
        if key == ord('d'):
            debug = not debug
            print("[INFO] Debug mode: " + str(debug))

        # Toggle thermal overlay
        elif key == ord('o') and THERMAL_MODE:
            thermal_overlay = not thermal_overlay
            print("[INFO] Thermal overlay: " + str(thermal_overlay))

        # Change thermal offset
        elif key == ord('u') and THERMAL_MODE:
            temp_offset += 0.25
            print("[INFO] Temperature offset (+0.25 C): " + str(temp_offset) + " C")
        elif key == ord('j') and THERMAL_MODE:
            temp_offset -= 0.25
            print("[INFO] Temperature offset (-0.25 C): " + str(temp_offset) + " C")
        elif key == ord('r') and THERMAL_MODE:
            temp_offset = TEMP_OFFSET_ORG
            print("[INFO] Temperature offset reset to: " + str(temp_offset) + " C")

        # Toggle fullscreen
        elif key == ord('f'):
            fullscreen_mode = not fullscreen_mode
            cv2.destroyWindow(output_window)
            print("[INFO] Fullscreen mode: " + str(fullscreen_mode))

    print("[INFO] Thank you for using mask detection!")

    # Clean up and shutdown
    facemask_process.terminate()

    if THERMAL_MODE:
        shared_dict['thermal_process_terminate'] = True
        thermal_grabber_process.join()

    cv2.destroyAllWindows()
