import os
import sys
import cv2
import time
import signal
import argparse
import numpy as np
import subprocess
import multiprocessing
import tkinter.filedialog
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from sys import platform

def thermal_grabber_worker(shared_dict, thermal_program_path, FLIP_THERMAL):
    thermal_grabber = subprocess.Popen(["./thermal_grabber"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=thermal_program_path, bufsize=1)

    concat_data = ""
    for line in iter(thermal_grabber.stdout.readline, b''):
        # Allow the C++ program to handle its termination
        if shared_dict['thermal_process_terminate']:
            thermal_grabber.send_signal(signal.SIGINT)
            break
        
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
            
            # Create a copy of the thermal data for future use
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

def img_convert(image):
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image

def data_process(event):
    global TEMP_OFFSET, THERMAL_CONVERSION

    current_temp = round(shared_dict["thermal_data"][event.y//THERMAL_SCALE_FACTOR, event.x//THERMAL_SCALE_FACTOR]*THERMAL_CONVERSION,2)
    current_temp_msg = "Position = ({0},{1})\n".format(event.x, event.y) + "Current Temperature = " + str(current_temp)
    
    curr_temp_label.grid(row=3, columnspan=2)
    curr_temp_label.configure(text=current_temp_msg)
    
    root.update()
    
    tk.Label(root, text="Real Temperature: ").grid(row=4, column=0, sticky="E")
    input_entry = tk.Entry()
    input_entry.grid(row=4, column=1, sticky="W")

    def get():
        global TEMP_OFFSET
        real_temp = input_entry.get()
        try:
            TEMP_OFFSET = str(round(float(real_temp)-float(current_temp), 2))
            offset_msg = "Temperature Offset: " + TEMP_OFFSET
            offset_label.grid(row=5, columnspan=2)

            offset_label.configure(text=offset_msg)
            
            try: 
                offset_file = open("TEMP_OFFSET.dat", "w+")
                offset_file.write(TEMP_OFFSET)
                offset_file.close()
            except:
                print("[ERROR] Could not write temperature offset to file. Do you have permission to write to this folder?")
        except:
            print("[WARNING] Invalid input for real temperature")

    new_offset_temp_msg = "Temperature plus offset: " + str(float(current_temp) + float(TEMP_OFFSET))
    offset_temp_label.grid(row=6, columnspan=2)
    offset_temp_label.configure(text=new_offset_temp_msg)

    comform_button = tk.Button(root, text="Set", command=get)
    comform_button.grid(row=4, column=1, sticky="E")

if __name__ == "__main__":

    MAIN_DIR = os.getcwd()

    ap = argparse.ArgumentParser()

    # Thermal grabber path
    ap.add_argument("-p", "--path", type=str,
                    default="thermal_grabber/build/thermal_grabber",
                    help="Relative path of thermal frame grabber to script working directory.")
    # Debug mode switch
    ap.add_argument("-d", "--debug", dest="debug", action="store_true", help="Activate debug mode")
    # Flip thermal switch
    ap.add_argument("-ft", "--flipthermal", dest="flipthermal", action="store_true",
                    help="Flip thermal image 180 degrees")

    ap.set_defaults(debug=False, flipthermal=False)
    args = vars(ap.parse_args())

    # Variables initialization
    unpause = True

    THERMAL_CONVERSION = 0.0092
    TEMP_OFFSET = 0
    TEMP_MINIMUM = 32.5
    TEMP_MAXIMUM = 42
    THERMAL_SCALE_FACTOR = 5
    FLIP_THERMAL = args['flipthermal']
    DEBUG = args["debug"]

    thermal_program_path = os.path.join(MAIN_DIR, args["path"])

    # Multiprocessing setup
    manager = multiprocessing.Manager()
    shared_dict = manager.dict()

    shared_dict['thermal_process_terminate'] = False
    shared_dict['thermal_data'] = None
    shared_dict['thermal_frame'] = None
    shared_dict['resync_count'] = 0
    thermal_grabber_process = multiprocessing.Process(target=thermal_grabber_worker,
                                                      args=(shared_dict, thermal_program_path, FLIP_THERMAL))
    thermal_grabber_process.start()

    # GUI rendering
    root = tk.Tk()
    root.title("Thermal Calibrator")
    curr_temp_label = tk.Label(root)
    offset_label = tk.Label(root)
    offset_temp_label = tk.Label(root)

    thermal_img_panel = tk.Label(root, bd=0, image=None, cursor="target")
    thermal_img_panel.grid(row=0, padx=5, pady=5, columnspan=3)

    running = True
    while running:
        thermal_data, thermal_frame, resync_count = shared_dict['thermal_data'], shared_dict['thermal_frame'], shared_dict['resync_count']
        if thermal_data is None or thermal_frame is None or resync_count > 6:
            thermal_status = False
        else:
            thermal_status = True

        if thermal_status:
            try:
                image = img_convert(thermal_frame)
                thermal_img_panel.configure(image=image)
                thermal_img_panel.image = image
                thermal_img_panel.bind('<Button-1>', data_process)
                root.update()
            except:
                break

    def on_closing():
        print("[INFO] Closing...")
        running = False
        shared_dict['thermal_process_terminate'] = True
        thermal_grabber_process.join()
        print("[INFO] Closed successfully!")
        
    try:
        root.protocol("WM_DELETE_WINDOW", on_closing())
    except:
        pass