import cv2, time, threading, csv, datetime, sys, os
import matplotlib.pyplot as plt
import numpy as np
from matriz_conversion import conversion_2p
from database import Database

# Access the argument
argument = sys.argv[1]  # The first argument is at index 1
userKey = sys.argv[2]

try:
    # Instantiate the Database connection object
    obj = Database()
    print("Database connection object created successfully.")
except Exception as e:
    print("Error occurred while creating the Database connection object:", str(e))

# Initialize video capture for User Camera
capUser = cv2.VideoCapture(2)
# Initialize 2 tracking cameras, cap1 (x,y) axis, cap2 (z) axis
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

# Initialize video capture for User Camera, cap1 (x,y) axis, and cap2 (z) axis
cameras = {'User_Camera': capUser, 'XY_Cam': cap1, 'Z_Cam': cap2}

global finish
global file_name

# Check if each camera has been successfully opened and is capturing video
for cam_name, cam in cameras.items():
    if cam.isOpened():
        print(f"{cam_name} is capturing video")
    else:
        print(f"Failed to open {cam_name}")
      
def detect_color(frame, lower_color, upper_color, color_bgr):
    x=0
    y=0
    frame = brillo(frame)
    # convert to hsv color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # create mask for color
    mask = cv2.inRange(hsv, lower_color, upper_color)
    # Apply a series of dilations and erosions to eliminate any small blobs left in the mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    # Find contours and centroid in the binary image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame=center_object(frame, contours) # Centroid
    
    if contours:
        # draw bounding box around object
        object_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(object_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.putText(frame, str(x) + "," + str(y),
            (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 55), 1)
    return frame, x, y

def center_object(frame, contours):
    if len(contours) > 0:
        # Find the largest contour, assuming it is the object
        max_contour = max(contours, key=cv2.contourArea)
        # Calculate the center of mass of the object
        M = cv2.moments(max_contour)
        # Enters if the center is different than 0
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            # Draw a circle at the center of the object
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
    return frame

def val_coords(x, y, xval, yval):
    x_in_interval = xval-10 <= x <= xval+10
    y_in_interval = yval-10 <= y <= yval+10
    return x_in_interval and y_in_interval

def gen_graph():
    global file_name
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    x2 = []
    y2 = []
    z2 = []
    
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))
            z.append(float(row[2]))            
            x2.append(float(row[3]))
            y2.append(float(row[4]))
            z2.append(float(row[5]))

    ax.plot(x, y, z, c='blue')
    ax.plot(x2, y2, z2, c='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Save the graph as an image
    base_name = os.path.splitext(os.path.basename(file_name))[0]
    img_path = os.path.join("Graph3D/", base_name + ".png")
    fig.savefig(img_path)
    plt.close(fig)  # Close the figure to free up memory

    file_directory = os.getcwd().replace("\\", "/")+"/"+file_name
    graph_directory = os.getcwd().replace("\\", "/")+"/"+img_path

    obj.upload_files_to_endpoint('http://143.110.148.122:8999', file_directory, graph_directory)
    _ = obj.addDataBase(file_name, userKey, argument, "3")

def brillo(img):
    # Aplicar brillo para reflejar blancos
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    contrast = 1.0
    brightness = 20
    frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    return frame

def XYZ_Webcam():
    global finish
    global file_name
    
    # Format the date as YYYY-MM-DD
    filename_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Construct the file name with the date variable
    file_name = f'Datos_Sutura/{argument}_{filename_date}.csv'
    
    start_time = time.time()

    while True:
        # read frames from both cameras
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        elapsed_time = time.time() - start_time

        # detect colors in frame1 (blue/red)
        frame1, cx1B, cy1B  = detect_color(frame1, (100, 50, 50), (130, 255, 255), (255, 0, 0))
        frame1, cx1R, cy1R = detect_color(frame1, (0, 50, 50), (10, 255, 255), (0, 0, 255))
        # detect colors in frame2 (blue/red)
        frame2, cx2B, cy2B = detect_color(frame2, (100, 50, 50), (130, 255, 255), (255, 0, 0))
        frame2, cx2R, cy2R = detect_color(frame2, (0, 50, 50), (10, 255, 255), (0, 0, 255))
        
        if all([cx1B, cy1B, cx2B, cy2B,cx1R, cy1R, cx2R, cy2R]):
            xCmB, yCmB, zCmB, xCmR, yCmR, zCmR = conversion_2p(cx1B, cy1B, cx2B, cy2B,cx1R, cy1R, cx2R, cy2R)
            #print(xCmB, yCmB, zCmB, xCmR, yCmR, zCmR, elapsed_time)
            
            with open(file_name, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([xCmB, yCmB, zCmB, xCmR, yCmR, zCmR, elapsed_time])
        
        #show the frames from both cameras
        #cv2.imshow("Camera XY", frame1)
        #cv2.imshow("Camera Z", frame2)

        # Check for coordinates and time to exit thread
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        # Surdo if elapsed_time>=20 and val_coords(cx2B,cy2B, 152, 342) and val_coords(cx2R, cy2R, 405, 387):
        if elapsed_time>=20 and val_coords(cx2B,cy2B, 160, 380) and val_coords(cx2R, cy2R, 418, 353):
            finish=False
            break

def main():
    global finish
    finish=True
    # Initialize flag and time for camera 2
    process_cam2 = True
    show_animation = False
    centrar=False

    animation_start_time = 0
    animation_duration = 3
    prev_time = time.time()

    # Loop to read frames from both cameras
    while finish:
        # Read frame from camera 1
        ret1, frame1 = capUser.read()
     
        # Read frame every second from camera 2 only if process_cam2 is True
        if process_cam2 and time.time()-prev_time>=1:
            ret2, frame2 = cap2.read()

            # Check if frame was read successfully
            if ret2:
                # detect colors in frame2 (blue/red)
                frame2, cx2B, cy2B = detect_color(frame2, (100, 50, 50), (130, 255, 255), (255, 0, 0))
                frame2, cx2R, cy2R = detect_color(frame2, (0, 50, 50), (10, 255, 255), (0, 0, 255))

                # Surdo if val_coords(cx2B,cy2B, 152, 342) and val_coords(cx2R, cy2R, 405, 387):
                if val_coords(cx2B,cy2B, 160, 380) and val_coords(cx2R, cy2R, 418, 353):
                    # Start the run_with_loading_bar function in a separate thread
                    process_cam2 = not process_cam2
                    show_animation = True
                    animation_start_time = time.time()
                    
            # update the previous time for camera 2
            prev_time = time.time()

        # Check if animation should be shown
        if show_animation:
            current_time = time.time()
            elapsed_time = current_time - animation_start_time

            # Show animation for animation_duration seconds
            if elapsed_time < animation_duration:
                # Add "START" text in the center of the frame
                text = "COMIENZA"
                text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
                text_x = (frame1.shape[1] - text_size[0]) // 2
                text_y = (frame1.shape[0] + text_size[1]) // 2
                # Draw the text on the frame
                cv2.putText(frame1, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            else:
                show_animation = False
                # Start the XYZ_Webcam function 
                t1 = threading.Thread(target=XYZ_Webcam)
                t1.start()
                
        # Check for key press 'q' to exit program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Check if frame was read successfully
        if ret1:
            frame1 = cv2.resize(frame1, (1000,800))
            # Show frame from camera 1
            cv2.imshow(str("Sutura "+argument), frame1)

    # Release video captures and destroy windows
    cap1.release()
    cap2.release()
    capUser.release()
    cv2.destroyAllWindows()
    gen_graph()

if __name__ == "__main__":
    main()