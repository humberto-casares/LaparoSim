import cv2, csv
import matplotlib.pyplot as plt
from matriz_conversion import conversion_2p
import datetime
import numpy as np

global finish
global file_name

# Initialize 2 tracking cameras, cap1 (x,y) axis, cap2 (z) axis
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)

# Initialize video capture for User Camera, cap1 (x,y) axis, and cap2 (z) axis
cameras = {'XY_Cam': cap1, 'Z_Cam': cap2}

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

def brillo(img):
    # Aplicar brillo para reflejar blancos
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    contrast = 1.0
    brightness = 20
    frame[:,:,2] = np.clip(contrast * frame[:,:,2] + brightness, 0, 255)
    frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
    return frame

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
    plt.show()

def XYZ_Webcam():
    global finish
    global file_name
    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date as YYYY-MM-DD
    filename_date = now.strftime('%Y-%m-%d_%H-%M-%S')
    # Construct the file name with the date variable
    file_name = f'Transferencia_{filename_date}.csv'
    con=0
    while True:
        # read frames from both cameras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 and ret2:
            # detect colors in frame1 (blue/red)
            frame1, cx1B, cy1B  = detect_color(frame1, (100, 50, 50), (130, 255, 255), (255, 0, 0))
            frame1, cx1R, cy1R = detect_color(frame1, (0, 50, 50), (10, 255, 255), (0, 0, 255))
 
             # detect colors in frame2 (blue/red)
            frame2, cx2B, cy2B = detect_color(frame2, (100, 50, 50), (130, 255, 255), (255, 0, 0))
            frame2, cx2R, cy2R = detect_color(frame2, (0, 50, 50), (10, 255, 255), (0, 0, 255))
            
            if all([cx1B, cy1B, cx2B, cy2B,cx1R, cy1R, cx2R, cy2R]):
                xCmB, yCmB, zCmB, xCmR, yCmR, zCmR = conversion_2p(cx1B, cy1B, cx2B, cy2B,cx1R, cy1R, cx2R, cy2R)
                print(xCmB, yCmB, zCmB, xCmR, yCmR, zCmR,"\n------------------ \n")

                '''
                with open(file_name, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([xCmB, yCmB, zCmB, xCmR, yCmR, zCmR])
                '''

            #show the frames from both cameras
            cv2.imshow("Camera XY", frame1)
            cv2.imshow("Camera Z", frame2)

        # Check for coordinates to exit thread
        if cv2.waitKey(1) & 0xFF == ord('t'):
        #if val_coords(cx2R, cy2R, 350, 320):
            break

    # Release video captures and destroy windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    #gen_graph()

if __name__ == "__main__":
    XYZ_Webcam()