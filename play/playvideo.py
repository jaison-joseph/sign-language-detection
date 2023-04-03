import cv2

# Open the video file
cap = cv2.VideoCapture('countdown.webm')

# Check if the video file was opened successfully
if not cap.isOpened():
    print('Error opening video file')

# Play the video frame by frame
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    
    # If a frame was successfully read
    if ret:
        # Display the frame
        cv2.imshow('Video', frame)
        
        # Exit if the user presses 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        # Exit the loop if no more frames are available
        break

# Release the video file and close the window
cap.release()
cv2.destroyAllWindows()