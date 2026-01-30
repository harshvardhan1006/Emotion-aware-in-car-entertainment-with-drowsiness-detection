import cv2
import dlib
from scipy.spatial import distance
import pygame  # For playing alarm sound
from collections import deque
import time

# Initialize the pygame mixer for sound playback
pygame.mixer.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound("music.wav")  # Ensure you have the correct file

# Load the face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")

# Calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Parameters for drowsiness detection
EAR_THRESHOLD = 0.25  # Dynamic threshold
CONSEC_FRAMES = 10  # Fewer frames for faster detection
ALARM_DELAY = 1 # Alarm triggers after 2 seconds of closed eyes
ALARM_COOL_OFF_TIME = 5  # Cool-off time between alarms (in seconds)

# Track EAR values for smoothing
ear_history = deque(maxlen=10)  # Increased history length for better smoothing
drowsy_frames = 0
alarm_playing = False
alarm_start_time = None
smoothed_ear = 0

# Smoothed EAR using exponential smoothing
alpha = 0.1  # Weight factor for smoothing

# Drowsiness detection function
def detect_drowsiness(frame):
    global drowsy_frames, alarm_playing, alarm_start_time, smoothed_ear

    # Convert to grayscale and apply CLAHE for better lighting in low-light conditions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Detect faces in the grayscale image
    faces = detector(gray)

    for face in faces:
        # Get landmarks for the detected face
        landmarks = predictor(gray, face)

        # Extract eye landmarks (left and right eyes)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Calculate Eye Aspect Ratio (EAR) for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Update EAR history and calculate smoothed EAR using exponential smoothing
        ear_history.append(ear)
        smoothed_ear = alpha * ear + (1 - alpha) * smoothed_ear

        # Check for drowsiness based on the smoothed EAR
        if smoothed_ear < EAR_THRESHOLD:
            drowsy_frames += 1
            if drowsy_frames >= CONSEC_FRAMES:
                if alarm_start_time is None:
                    alarm_start_time = time.time()
                elif time.time() - alarm_start_time >= ALARM_DELAY:
                    # Display "DROWSY!" on the frame
                    cv2.putText(frame, "DROWSY!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Play alarm sound if it's not already playing
                    if not alarm_playing:
                        alarm_sound.play()
                        alarm_playing = True
        else:
            drowsy_frames = 0
            alarm_playing = False
            alarm_start_time = None

    return frame

# Main function to test drowsiness detection
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform drowsiness detection on the current frame
        frame = detect_drowsiness(frame)

        # Display the result
        cv2.imshow("Drowsiness Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
