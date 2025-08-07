import cv2
import numpy as np
import time
import os
import pygame

EYE_CLOSED_CONSEC_FRAMES = 25
# Re-typed the path to remove the invisible control character
ALARM_SOUND_PATH = r'D:\pro\alarm.wav'
ALARM_ON = False
COUNTER = 0

def play_alarm():
    global ALARM_ON
    if not ALARM_ON and pygame.mixer.music.get_busy() == 0:
        try:
            pygame.mixer.music.play(-1)
            ALARM_ON = True
        except pygame.error:
            print("Warning: Attempted to play music that was not loaded. Skipping.")

def stop_alarm():
    global ALARM_ON
    if ALARM_ON:
        pygame.mixer.music.stop()
        ALARM_ON = False

def main():
    print("[INFO] Loading Haar Cascade models...")
    face_cascade = cv2.CascadeClassifier('frontalface.xml')
    eye_cascade = cv2.CascadeClassifier('eye_detector.xml')
    
    if face_cascade.empty() or eye_cascade.empty():
        print("Error: Could not load Haar Cascade XML files. Please check the paths.")
        return

    print("[INFO] Initializing pygame for audio...")
    pygame.mixer.init()
    
    music_loaded = False
    try:
        pygame.mixer.music.load(ALARM_SOUND_PATH)
        music_loaded = True
    except pygame.error:
        print(f"Warning: Could not load alarm sound file at '{ALARM_SOUND_PATH}'. "
              "Please ensure the file exists and is a valid format (e.g., .wav, .mp3).")

    print("[INFO] Starting video stream...")
    video_stream = cv2.VideoCapture(0)

    global COUNTER

    while video_stream.isOpened():
        ret, frame = video_stream.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        eyes_detected = False

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10))
            
            if len(eyes) >= 2:
                eyes_detected = True
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        if not eyes_detected:
            COUNTER += 1
            if COUNTER >= EYE_CLOSED_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if music_loaded:
                    play_alarm()
        else:
            COUNTER = 0
            if music_loaded:
                stop_alarm()
        
        cv2.imshow("Drowsiness Detection System", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    print("[INFO] Shutting down...")
    stop_alarm()
    cv2.destroyAllWindows()
    video_stream.release()

if __name__ == "__main__":
    main()
