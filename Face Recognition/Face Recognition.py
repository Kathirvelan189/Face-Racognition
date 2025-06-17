import cv2

# Load Haar cascades for face and eye detection

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Validate that cascade files loaded correctly

if face_cascade.empty() or eye_cascade.empty():
    print("Error: Could not load Haar cascade XML files. Ensure they are in the same directory.")
    exit()

# Open the default webcam

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Run face and eye detection loop

while True:
    ret, img = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    cv2.imshow('Face & Eye Detection', img)

    # Exit on pressing 'ESC'
    
    if cv2.waitKey(30) & 0xFF == 27:
        break

# Release resources

cap.release()
cv2.destroyAllWindows()
