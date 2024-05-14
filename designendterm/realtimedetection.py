import cv2
from keras.models import model_from_json
import numpy as np
import subprocess

# Load pre-trained model for emotion detection
json_file = open("facialemotionrecognition.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionrecognition.h5")

# Load Haar cascade classifier for face detection
haar_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

# Define function to extract features from image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Define function to generate text message based on detected emotion
def generate_text_message(emotion):
    messages = {
        'angry': "You seem angry!",
        'fear': "It looks like you're feeling fearful.",
        'happy': "You're wearing a happy expression!",
        'sad': "You appear to be sad.",
        'surprise': "You seem surprised!"
    }
    return messages.get(emotion, "I'm not sure how you're feeling.")

# Initialize webcam
webcam = cv2.VideoCapture(0)

# Labels for emotions
labels = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'sad', 4: 'surprise'}

# Main loop for video processing
while True:
    # Read frame from webcam
    ret, frame = webcam.read()

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract face region
        face_image = gray[y:y+h, x:x+w]

        # Resize face image to match input size of the model
        face_image = cv2.resize(face_image, (48, 48))

        # Extract features and make prediction
        img = extract_features(face_image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Generate text message based on detected emotion
        text_message = generate_text_message(prediction_label)

        # Use subprocess to speak the text message
        subprocess.call(['say', text_message])

        # Display emotion label and text message on the frame
        cv2.putText(frame, prediction_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, text_message, (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Emotion Detection", frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
webcam.release()
cv2.destroyAllWindows()
