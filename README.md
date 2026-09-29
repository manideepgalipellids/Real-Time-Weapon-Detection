# Real-Time Weapon Detection

AI-powered weapon detection using YOLO and computer vision.

## Live Demo

https://real-time-weapon-detection-keynhx7wpn6ypjsz5jwifa.streamlit.app/

## Features

- Image upload for weapon detection
- Real-time camera detection
- Custom-trained YOLO model
- Adjustable detection confidence
- Detection count and confidence score
- Knife, Gun, and Rifle detection
- Streamlit web interface
- WebRTC-based live camera processing

## Technologies

- Python
- YOLO
- Ultralytics
- OpenCV
- Streamlit
- Streamlit WebRTC
- PyTorch
- NumPy
- Pillow
- PyAV

## Supported Classes

| Class | Description |
|---|---|
| Knife | Detects knives |
| Gun | Detects guns |
| Rifle | Detects rifles |

## How It Works

The system uses a custom-trained YOLO model. Uploaded images and live camera frames are processed by the model, which draws bounding boxes around detected weapons and displays their confidence scores.

## Model

Custom-trained YOLO model:

models/best.pt

Detection confidence can be adjusted between 0.10 and 0.90. The default is 0.25.

## Run Locally

Clone the repository:

git clone https://github.com/manideepgalipellids/Real-Time-Weapon-Detection.git

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app/app.py

## Deployment

The application is deployed using Streamlit Community Cloud.

Live Demo:

https://real-time-weapon-detection-keynhx7wpn6ypjsz5jwifa.streamlit.app/

## Limitations

- Detection performance depends on the trained model and input image.
- Detection confidence affects false detections and missed detections.
- Camera performance depends on available computing resources and network conditions.
- This project is intended as a computer-vision research and demonstration application.

## Future Improvements

- Improve the training dataset and model performance
- Add more weapon classes
- Improve low-light detection
- Add object tracking
- Add alerts and notifications
- Add detection logs and analytics

## Author

Manideep

GitHub: https://github.com/manideepgalipellids