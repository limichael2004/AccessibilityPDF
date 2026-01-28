
# AccessibilityPDF

#YOUTUBE WALKTHROUGH!
https://youtu.be/4hv1thAHtE0

## About the Project
I created AccessibilityPDF to explore how computer vision can support digital autonomy for individuals with fine motor limitations. For many people with conditions like ALS, spinal cord injuries, or severe arthritis, standard input devices like mice and keyboards create barriers to basic tasks like reading. This project aims to remove those barriers by turning a standard webcam into a fully functional input controller. It allows users to navigate, zoom, and read PDF documents entirely through facial expressions and head movements.

## How it Works
The application uses the MediaPipe library to track facial geometry in real-time. It maps specific physical actions—like turning the head, raising eyebrows, or smiling—to digital commands within the PDF reader.

1. Geometric Tracking: The system calculates the aspect ratios of the eyes and mouth to detect intentional gestures rather than just random movement.

2. Privacy-First: All processing happens locally on your computer. No video feed or biometric data is ever stored or sent to the cloud.

3. Adaptive Configuration: I realized that "accessibility" means something different for everyone. The software includes a configuration suite that allows users to remap every gesture and adjust sensitivity thresholds. This ensures the tool adapts to the user's specific range of motion, rather than forcing the user to adapt to the tool.

## Core Features

1. Gesture Mapping: Users can link actions like "Look Left" or "Long Blink" to commands like "Previous Page" or "Toggle Fullscreen."

2. Sensitivity Tuning: Thresholds for head rotation and gesture duration are fully adjustable to account for tremors or limited mobility.

3. Integrated Reader: A custom-built PDF viewer that responds instantly to the vision engine without needing external drivers.

4. Visual Feedback: A discreet camera overlay helps users stay aligned without obstructing the document view.
## Technical Stack/Acknowledgements

Python 3.9+

MediaPipe: For high-fidelity facial landmark tracking.

OpenCV: For image processing and pose estimation.

Tkinter: For the user interface and configuration tools.

PyMuPDF: For rendering PDF documents.
