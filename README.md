
# Forward Collision Warning (FCW) System Code Generator

This repository automates the generation of Python code for a **Forward Collision Warning (FCW)** system using OpenAI's GPT API. The system is designed based on predefined requirements such as object detection, collision warning, compliance standards, and more.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Generated Output](#generated-output)
- [Compliance](#compliance)
- [Notes](#notes)
- [License](#license)

---

## Overview

The program reads functional requirements for an FCW system from a JSON file and uses the OpenAI GPT model (gpt-3.5-turbo) to generate Python code. The generated code is stored in a separate Python file for further use.

---

## Features

1. **Automated Code Generation:** Leverages OpenAI GPT-3.5-turbo to generate Python code based on provided FCW requirements.
2. **Object Detection:** Implements object detection functionality with specified accuracy and detection range.
3. **Collision Warning:** Triggers collision alerts based on a configurable distance and response time.
4. **Annotated Video Frames:** The system provides visual feedback with annotations to highlight detected objects and potential collisions.
5. **Easy Compliance Configuration:** Ensures the generated system aligns with industry standards such as MISRA, ASPICE, and ISO 26262.

---

## Requirements

- **Python 3.8+**
- **OpenAI Python Library**
- **JSON File Handling**
- **OpenAI API Key**

### Libraries
Install the required dependencies:
```bash
pip install openai
```

---

## Installation
1. Clone the repository:
   ```bash
   https://github.com/piyushirish/FCW_Development_using_GenAI_Prototype.git
   ```
2. Ensure you have the **OpenAI API key**.
3. Install the dependencies:
   ```bash
   pip install openai
   ```
---

## Usage

1. Update the **requirements** in `fcw_requirements.json`:
   ```json
   {
       "object_detection": {
           "method": "AI/ML",
           "accuracy": "95%",
           "detection_range": "50 meters"
       },
       "collision_warning": {
           "trigger_distance": "5 meters",
           "response_time": "100 ms"
       },
       "compliance": {
           "standards": ["MISRA", "ASPICE", "ISO 26262"]
       },
       "test_coverage": "100%"
   }
   ```
2. Run the code to generate Python code based on the requirements:
   ```bash
   python generate_fcw_code.py
   ```
3. The generated Python code will be saved to `generated_code_FCW.py`.

4. Review or execute the generated file:
   ```bash
   python generated_code_FCW.py
   ```

---

## Configuration

### OpenAI API Key
Ensure you have a valid OpenAI API key. Replace the placeholder in the script with your key:
```python
openai.api_key = "your-openai-api-key"
```

### JSON File for Requirements
Modify the `fcw_requirements.json` file to customize the input requirements.

---

## Generated Output
The generated file, `generated_code_FCW.py`, includes:
- Object Detection methods
- Collision Warning logic with trigger thresholds
- Video frame annotation

You can integrate or customize this output further based on your project requirements.

---

## Compliance
The generated FCW system aims to adhere to:
- **MISRA**
- **ASPICE**
- **ISO 26262**

> Note: Compliance validation is beyond the scope of automated code generation. Manual verification is recommended.

---

## Notes
1. Ensure a stable internet connection for API calls.
2. The code generation accuracy depends on the clarity and completeness of the input requirements.
3. This repository is for educational and prototyping purposes.

---

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.


# Forward Collision Warning (FCW) System

This project implements a **Forward Collision Warning System** using the **YOLOv8** object detection model and OpenCV. The system detects vehicles in a video feed, estimates their distance based on bounding box size, and triggers a collision warning if a detected vehicle is within a specified distance threshold.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Usage](#usage)
6. [How It Works](#how-it-works)
7. [Directory Structure](#directory-structure)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview

The Forward Collision Warning System processes video footage from a front-facing camera to detect vehicles in the same lane. It estimates the distance to each detected vehicle and provides a visual collision warning if any vehicle is dangerously close. The system leverages the **Ultralytics YOLOv8** model for real-time vehicle detection and OpenCV for video processing and lane detection.

---

## Features

- **Vehicle Detection**: Detects cars using the YOLOv8 model.
- **Distance Estimation**: Approximates distance based on bounding box size.
- **Lane Detection**: Ensures vehicles are detected in the same lane.
- **Collision Warning**: Displays a warning when a vehicle is too close.
- **Multi-threading**: Uses threading for optimized lane detection and video processing.

---

## Dependencies

Ensure the following libraries and tools are installed:

- **Python 3.8+**
- **OpenCV**
- **NumPy**
- **Ultralytics** (for YOLOv8)

Install dependencies using the following command:

```bash
pip install opencv-python-headless numpy ultralytics
```

---

## Installation

1. Clone the repository:
   ```bash
   https://github.com/piyushirish/FCW_Development_using_GenAI_Prototype.git
   ```
2. Download the YOLOv8 model weights (**yolov8n.pt**):
   - Visit the Ultralytics YOLOv8 page to obtain the weights.
   - Place the weights file (`yolov8n.pt`) in the project directory.
3. Verify your Python environment and install required dependencies.

---

## Usage

1. Place your input video in the project directory.
2. Update the `video_path` variable in the script with your video file path.
   ```python
   video_path = "/path/to/your/video.mp4"
   ```
3. Run the script:
   ```bash
   python forward_collision_warning.py
   ```
4. Press `` during execution to exit the program.

---

## How It Works

The project workflow consists of the following steps:

1. **Lane Detection**: Using edge detection and Hough Line Transform, the system identifies lane boundaries.
2. **Vehicle Detection**: The YOLOv8 model detects vehicles in each video frame.
3. **Distance Estimation**: The system calculates distance based on bounding box size and detection confidence:
   - Larger bounding box → Closer distance.
   - Smaller bounding box → Farther distance.
4. **Collision Detection**: The system checks if:
   - The vehicle is in the same lane.
   - The estimated distance is below the warning threshold.
5. **Warning Display**: If a collision is predicted, a **red collision warning** appears on the video feed.

---

## Directory Structure

```
forward-collision-warning/
|-- forward_collision_warning.py  # Main script
|-- yolov8n.pt                    # YOLOv8 model weights
|-- test_video.mp4                # Input video (replace with your own)
|-- README.md                     # Project documentation
```

---
## Sample results
![WhatsApp Image 2024-12-17 at 20 24 17](https://github.com/user-attachments/assets/4d1778e1-d526-40af-8d1a-6e75aa477819)
![WhatsApp Image 2024-12-17 at 20 24 16](https://github.com/user-attachments/assets/ae9bb004-be95-47d3-936f-9fca78e3db26)


## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this project with attribution.

---

# Lane Detection using TuSimple Dataset

This project implements a **Lane Detection Model** using the **TuSimple Dataset**. The model is designed to accurately identify lane markings in driving scenarios, making it applicable to autonomous driving systems and advanced driver assistance systems (ADAS).

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Details](#dataset-details)
3. [Technologies Used](#technologies-used)
4. [Installation Instructions](#installation-instructions)
5. [How to Use the Project](#how-to-use-the-project)
6. [Results and Performance](#results-and-performance)
7. [Future Improvements](#future-improvements)
8. [Acknowledgements](#acknowledgements)
9. [License](#license)

---

## Project Overview
The project uses computer vision and deep learning techniques to detect lane lines from input images or video frames. It uses the **TuSimple dataset**, a widely used benchmark dataset for lane detection tasks.

The workflow includes:
- Preprocessing the TuSimple dataset for model training.
- Developing a neural network-based lane detection model.
- Evaluating the model's performance.

---

## Dataset Details
- Name: TuSimple Lane Detection Dataset
- Description: The TuSimple dataset consists of images in two folders,frames and lane-masks.
- Annotations: Lane coordinates provided for each image.
- Usage: Primarily used for supervised learning in lane detection.

You can access the TuSimple dataset from kaggle

---

## Technologies Used
- Python
- OpenCV: Image processing and visualization.
- TensorFlow/Keras: Model development and training.
- NumPy: Data handling and numerical computation.
- Matplotlib: Visualization of results.

---

## Installation Instructions
Follow the steps below to set up the project:

1. **Clone the Repository**:
   ```bash
   https://github.com/piyushirish/FCW_Development_using_GenAI_Prototype.git
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download TuSimple Dataset**:
   - Download the dataset from kaggle.
   - Place the dataset in the `data/` folder.

5. **Run the Project**:
   ```bash
   run the file lane_detection_model.ipynb
   ```

---

## How to Use the Project
1. image files in the `input/` directory.
2. Run the script `lane_detection_model.ipynb` to process the data.
3. The detected lane images will be saved in the `output/` folder.

## Results and Performance
### Sample Results
Here are sample images showcasing lane detection:
<img width="1440" alt="Screenshot 2024-12-14 at 11 59 18 PM" src="https://github.com/user-attachments/assets/3d048bab-5cda-43da-938f-8fb324d6ffa1" />
<img width="1440" alt="Screenshot 2024-12-14 at 11 58 35 PM" src="https://github.com/user-attachments/assets/30ab04f7-75a6-4864-92b1-4b44e5e38402" />
<img width="1440" alt="Screenshot 2024-12-14 at 11 58 02 PM" src="https://github.com/user-attachments/assets/0e8fecf8-7cd4-4aa9-a29f-ff88ab511bd7" />



## Future Improvements
- Implement real-time lane detection using video streams.
- Optimize the model for faster inference.
- Integrate advanced deep learning models like **U-Net** or **YOLO** for improved performance.

---

## Acknowledgements
- TuSimple for providing the dataset.
- Open-source libraries: OpenCV, TensorFlow, and Keras.
- Community contributors for inspiration and resources.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

