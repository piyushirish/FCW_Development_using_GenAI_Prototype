
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

This repository implements a **Forward Collision Warning (FCW)** system using **YOLOv8** for vehicle detection, lane detection for lane tracking, and a distance estimation algorithm to trigger collision warnings when another vehicle is detected within a defined distance in the same lane.

## Features

- **Vehicle Detection** using YOLOv8: Detects vehicles (cars, trucks, etc.) in the video stream.
- **Lane Detection** using edge detection and the Hough Transform to detect lane boundaries.
- **Collision Warning**: Triggered when another vehicle is detected within a specified distance (default: 30 meters).
- **Real-time Video Processing**: Processes video files and displays results with bounding boxes and collision warnings.
- **Efficient Multi-threading**: Optimized lane detection with multi-threading for faster performance.

## Requirements

### 1. Install Python (3.7+)

Ensure you have **Python 3.7** or higher. You can download it from [here](https://www.python.org/downloads/).

### 2. Install Dependencies

Create a Python virtual environment (optional but recommended), and then install the necessary libraries:

```bash
# Create and activate a virtual environment (optional)
python -m venv fcw_env
# On Windows
fcw_env\Scripts\activate
# On macOS/Linux
source fcw_env/bin/activate

# Install required libraries
pip install opencv-python torch ultralytics numpy

---

## Sample Outputs
![1000285048](https://github.com/user-attachments/assets/e842cb3a-e7d6-4c6c-8e2c-75dc63081f4b)
![1000285047](https://github.com/user-attachments/assets/5b92b67f-28df-4fd2-bb5d-6f69b5e1d0fe)


## Notes

- Ensure you have the YOLOv4 weights and configuration files in the correct location.
- The system is designed for educational purposes and may require further tuning for production use.
- Video processing requires a capable system for real-time execution.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

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

