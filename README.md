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
- Description: The TuSimple dataset consists of video clips with annotated lane lines for highway driving scenarios.
- Annotations: Lane coordinates provided for each video frame.
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
   git clone https://github.com/piyushirish/lane-detection-prototype.git
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
2. Run the script `main.py` to process the data.
3. The detected lane images will be saved in the `output/` folder.
4. Modify parameters in the configuration file (`config.py`) for customization.


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
