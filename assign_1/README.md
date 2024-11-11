# Assignment 1: Text Detection from Images Using OpenCV

## Task

### Task objective
The objective of this assignment is to use Google Colab, Jupyter Notebook, or any other suitable tool to detect text from images using OpenCV. 
This will help you understand the concepts of fundamental Computer Vision algorithms and techniques, evaluate the efficiency and effectiveness of computer vision systems, and apply and implement these tools and techniques. 

### Task description:
* Setup Environment:
  * Use Google Colab, Jupyter Notebook, or any other platform you are comfortable with.
  * Ensure that OpenCV and other necessary libraries are installed.
* Load an Image:
  * Load a sample image containing text. You can use images from various sources like scanned documents, street signs, or any image with clear text.
* Pre-process the Image:
  * Convert the image to grayscale.
  * Apply necessary image processing techniques like thresholding, blurring, or edge detection to enhance text visibility.
* Text Detection:
  * Use OpenCV's text detection methods such as the EAST text detector, or any other OpenCV-compatible methods.
  * Draw bounding boxes around detected text areas.
* Extract and Display Text:
  * Use OCR (Optical Character Recognition) tools like Tesseract to extract text from the detected regions.
  * Display the extracted text and the annotated image with bounding boxes.
* Efficiency and Effectiveness Evaluation:
  * Measure the performance of your text detection system.
  * Discuss the efficiency (speed) and effectiveness (accuracy) of your approach.
  * Suggest potential improvements.

## Installation

### Requirements
```shell
pip install -r ../../requirements.txt
```

### EAST Text Detector
download from [here](https://raw.githubusercontent.com/demuxin/OpenCV_project/refs/heads/master/EAST_Text_Detection/frozen_east_text_detection.pb)