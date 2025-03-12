# PPE-Det
This repository provides the implementation and dataset for the paper **"PPE-Det: Lightweight Object Detection for Edge Devices"**, submitted to the **2nd International Conference on Intelligent Systems, Blockchain, and Communication Technologies (ISBCom) 2025**.  

The project focuses on evaluating lightweight object detection models for detecting **helmets and coveralls (red, blue, yellow)** in industrial environments. The models are optimized for **Raspberry Pi 5** using **PyTorch and NCNN frameworks**.  

---

## Table of Contents  
- [Dataset](#dataset)  
- [Methods](#methods)  
- [Results & Discussion](#results--discussion)  
- [Running the Experiment](#running-the-experiment)  

---
## Dataset

### Overview
The PPE-Det dataset comprises approximately **5,200 images** collected to facilitate Personal Protective Equipment (PPE) detection in industrial environments. The primary objective of this PPE-Det dataset is to enhance the accuracy of industrial models by providing diverse object detection samples in real-world scenarios. The images were extracted from a recorded video in an **Egyptian industrial setting** and split into frames at a rate of **2 frames per second (fps).**

### Preprocessing and Annotation
The PPE-Det dataset utilizes **bounding box annotations** to enable object detection. The annotations follow the **YOLO format**, where each labeled instance is represented by a class ID and four normalized values (**center x, center y, width, height**) defining the bounding box.

All images were initially **auto-labeled using an object detection model**. The auto-generated labels were then manually reviewed and corrected for accuracy. All images were resized so that the largest dimension is **640 pixels**, while preserving the aspect ratio.

### Class Distribution
The PPE-Det dataset contains object instances categorized across three distance ranges: **Far, Mid, and Near**, representing different levels of detail and resolution.
The three distance ranges are quantified as follows: 
- **Near Range:** Approximately 2.5-3m
- **Mid Range:** 4-5m
- **Far Range:** 8-10m

These distances refer to the **camera height from the ground**.

| Class ID | Description      | Far-Range | Mid-Range | Near-Range | Total Instances |
|----------|------------------|-----------|-----------|------------|-----------------|
| 0        | Helmet            | 1,884      | 3,593      | 1,878       | 7,355           |
| 1        | Blue Coverall     | 1,412      | 553        | 385         | 2,350           |
| 2        | Red Coverall      | 338        | 1,604      | 855         | 2,797           |
| 3        | Yellow Coverall   | 341        | 1,669      | 905         | 2,915           |
| **Total**| -                | **3,975**  | **7,419**  | **4,023**   | **15,417**      |

### Number of Images per Distance Range
| Distance Range | Number of Images |
|----------------|------------------|
| Far-Range      | 1,695             |
| Mid-Range      | 1,676             |
| Near-Range     | 1,840             |
| **Total**      | **5,211**         |

### Data Splits and Organization
The data is split into three subsets: **70\% for training, 15\% for validation, and 15\% for testing**. 

The dataset is hierarchically organized. At the top level, the dataset is divided into three main splits: **train, val, and test**. Each of these splits is categorized into three subdirectories—**far, mid, and near**—based on object distance. Each of these subdirectories contains two subfolders: **images/** (JPG format) and **labels/** (TXT in YOLO format).

```
dataset/
│── train/
│   ├── far/
│   │   ├── images/ (JPG)
│   │   ├── labels/ (TXT in YOLO format)
│   ├── mid/
│   ├── near/
```

### YOLO Annotation Format
Each YOLO annotation file follows a structured format where each line represents an object in the image:
```
<class_id> <center_x> <center_y> <width> <height>
```

**Class ID Mapping:**
- **0:** Helmet
- **1:** Blue Coverall
- **2:** Red Coverall
- **3:** Yellow Coverall



### 🔗 Dataset Access
The PPE-Det dataset can be accessed via this [Google Drive Link](https://drive.google.com/drive/folders/1RPhOT1OLkAopfAbRdLiQo_hbQeRj2HT6?usp=sharing).

### 📸 Sample Images
Below are sample images representing the three distance ranges:
- **Far Range**: ![Far Range Image](Images\far.jpg)
- **Mid Range**: ![Mid Range Image](Images\mid.jpg)
- **Near Range**: ![Near Range Image](Images\near.jpg)



<!-- ### Models Trained on the PPE-Det Dataset
The following models were trained and evaluated on the PPE-Det dataset for object detection:

| Model       | Architecture | Description | 
|-------------|--------------|-------------|
| YOLOv5n     | Ultra-lightweight | Designed for fast inference on edge devices. |
| YOLOv8n     | Next-gen YOLO | Improved accuracy and efficiency over previous YOLO versions. |
| YOLOv9t     | Transformer-enhanced YOLO | Incorporates transformer layers for better feature representation. |
| YOLOv10n    | Enhanced lightweight | Optimized for better performance on low-power devices. |
| YOLOv11n    | Further optimized | Improved speed-accuracy tradeoff compared to YOLOv10n. |
| YOLOX-Nano  | Nano variant | Efficient object detection designed specifically for mobile and edge devices. |
| NanoDet-M   | Lightweight anchor-free model | Specializes in fast and efficient object detection. |

These models were evaluated based on metrics such as **Mean Average Precision (mAP), Inference Time, and Model Size**. Performance evaluations were conducted on a **Raspberry Pi 5** to simulate real-world edge deployment scenarios. -->
