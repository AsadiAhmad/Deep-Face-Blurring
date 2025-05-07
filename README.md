# Deep-Face-Blurring
Deep-Face-Blurring is a Python-based tool for detecting and anonymizing human faces in images using deep learning models and OpenCV. It leverages YuNet face detection and applies median blurring to protect identities. This project is useful for privacy-preserving preprocessing in datasets, surveillance footage, or public image sharing.

<div display=flex align=center>
  <img src="/Pictures/result.jpg"/>
</div>

## Tech :hammer_and_wrench: Languages and Tools :

<div>
  <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg" title="Python" alt="Python" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/jupyter/jupyter-original.svg" title="Jupyter Notebook" alt="Jupyter Notebook" width="40" height="40"/>&nbsp;
  <img src="https://assets.st-note.com/img/1670632589167-x9aAV8lmnH.png" title="Google Colab" alt="Google Colab" width="40" height="40"/>&nbsp;
  <img src="https://raw.githubusercontent.com/github/explore/8f19e4dbbf13418dc1b1d58bb265953553c15a46/topics/google-drive/google-drive.png" title="Google Drive" alt="Google Drive" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/opencv/opencv-original.svg" title="OpenCV" alt="OpenCV" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/numpy/numpy-original.svg" title="Numpy" alt="Numpy" width="40" height="40"/>&nbsp;
  <img src="https://github.com/devicons/devicon/blob/master/icons/matplotlib/matplotlib-original.svg"  title="MatPlotLib" alt="MatPlotLib" width="40" height="40"/>&nbsp;
</div>

- Python : Popular language for implementing Neural Network
- Jupyter Notebook : Best tool for running python cell by cell
- Google Colab : Best Space for running Jupyter Notebook with hosted server
- Gdown : Download models from the google drive
- OpenCV : Best Library for working with images
- Numpy : Best Library for working with arrays in python
- MatPlotLib : Library for showing the charts in python

## 💻 Run the Notebook on Google Colab

You can easily run this code on google colab by just clicking this badge [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AsadiAhmad/Deep-Face-Blurring/blob/main/Code/Deep_Face_Blurring.ipynb)

## 📝 Tutorial

### Step 1: Import Libraries

we need to import these libraries :

`cv2`, `numpy`, `matplotlib`

```python
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
```

### Step 2: Download Resources

We need to download the YuNet model to the github so please have attention in donwloading new models.

also We need to download the images from my `Github` repository or you can download your own sets.

```sh
!wget https://raw.githubusercontent.com/AsadiAhmad/Image-Matching/main/Pictures/ps5_games.jpg -O ps5_games.jpg
!wget https://raw.githubusercontent.com/AsadiAhmad/Image-Matching/main/Pictures/gost_of_tsushima.jpg -O gost_of_tsushima.jpg
```

### Step 3: Load Image

We need to load images into `python` variables we ues `OpenCV` library to read the images also the format of the images are `nd.array`.

```python
image = cv.imread('many_faces.jpg')
```

<div display=flex align=center>
  <img src="/Pictures/many_faces.jpg" width="400px"/>
</div>

### Step 4: Initialize YuNet face detector

```python
height, width, _ = image.shape

detector = cv.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (width, height),  # Input size (width, height)
    0.8,              # Score threshold
    0.3,              # NMS threshold
    5000              # Top-K candidates
)
# we can handel the input size by set the size and the model just reize that in input
# bigger size more accuracy but decrease the speed

detector.setInputSize((width, height))
result = detector.detect(image)
```

### Step 5: Detect All faces

```python
thickness=5
canvas = image.copy()
faces = []
if result[1] is not None: # check if the face is detected or not
    for idx, face in enumerate(result[1]):
        coords = face[:-1].astype(np.int32)
        x, y, w, h = coords[:4]
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        faces.append([x, y, w, h])
        cv.rectangle(canvas, (x, y), (x+w, y+h), (0, 255, 0), thickness)
```

<div display=flex align=center>
  <img src="/Pictures/Detected.jpg" width="400px"/>
</div>

### Step 6: Blur All faces

```python
blured_faces = []
for face in faces:
    x, y, w, h = face[:4]
    face_shape = image[y:y+h , x:x+w]
    blured_faces.append([cv.medianBlur(face_shape, 55), x, y, w, h])
```

```python
blured_image = image.copy()
for face in blured_faces:
    face_border, x, y, w, h = face[:5]
    blured_image[y:y+h , x:x+w] = face_border
```

<div display=flex align=center>
  <img src="/Pictures/Blurred.jpg" width="400px"/>
</div>

### Step 7: Show Blured Image

```python
plt.figure(figsize=[12,4])
plt.subplot(131),plt.imshow(image[...,::-1]),plt.title('Input');
plt.subplot(132),plt.imshow(canvas[...,::-1]),plt.title('Detected Faces');
plt.subplot(133),plt.imshow(blured_image[...,::-1]),plt.title('Blured Faces');
```

<div display=flex align=center>
  <img src="/Pictures/result.jpg"/>
</div>

## 🪪 License

This project is licensed under the MIT License.
