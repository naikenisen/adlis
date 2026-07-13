import os
import random
from PIL import Image

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dataset directory structure:
# dataset_split/
#   train/
#     images/
#     annotations/
#   valid/
#     images/
#     annotations/
#   test/
#     images/
#     annotations/

image_path = "object_detection_dataset/dataset_split/train/images/Image_20240528_092815709.jpg"

xml_path = "object_detection_dataset/dataset_split/train/annotations/Image_20240528_092815709.jpg.xml"

# Parse Pascal VOC XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Load image
img = Image.open(image_path)
fig, ax = plt.subplots()
ax.imshow(img)

# Draw bounding boxes (all same color, no annotation)
box_color = 'lime'
for obj in root.findall('object'):
    bbox = obj.find('bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=box_color, facecolor='none')
    ax.add_patch(rect)

plt.axis('off')
plt.show()