#!/usr/bin/env python
# coding: utf-8

# In[18]:


import os
import glob
import cv2
import tqdm
import random


# In[39]:


def process_image(image):
    # Example processing: convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert the grayscale image to binary
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)

    # To detect object contours, we want a black background and a white foreground, so we invert the image (i.e. 255 - pixel value)
    inverted_binary = ~binary
    width, height = inverted_binary.shape

    # Find the contours on the inverted binary image, and store them in a list
    # Contours are drawn around white blobs. hierarchy variable contains info on the relationship between the contours
    contours, hierarchy = cv2.findContours(inverted_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    bboxes = []
    threshold = random.randint(80, 120) / 100
    print(threshold)
    # Draw a bounding box around all contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Make sure contour area is large enough
        if h < threshold * height / 100 and w < threshold * width / 100:
            bboxes.append([x, y, w, h])

    for b in bboxes:
        x = b[0]
        y = b[1]
        w = int(b[2])
        h = int(b[3])
        cv2.rectangle(gray, (x,y), (x + w,y + h), (0, 0, 255), -1)
        
    #rgb_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        
    return gray


# In[40]:


def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for img_path in tqdm.tqdm(glob.glob(os.path.join(input_folder, '*.jpg'))):
        # Read the image
        image = cv2.imread(img_path)
        
        # Process the image
        processed_image = process_image(image)
        
        # Get the filename from the path
        filename = os.path.basename(img_path)
        
        # Save the processed image to the output folder with the same filename
        cv2.imwrite(os.path.join(output_folder, filename), processed_image)


# In[44]:


# Example usage
input_folder = '/home/dhruv/Projects/TD-Dataset/Sample-EMBLEM/raw'
output_folder = '/home/dhruv/Projects/TD-Dataset/Sample-EMBLEM/masked'
process_images(input_folder, output_folder)


# In[ ]:





# In[ ]:




