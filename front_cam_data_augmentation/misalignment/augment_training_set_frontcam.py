# For frontCam

import cv2
import numpy as np
import random
import math
import os
import shutil as sh

import random
from skimage.metrics import structural_similarity as ssim

import statistics
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
handheld_image_set = '/home/ubuntu/data_augmentation/original_datasets/all_handheld_images_frontcam'
dataset_path = '/home/ubuntu/data_augmentation/original_datasets/training_dataset_frontcam'
#augmented_dataset_path = '/home/ubuntu/data_augmentation/augmented_datasets/misaligned_frontcam_training_dataset_12_july'
augmented_dataset_path = '/home/ubuntu/data_augmentation/frontcam_dataset_version_2/augmented_training_dataset_version_2'
if not os.path.isdir(f'{augmented_dataset_path}'):
    os.makedirs(f'{augmented_dataset_path}') 

i=0

handheld_files = []

for root, dirs, files in os.walk(handheld_image_set):
        
    #if i>9: break
    if i>=1:
        temp = [] 
        for fname in files:
            if 'ev' in fname: 
                temp.append(f'{root}/{fname}')
        if len(temp)>0:
            handheld_files.append(temp)    
    
    i+=1    

print(f'i = {i}')

k= 1


cropped_width = 3264 - 196
cropped_height = 2448 - 132

def center_crop(image, desired_width=cropped_width, desired_height=cropped_height):
    height, width = image.shape[:2]

    start_x = max(0, width // 2 - desired_width // 2)
    end_x = start_x + desired_width
    start_y = max(0, height // 2 - desired_height // 2)
    end_y = start_y + desired_height

    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image

def get_transformation(image_nos, evs):
    
    img1 = cv2.imread(handheld_files[image_nos][evs[0]])
    img2 = cv2.imread(handheld_files[image_nos][evs[1]])

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    align_method = cv2.MOTION_EUCLIDEAN

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-7)

    _, transformation = cv2.findTransformECC(gray1, gray2, np.eye(3, 3, dtype=np.float32)[:2], align_method, criteria)

    return transformation

def apply_transform(image):

    image_nos = random.randint(0, len(handheld_files)-1)

    random_int1 = random.randint(0, len(handheld_files[0])-1)
    random_int2 = random.randint(0, len(handheld_files[0])-1)
    while random_int2 == random_int1:
        random_int2 = random.randint(0, len(handheld_files[0])-1)

    evs = [random_int1, random_int2]
    print(f'evs : {evs}')

    transformation = get_transformation(image_nos, evs)

    new_image = cv2.warpAffine(image, transformation, (image.shape[1], image.shape[0]))

    new_image = center_crop(new_image)

    return new_image

def get_ssim(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the SSIM
    ssim_score = ssim(image1_gray, image2_gray)

    return ssim_score


ssims = []
# Iterate over all files in the directory
for filename in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, filename)
    if os.path.isfile(file_path) and 'jpg' in filename:
        # Perform actions on the file
        print(file_path)
        image = cv2.imread(file_path)
        if 'ev_-24' in filename: 
            #print('I would augment this file! EV_-24')
            new_image = apply_transform(image)
            score = get_ssim(center_crop(image), new_image)
            ssims.append(score)
            print(f'SSIM value : {score}')

        else:
            new_image = center_crop(image)

        print(f'Original Image Shape : {image.shape}')
        print(f'New Image Shape : {new_image.shape}')

        cv2.imwrite(f'{augmented_dataset_path}/{filename}', new_image)


    
    #if k>40: break
    print(f'k= {k}')
    k+=1


print(f'Number of images transformed : {len(ssims)}')
print(f'Max ssim : {max(ssims)}\nMin SSIM: {min(ssims)}')

std_dev = statistics.stdev(ssims)

print(f'Average SSIM:{sum(ssims)/len(ssims)}\nStandard Deviation: {std_dev}')

plt.hist(ssims, bins=6, rwidth=0.8)

# Set labels and title
#plt.figure(figsize=(9, 6))
plt.xlabel('SSIM')
plt.ylabel('Frequency')
plt.title('SSIM distribution for transformed image, for the Frontcam Dataset')

plot_path = '/home/ubuntu/data_augmentation/frontcam_dataset_version_2/plots'

plt.savefig(f'{plot_path}/SSIM_comparison_frontcam_dataset.png')