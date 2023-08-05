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

import copy
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte

from models.TwoHeadsNetwork import TwoHeadsNetwork

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from utils.visualization import save_kernels_grid
import pickle

from PIL import Image, ImageEnhance

handheld_image_set = '/mnt/sdd1/sarthakagarwal/misc/dataset_version_4_26_july/all_handheld_images/all_handheld_images_frontcam'

high_blur_kernels_folder = './high_blur_kernels'

if not os.path.isdir(high_blur_kernels_folder):
    os.makedirs(high_blur_kernels_folder)

#kernel_dir = '/mnt/sdd1/sarthakagarwal/misc/dataset_version_4_26_july/blur_kernels'

if not os.path.isdir(f'./plots'):
    os.makedirs(f'./plots') 


blur_kernels = {}

evs = [-24, -20, -3, -1, 0, 1, 2, 3, 4]


kernel_strength_ev_distribution = {}
kernel_strength_lux_distribution = [[] for _ in range(11)]

tmp_dict = {}
for ev in evs:
    tmp_dict[ev] = []
    kernel_strength_ev_distribution[ev] = []

blur_kernels = [copy.deepcopy(tmp_dict) for _ in range(11)]
i = 0
def get_ev_from_fname(fname):
    temp = fname.split('_')[-2]
    temp = temp.split('.')[0]
    ev = int(temp)
    return ev

def get_lux_index_from_fname(fname):
    tmp = fname.split('_')[1]
    lux = float(tmp)//1
    lux_index = lux//2
    lux_index = int(lux_index)
    return lux_index

K = 25  # number of elements in the base
model_file = '/mnt/sdd1/sarthakagarwal/misc/front_cam_data_augmentation/NonUniformBlurKernelEstimation/TwoHeads.pkl'
two_heads = TwoHeadsNetwork(K)

print('Loading weights model')
two_heads.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
two_heads.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

fname_to_kernel_strength_map = []

def get_enhanced_image(file_path):
    im = Image.open(file_path)
    ev = get_ev_from_fname(file_path.split('/')[-1])
    if ev == -24:
        im1 = ImageEnhance.Brightness(im)
        enhanced_image = im1.enhance(8.0)
        return enhanced_image
    elif ev == -20:
        im1 = ImageEnhance.Brightness(im)
        enhanced_image = im1.enhance(5.0)
        return enhanced_image
    
    return im


def get_blur_kernel(blurry_image_filename):
    blurry_image = get_enhanced_image(blurry_image_filename)
    blurry_tensor = transform(blurry_image)
    blurry_tensor_to_compute_kernels = blurry_tensor**2.2 - 0.5
    kernels_estimated, masks_estimated = two_heads(blurry_tensor_to_compute_kernels[None, :, :, :])
    masks = masks_estimated.detach().numpy()
    all_masks = masks[0, :, :, :]
    mask_sums = np.sum(all_masks, axis=(1, 2))
    biggest_mask_index = np.argmax(mask_sums)

    blur_kernel = kernels_estimated[0, biggest_mask_index, :, :]

    return blur_kernel.detach().numpy()


def calculate_kernel_spread(kernel):
    # Step 1: Normalize the kernel
    normalized_kernel = kernel / np.sum(kernel)

    # Step 2: Calculate the centroid
    centroid_x = np.sum(np.arange(kernel.shape[0]) * normalized_kernel)
    centroid_y = np.sum(np.arange(kernel.shape[1]) * normalized_kernel)

    # Step 3: Calculate second-order moments
    mu_xx = np.sum((np.arange(kernel.shape[0]) - centroid_x) ** 2 * normalized_kernel)
    mu_yy = np.sum((np.arange(kernel.shape[1]) - centroid_y) ** 2 * normalized_kernel)
    mu_xy = np.sum((np.arange(kernel.shape[0]) - centroid_x)[:, np.newaxis]
                   * (np.arange(kernel.shape[1]) - centroid_y) * normalized_kernel)

    # Step 4: Construct the covariance matrix
    covariance_matrix = np.array([[mu_xx, mu_xy], [mu_xy, mu_yy]])

    # Step 5: Find eigenvalues
    eigenvalues, _ = np.linalg.eigh(covariance_matrix)

    # Step 6: Calculate the overall spread as the mean of the eigenvalues
    overall_spread = np.mean(eigenvalues)

    return overall_spread

def saveHighBlurKernel(image_path, blur_kernel, blur_strength):
    folder_name = image_path.split('/')[-2]
    fname = image_path.split('/')[-1]

    new_path = f'{high_blur_kernels_folder}/{folder_name}/{fname[:-4]}'

    if not os.path.isdir(new_path):
        os.makedirs(new_path)

    sh.copy(image_path, f'{new_path}/{fname}')

    ev = get_ev_from_fname(fname)

    plt.close()
    plt.imshow(kernel, cmap='viridis')  # You can choose a different colormap if you prefer
    plt.colorbar()  # Add a colorbar to show the mapping of values to colors
    title = f'Blurrying kernel, EV {ev}, Strength {blur_strength:.1f}'
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.savefig(f"{new_path}/{title.replace(' ', '_')}.png")
    plt.close()


def savePlots():
    print('Saving Plots...')
    ev_wise_average = []

    for ev in evs:
        
        avg = 0
        if len(kernel_strength_ev_distribution[ev]) > 0:
            avg = sum(kernel_strength_ev_distribution[ev])/len(kernel_strength_ev_distribution[ev])
        
        ev_wise_average.append(avg)

    #print(f'len(ev_wise_average) : {len(ev_wise_average)}')

    lux_wise_average = []
    nos_in_lux_range = []

    for i in range(len(kernel_strength_lux_distribution)):
        avg = 0
        if len(kernel_strength_lux_distribution[i]) > 0:
            avg = sum(kernel_strength_lux_distribution[i])/len(kernel_strength_lux_distribution[i])
        lux_wise_average.append(avg)
        nos_in_lux_range.append(len(kernel_strength_lux_distribution[i]))

    #---------------------------------------------------

    numbers = ev_wise_average
    #print(len(numbers))
    labels = [f'EV_{i}' for i in evs]

    # Setting up the figure and axis
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Plotting the bar graph with proper spacing
    bar_width = 0.5
    bar_positions = range(len(numbers))

    #print(f'Bar positions : {bar_positions}')
    plt.bar(bar_positions, numbers, width=bar_width, align='center')

    # Adding labels to the bars
    #for i, num in enumerate(numbers):
    #    plt.text(bar_positions[i], num + 1, str(num), ha='center', va='bottom', fontsize=12)

    # Adding labels to the x-axis
    plt.xticks(bar_positions, labels)

    # Adding a dotted grid
    ax.yaxis.grid(True, linestyle='dotted')

    # Adding labels and title
    plt.xlabel('EV values')
    plt.ylabel('Kernel Strength')
    title = 'Blur kernel strength distribution for different EVs'
    plt.title(title)

    plt.savefig(f"./plots/{title.replace(' ', '_')}.png")
    plt.close()

    #---------------------------------------------------

    numbers = lux_wise_average
    labels = [f'{2*i} to {2*i + 2}' for i in range(11)]

    # Setting up the figure and axis
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Plotting the bar graph with proper spacing
    bar_width = 0.5
    bar_positions = range(len(numbers))
    plt.bar(bar_positions, numbers, width=bar_width, align='center')

    # Adding labels to the bars
    #for i, num in enumerate(numbers):
    #    plt.text(bar_positions[i], num + 1, str(num), ha='center', va='bottom', fontsize=12)

    # Adding labels to the x-axis
    plt.xticks(bar_positions, labels)

    # Adding a dotted grid
    ax.yaxis.grid(True, linestyle='dotted')

    # Adding labels and title
    plt.xlabel('Lux ranges')
    plt.ylabel('Kernel Strength')
    title = 'Blur kernel strength distribution for different lux ranges'
    plt.title(title)

    plt.savefig(f"./plots/{title.replace(' ', '_')}.png")

    #---------------------------------------------------
    plt.close()
    numbers = nos_in_lux_range
    labels = [f'{2*i} to {2*i + 2}' for i in range(11)]

    # Setting up the figure and axis
    plt.figure(figsize=(15, 10))
    ax = plt.gca()

    # Plotting the bar graph with proper spacing
    bar_width = 0.5
    bar_positions = range(len(numbers))
    plt.bar(bar_positions, numbers, width=bar_width, align='center')

    # Adding labels to the bars
    #for i, num in enumerate(numbers):
    #    plt.text(bar_positions[i], num + 1, str(num), ha='center', va='bottom', fontsize=12)

    # Adding labels to the x-axis
    plt.xticks(bar_positions, labels)

    # Adding a dotted grid
    ax.yaxis.grid(True, linestyle='dotted')

    # Adding labels and title
    plt.xlabel('Lux ranges')
    plt.ylabel('Number of Images')
    title = 'Number of images in different lux ranges'
    plt.title(title)

    plt.savefig(f"./plots/{title.replace(' ', '_')}.png")

    print('Saving Kernels...')
    with open('blur_kernel_data_tmp.pkl', 'wb') as f:
        pickle.dump(blur_kernels, f)

    print('Saving File to strength mapping...')
    with open('image_file_blur_strength.pkl', 'wb') as f:
        pickle.dump(fname_to_kernel_strength_map, f)





for root, dirs, files in os.walk(handheld_image_set):
        
    #if i>0: break
    print(f'i = {i}')

    if i>=1:
        temp = {} 
        for fname in files:
            if 'ev' in fname: 
                print(f'Reading File {root}/{fname}')
                ev = get_ev_from_fname(fname)
                lux_index = get_lux_index_from_fname(fname)
                file_path = f'{root}/{fname}'

                kernel = get_blur_kernel(file_path)

                blur_kernels[lux_index][ev].append(kernel)

                strength = calculate_kernel_spread(kernel)

                if strength>2.2:
                    saveHighBlurKernel(f'{root}/{fname}', kernel, strength)

                kernel_strength_ev_distribution[ev].append(strength)
                kernel_strength_lux_distribution[lux_index].append(strength)

                print(f'Lux_index : {lux_index}, EV: {ev}, Kernel strength : {strength}')
                fname_to_kernel_strength_map.append((f'{root}/{fname}', strength))
    i+=1
    if i>3: savePlots()    

print(f'i = {i}')

with open('blur_kernel_data_final.pkl', 'wb') as f:
    pickle.dump(blur_kernels, f)

'''
with open('blur_kernel_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
'''

