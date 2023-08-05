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

import pickle


dataset_path = '/home/ubuntu/data_augmentation/frontcam_dataset_version_2_new/augmented_training_dataset_version_2'

augmented_dataset_path = '/home/ubuntu/data_augmentation/frontcam_dataset_version_4/using_stored_kernels/front_cam_dataset_version_4'

sample_image_folder = '/home/ubuntu/data_augmentation/frontcam_dataset_version_4/using_stored_kernels/sample_images'

high_blur_images = '/home/ubuntu/data_augmentation/frontcam_dataset_version_4/using_stored_kernels/high_blur_images'

if not os.path.isdir(f'./plots'):
    os.makedirs(f'./plots') 

if not os.path.isdir(augmented_dataset_path):
    os.makedirs(augmented_dataset_path) 

if not os.path.isdir(sample_image_folder):
    os.makedirs(sample_image_folder) 



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

lux_dist = [0]*11

def plot_lists(list_x, list_y):
    if len(list_x) != len(list_y):
        raise ValueError("Both lists must have the same length")

    plt.figure(figsize=(15, 10))
    plt.scatter(list_x, list_y)
    plt.xlabel('Blur Kernel Strength')
    plt.ylabel('SSIM of transformed and original image')
    title = 'SSIM Vs Blur-Kernel-Strength'
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"./plots/{title.replace(' ', '_')}.png")
    plt.close()




with open('blur_kernel_data_tmp_31jul.pkl', 'rb') as f:
    blur_kernels = pickle.load(f)

print('Blur kernels loaded...')


def get_ssim(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate the SSIM
    ssim_score = ssim(image1_gray, image2_gray)

    return ssim_score


ssims = []
kernel_strengths =[]
def normalize_2d_array(array):
    # Calculate the sum of all elements in the array
    total_sum = np.sum(array)

    # Divide each element by the total sum to normalize
    normalized_array = array / total_sum

    return normalized_array

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

def apply_transform_and_save(image_path):
    fname = image_path.split('/')[-1]
    ev = get_ev_from_fname(fname)
    lux_index = get_lux_index_from_fname(fname)

    lux_dist[lux_index]+=1

    if len(blur_kernels[lux_index][ev]) > 0:
        kernel_index = random.randint(0, len(blur_kernels[lux_index][ev])-1)
        blur_kernel = blur_kernels[lux_index][ev][kernel_index]
    else:
        print(f'Using default kernel for lux_index : {lux_index}, ev_value : {ev}')
        blur_kernel = blur_kernels[0][0][0]

    image = cv2.imread(image_path)
    blur_kernel = np.array(blur_kernel)
    blur_kernel = blur_kernel.astype(np.float32)
    #blur_kernel = normalize_2d_array(blur_kernel)

    b, g, r = cv2.split(image)

    conv_b = cv2.filter2D(b, -1, blur_kernel)
    conv_g = cv2.filter2D(g, -1, blur_kernel)
    conv_r = cv2.filter2D(r, -1, blur_kernel)

    convolved_image = cv2.merge([conv_b, conv_g, conv_r])

    ssim_score = get_ssim(image, convolved_image)

    ssims.append(ssim_score)

    strength = calculate_kernel_spread(blur_kernel)
    kernel_strengths.append(strength)
    
    concatenated_horizontal = cv2.hconcat([image, convolved_image])
    cv2.putText(concatenated_horizontal, f'SSIM score : {ssim_score:.2f}, Blur kernel strength : {strength:.2f}', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(concatenated_horizontal, f'SSIM score : {ssim_score:.2f}, Blur kernel strength : {strength:.2f}', (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6, cv2.LINE_AA)



    cv2.imwrite(f'{sample_image_folder}/{fname}', concatenated_horizontal)

    cv2.imwrite(f'{augmented_dataset_path}/{fname}', convolved_image)

    if strength > 2:
        cv2.imwrite(f'{high_blur_images}/{fname}', concatenated_horizontal)


    

    #print(f'blur_kernel.shape : {blur_kernel.shape}')

    print(f'SSIM score : {ssim_score}, Blur kernel strength : {strength}')

k = 0
for filename in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, filename)
    if os.path.isfile(file_path):
        if 'jpg' in filename and 'ev' in filename:
            # Perform actions on the file
            print(f'Transforming : {file_path}')

            apply_transform_and_save(file_path)
        else:
            print(f'Copying : {file_path}')

            sh.copy(file_path, file_path.replace(dataset_path, augmented_dataset_path))

    
    #if k>9: break
    print(f'k= {k}')
    k+=1


plot_lists(ssims, kernel_strengths)

print(f'Number of images transformed : {len(ssims)}')
print(f'Max ssim : {max(ssims)}\nMin SSIM: {min(ssims)}')

std_dev = statistics.stdev(ssims)

print(f'Average SSIM:{sum(ssims)/len(ssims)}\nStandard Deviation: {std_dev}')


# Set labels and title
#plt.figure(figsize=(9, 6))
plt.figure(figsize=(15, 10))
plt.hist(ssims, bins=6, rwidth=0.8)

plt.xlabel('SSIM')
plt.ylabel('Frequency')
plt.title('SSIM distribution for transformed image, for the Frontcam Dataset')

plot_path = './plots'

if not os.path.isdir(f'{plot_path}'):
    os.makedirs(f'{plot_path}') 
    
plt.savefig(f'{plot_path}/SSIM_comparison_frontcam_dataset.png')
plt.close()

plt.figure(figsize=(15, 10))
plt.hist(kernel_strengths, bins=6, rwidth=0.8)

plt.xlabel('Blur Kernel Strength')
plt.ylabel('Frequency')

title = 'Blur Kernel Strength Distribution'
plt.title(title)

plot_path = './plots'

if not os.path.isdir(f'{plot_path}'):
    os.makedirs(f'{plot_path}') 
    
plt.savefig(f"./plots/{title.replace(' ', '_')}.png")

if True:    
    plt.close()
    numbers = lux_dist
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