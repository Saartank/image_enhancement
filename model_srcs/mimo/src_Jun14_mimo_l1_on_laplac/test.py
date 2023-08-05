# test.py
import os, scipy.io, time
import tensorflow as tf
import numpy as np
import cv2
from net_dataloader_test import *
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from model import MyModel


method_name = "Mimo_June_14_L-1_loss_on_laplacian_weighted_sum"
#result_dir = '/mnt/sdd1/sarthakagarwal/Feb_09/copy1_DarkBurst/dark-burst-photography-v2/src_May25_mimo/junk'
#result_dir = './junk'
result_dir = '/mnt/sdd1/sarthakagarwal/Feb_09/copy1_DarkBurst/dark-burst-photography-v2/old_test_results'
#result_dir = '/mnt/sdd1/sarthakagarwal/Feb_09/copy1_DarkBurst/dark-burst-photography-v2/new_test_tripod_results'
dataset_path = '/mnt/sdd1/sarthakagarwal/Feb_09/copy1_DarkBurst/dark-burst-photography-v2/rear_dataset/test_rear'
#dataset_path='/mnt/sdd1/sarthakagarwal/Feb_09/copy1_DarkBurst/dark-burst-photography-v2/new_test_tripod'
saved_model_dir='../saved_models/%s/'% method_name


my_model = MyModel()
my_model.compile()
my_model.load_weights(f'{saved_model_dir}/my_model.h5')

def space_to_depth(x, block_size=2):
	x = np.asarray(x)
	print("x shape", x.shape)
	batch = x.shape[0]
	height = x.shape[1]
	width = x.shape[2]

	depth = x.shape[3]
	reduced_height = height // block_size
	reduced_width = width // block_size
	y = x.reshape(batch, reduced_height, block_size,
				  reduced_width, block_size, depth)
	z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
	return z
def depth_to_space(x, block_size=2):
	x = np.asarray(x)
	height = x.shape[0]
	width = x.shape[1]

	depth = x.shape[2]

	original_height = height * block_size
	original_width = width * block_size
	original_depth=depth//(block_size**2)
	x = x.reshape(height, width, block_size,block_size)
	x= x.transpose([0,2,1,3]).reshape(original_height,original_width)
	x=np.expand_dims(x,axis=2)

	return x

def YUV444toYUV420(yuv444):

    y = yuv444[:,:,0]
    y = tf.expand_dims(y, axis=2)
    uv = yuv444[:,:,1:]
    uv=np.float32(uv)
    #print((uv[0::2,0::2,:] + uv[0::2,1::2,:] + uv[1::2,0::2,:] + uv[1::2,1::2,:]))
    uv = (uv[0::2,0::2,:] + uv[0::2,1::2,:] + uv[1::2,0::2,:] + uv[1::2,1::2,:]) / 4.0
    return y, np.uint8(uv)

def YUV420toYUV444(y, uv):  # format NHWC

    H2 = np.shape(uv)[0]
    W2 = np.shape(uv)[1]
    tmpuv = np.concatenate([uv, uv], axis = 2)
    tmpuv = np.reshape(tmpuv, [H2,2*W2,2])
    tmpuv = np.concatenate([tmpuv, tmpuv], axis = 1)
    tmpuv = np.reshape(tmpuv, [2*H2,2*W2,2])
    yuv444 = np.concatenate([y, tmpuv], axis = 2)
    yuv444 = np.clip(yuv444, 0, 255)
    # if DEBUG: print(np.shape(yuv444))

    return yuv444

def YUV2RGB(yuv):
    m = np.array([[1.0, 1.0, 1.0],
                  [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                  [1.4019975662231445, -0.7141380310058594, 0.00001542569043522235]])

    rgb = np.dot(yuv, m)
    rgb[:, :, 0] -= 179.45477266423404
    rgb[:, :, 1] += 135.45870971679688
    rgb[:, :, 2] -= 226.8183044444304

    rgb = np.clip(rgb, 0, 255)
    return rgb



def YUV420toRGB(y,uv):
    yuv444 = YUV420toYUV444(y,uv)
    rgb = YUV2RGB(yuv444)
    rgb = np.multiply(rgb, 1.0 / 255)
    return rgb

dataloader = Net_DataLoader(train_path=dataset_path,
							valPath=dataset_path, crop_size_h=-1,
							crop_size_w=-1,
							batch_size=1, normalize=True)

test_loader = dataloader.anv_testDataLoader(input_evs=[(0,1),(-20,1),(4,1)])
test_id=0

for data in test_loader:
	#print(data)
    #print(f'data[0].shape : {data[0].shape}, data[1].shape {data[1].shape}') # data[0].shape : (1, 3072, 4096, 1), data[1].shape (1, 1536, 2048, 2)

    print('Predicting image : ', test_id)
    y_temp = data[2]
    uv_temp = data[3]
    
    uv_temp = np.float32(uv_temp)
    uv_temp = np.minimum(uv_temp, 1.0)
    y_temp = np.float32(y_temp)
    y_temp = np.minimum(y_temp, 1.0)

    st = time.time()
    output_y,output_uv = my_model.test_step([y_temp, uv_temp])
    output_y = np.minimum(output_y, 1)
    output_uv = np.minimum(output_uv, 1)

    time_ = time.time() - st
    y_output = output_y[0, :, :, :]
    uv_output = output_uv[0, :, :, :]
            #print("y_output shape :", y_output.shape) # (512, 512, 1)
            #print("uv_output shape :", uv_output.shape) # (256, 256, 2)
    output = YUV420toYUV444((y_output + 1) * 255.0 / 2.0, (uv_output + 1) * 255.0 / 2.0)
            #print("test time:", time_)
    temp = cv2.cvtColor(np.uint8(output), cv2.COLOR_YUV2BGR)
            #print("Output shape :", temp.shape) # (512, 512, 3)
            #cv2.imwrite(result_dir + "%05d_00_out.jpg" % (test_id), temp)

    img = temp
    #print('Final output image has shape: ', img.shape)
    
    
    string_value = data[4].numpy()
    decoded_path = string_value.decode('utf-8')

    #save_path = f'{result_dir}/{method_name}_epoch_51 {decoded_path}.jpg'
    save_path = f'{result_dir}/{decoded_path}/{method_name}_72epoch {decoded_path}.jpg'

    print(f'Saving file : {save_path}')

    
    cv2.imwrite(f'{save_path}', img)

    gt_y = data[0][0, :, :, :]
    gt_uv = data[1][0, :, :, :]
    #print(f'gt_y shape {gt_y.shape}')
    #print(f'gt_uv shape {gt_uv.shape}')
    gt_yuv = YUV420toYUV444((gt_y + 1) * 255.0 / 2.0, (gt_uv + 1) * 255.0 / 2.0)
    gt = cv2.cvtColor(np.uint8(gt_yuv), cv2.COLOR_YUV2BGR)

    ev0_y = tf.expand_dims(data[2][0, :, :, 0], axis=2) 
    ev0_uv = data[3][0, :, :, 0:2]
    #print(f'ev0_y shape {ev0_y.shape}')
    #print(f'ev0_uv shape {ev0_uv.shape}')

    '''
    gt_y shape (3072, 4096, 1)
    gt_uv shape (1536, 2048, 2)
    ev0_y shape (3072, 4096, 1)
    ev0_uv shape (1536, 2048, 2)
    '''
    ev0_yuv = YUV420toYUV444((ev0_y + 1) * 255.0 / 2.0, (ev0_uv + 1) * 255.0 / 2.0)
    ev0 = cv2.cvtColor(np.uint8(ev0_yuv), cv2.COLOR_YUV2BGR)

    psnr_pred = cv2.PSNR(img, gt)
    psnr_ev0 = cv2.PSNR(ev0, gt)

    ssim_pred = tf.image.ssim(tf.convert_to_tensor(img, dtype=tf.float32), tf.convert_to_tensor(gt, dtype=tf.float32), max_val=255)
    ssim_ev0 = tf.image.ssim(tf.convert_to_tensor(ev0, dtype=tf.float32), tf.convert_to_tensor(gt, dtype=tf.float32), max_val=255)

    print(f'PSNR predicted image: {psnr_pred:.2f}, \nPSNR ev0 reference image: {psnr_ev0:.2f}\n')

    print(f'SSIM predicted image: {ssim_pred:.2f}, \nSSIM ev0 reference image: {ssim_ev0:.2f}\n')


    test_id=test_id+1


