# test.py
import os, scipy.io, time
import tensorflow as tf
import numpy as np
import cv2
from net_dataloader_test import *
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from model import MyModel


dataset_path = '/mnt/sdd1/sarthakagarwal/Feb_09/copy1_DarkBurst/dark-burst-photography-v2/rear_dataset/test_rear'

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

    print(f'y_min : {tf.reduce_min(y_temp)}')
    print(f'uv_min : {tf.reduce_min(uv_temp)}')
    print(f'y_max : {tf.reduce_max(y_temp)}')
    print(f'uv_max : {tf.reduce_max(uv_temp)}')
