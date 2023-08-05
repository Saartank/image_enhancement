import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


def encode_block_light(inputs, dims, max_pool=True, normalizer_fn=None, ksize1=3, ksize2=3, use_center=True):
    encs = tf.keras.layers.Conv2D(filters=dims, kernel_size=ksize1, activation='relu', padding="same")(inputs)
    #encs = tf.keras.layers.BatchNormalization()(encs)
    global_pool = encs

    results = encs
    if max_pool == True:
        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
												   strides=(2, 2), padding='SAME')
    results = max_pool_2d(results)
	
    return results, encs, global_pool



def upsample_and_concat(x1, x2, output_channels, in_channels, is_fine=True, block_idx=0):	
	#in_channels not used
	pool_size = 2
	# print("x1 shape is: ",x1.shape)
	# print("x2 shape is: ",x2.shape)
	if is_fine == True:
		name = "deconv_fine_0"
		if block_idx > 0:
			name = name+"_%d"%block_idx
		print(f'\nInside upscale and concat.... output_channels = {output_channels}')
		print(f'x1.shape = {x1.shape}, x2.shape = {x2.shape}')
		deconv = tf.keras.layers.Conv2DTranspose(filters= output_channels, kernel_size=2, strides=[pool_size, pool_size], padding="SAME")(x1)
		print(f'deconv.shape = {deconv.shape}')
		deconv_output =  tf.keras.layers.concatenate([deconv, x2], axis=3)
		print(f'deconv_output.shape = {deconv_output.shape}\n')


	return deconv_output

def build_net():

    inp_y = keras.Input(shape=[None, None, 3])
    inp_uv = keras.Input(shape=[None, None, 6])

    #input_placeholder = tf.placeholder(tf.float32, shape=inp_y1.shape)
    #h, w = tf.shape(input_placeholder)[1], tf.shape(input_placeholder)[2]		
    
    inp_y1 = tf.expand_dims(inp_y[:, :, :, 0], axis=3)
    inp_y2 = tf.expand_dims(inp_y[:, :, :, 1], axis=3)
    inp_y3 = tf.expand_dims(inp_y[:, :, :, 2], axis=3)

    inp_uv1 = inp_uv[:, :,:,0:2]
    inp_uv2 = inp_uv[:, :,:,2:4]
    inp_uv3 = inp_uv[:, :,:,4:6]

    print('shapes', inp_y1.shape, inp_uv1.shape)
    print('shapes', inp_y2.shape, inp_uv2.shape)
    print('shapes', inp_y3.shape, inp_uv3.shape)


    _, h, w, _ = tf.shape(inp_y1)
    inp_y2 = tf.image.resize(inp_y2, [h//2,w//2])
    inp_uv2 = tf.image.resize(inp_uv2, [h//4,w//4])

    inp_y3 = tf.image.resize(inp_y3, [h//4,w//4])
    inp_uv3 = tf.image.resize(inp_uv3, [h//8,w//8])

    print('After Resize ------------------')
    print('shapes', inp_y1.shape, inp_uv1.shape)
    print('shapes', inp_y2.shape, inp_uv2.shape)
    print('shapes', inp_y3.shape, inp_uv3.shape)


    dims=8 
    nres_block=2

    # Y encoder
    print('Y Encoder: -----------------------------')

    pool0s, conv1s, conv0 = encode_block_light(inp_y1, dims//2,)
	
    print(f'Enc 0 --> pool0s.shape={pool0s.shape}, conv0.shape={conv0.shape}')

    new_pool0s = tf.keras.layers.concatenate([pool0s, inp_y2], axis=3)
    pool1s, conv1s, conv1 = encode_block_light(new_pool0s, dims)
	
    print(f'Enc 1 --> pool1s.shape={pool1s.shape}, conv0.shape={conv1.shape}')
    new_pool1s = tf.keras.layers.concatenate([pool1s, inp_y3], axis=3)
    pool2s, conv2s, conv2 = encode_block_light(new_pool1s, dims*2)
	
    print(f'Enc 2 --> pool2s.shape={pool2s.shape}, conv2.shape={conv2.shape}')
    #new_pool2s = tf.keras.layers.concatenate([pool2s, inp_uv3], axis=3)
    pool3s, conv3s, conv3 = encode_block_light(pool2s, dims*4)
	
    print(f'Enc 3 --> pool3s.shape={pool3s.shape}, conv3.shape={conv3.shape}')
    pool4s, conv4s, conv4 = encode_block_light(pool3s, dims*8)
	
    print(f'Enc 4 --> pool4s.shape={pool4s.shape}, conv4.shape={conv4.shape}')
    pool5s, conv5s, conv5 = encode_block_light(pool4s, dims*16)

    print(f'Enc 5 --> pool5s.shape={pool5s.shape}, conv5.shape={conv5.shape}')


    #----------------------------------------------------------------------------

    # UV encoder
    print('UV Encoder: -----------------------------')

    uv_pool1s, conv1s, uv_conv1 = encode_block_light(inp_uv1, dims)
	
    print(f'Enc 1 --> pool1s.shape={uv_pool1s.shape}, conv1.shape={uv_conv1.shape}')
    uv_new_pool1s = tf.keras.layers.concatenate([uv_pool1s, inp_uv2], axis=3)
    uv_pool2s, conv2s, uv_conv2 = encode_block_light(uv_new_pool1s, dims*2)
	
    print(f'Enc 2 --> pool2s.shape={uv_pool2s.shape}, conv2.shape={uv_conv2.shape}')
    uv_new_pool2s = tf.keras.layers.concatenate([uv_pool2s, inp_uv3], axis=3)
    uv_pool3s, conv3s, uv_conv3 = encode_block_light(uv_new_pool2s, dims*4)
	
    print(f'Enc 3 --> pool3s.shape={uv_pool3s.shape}, conv3.shape={uv_conv3.shape}')
    uv_pool4s, conv4s, uv_conv4 = encode_block_light(uv_pool3s, dims*8)
	
    print(f'Enc 4 --> pool4s.shape={uv_pool4s.shape}, conv4.shape={uv_conv4.shape}')
    uv_pool5s, conv5s, uv_conv5 = encode_block_light(uv_pool4s, dims*16)

    print(f'Enc 5 --> pool5s.shape={uv_pool5s.shape}, conv5.shape={uv_conv5.shape}')
	
    #net = conv5

    net = tf.keras.layers.concatenate([conv5, uv_conv5], axis=3)
    for i in range(nres_block):
        temp = net
        net = tf.keras.layers.Conv2D(filters=dims*32, kernel_size=3, activation='relu', padding="same")(net)
        net = tf.keras.layers.Conv2D(filters=dims*32, kernel_size=3, activation=None, padding="same")(net)
		#net = se_block(net, dims*16, block_idx=i)
        net = net + temp

    print('Common Decoder: -----------------------------')

    net = tf.keras.layers.Conv2D(filters=dims*16, kernel_size=3, activation=None, padding="same")(net)
    conv5 = tf.keras.layers.concatenate([net, conv5, uv_conv5], axis = 3)

	#up6 = upsample_and_concat(conv5, conv4, dims*8, dims*16, is_fine=True, block_idx=0)
	#conv6 = tf.keras.layers.Conv2D(filters=dims*8, kernel_size=3, activation='relu', padding="same")(up6)

    up6= tf.keras.layers.Conv2DTranspose(filters= dims*8, kernel_size=2, strides=[2, 2], padding="SAME")(conv5)
    #conv6 = up6 + conv4
    conv6 = tf.keras.layers.concatenate([up6, conv4, uv_conv4], axis = 3)
    print(f'Dec 1 --> up6.shape={up6.shape}, conv6.shape={conv6.shape}')


	#up7 = upsample_and_concat(conv6, conv3, dims*4, dims*8, is_fine=True, block_idx=1)
	#conv7 = tf.keras.layers.Conv2D(filters=dims*4, kernel_size=3, activation='relu', padding="same")(up7)
	
    up7= tf.keras.layers.Conv2DTranspose(filters= dims*4, kernel_size=2, strides=[2, 2], padding="SAME")(conv6)
    #conv7 = up7 + conv3
    conv7 = tf.keras.layers.concatenate([up7, conv3, uv_conv3], axis = 3)

    print(f'Dec 2 --> up7.shape={up7.shape}, conv7.shape={conv7.shape}')

	#conv7 = tf.keras.layers.Conv2D(filters=dims*4, kernel_size=3, activation=lrelu, padding="same")(conv7)

	#up8 = upsample_and_concat(conv7, conv2, dims*2, dims*4, is_fine=True, block_idx=2)
	#conv8 = tf.keras.layers.Conv2D(filters=dims*2, kernel_size=3, activation='relu', padding="same")(up8)
    up8 = tf.keras.layers.Conv2DTranspose(filters= dims*2, kernel_size=2, strides=[2, 2], padding="SAME")(conv7)
    #conv8 = up8 + conv2
    conv8 = tf.keras.layers.concatenate([up8, conv2, uv_conv2], axis = 3)


    print(f'Dec 3 --> up8.shape={up8.shape}, conv8.shape={conv8.shape}')

	#conv8 = tf.keras.layers.Conv2D(filters=dims*2, kernel_size=3, activation=lrelu, padding="same")(conv8)

	#up9 = upsample_and_concat(conv8, conv1, dims, dims*2, is_fine=True, block_idx=3)
	#conv9 = tf.keras.layers.Conv2D(filters=dims, kernel_size=3, activation=lrelu, padding="same")(up9)
    up9 = tf.keras.layers.Conv2DTranspose(filters= dims, kernel_size=2, strides=[2, 2], padding="SAME")(conv8)
    #conv9 = up9 + conv1
    conv9 = tf.keras.layers.concatenate([up9, conv1, uv_conv1], axis = 3)

    pred_uv = tf.keras.layers.Conv2D(filters=2, kernel_size=5, strides=(1, 1), padding='same', activation=None)(conv9)

	
    print(f'Dec 4 --> up9.shape={up9.shape}, conv9.shape={conv9.shape}')

	#up10 = upsample_and_concat(conv9, conv0, dims//2, dims, is_fine=True, block_idx=3)
	#conv10 = tf.keras.layers.Conv2D(filters=dims//2, kernel_size=3, activation=lrelu, padding="same")(up10)
    up10 = tf.keras.layers.Conv2DTranspose(filters= dims//2, kernel_size=2, strides=[2, 2], padding="SAME")(conv9)
    #conv10 = up10 + conv0
    conv10 = tf.keras.layers.concatenate([up10, conv0], axis = 3)

	#conv10 = up10 
    print(f'Dec 5 --> up10.shape={up10.shape}, conv10.shape={conv10.shape}')

	#out = tf.keras.layers.Conv2D(filters=dims, kernel_size=1, activation=None, padding="same")(conv10)

    out = conv10
    print(f'out.shape = {out.shape}')
	
    pred_y = tf.keras.layers.Conv2D(filters=1, kernel_size=5, strides=(1, 1), padding='same', activation=None)(
		out)
    #pred_uv = tf.keras.layers.Conv2D(filters=2, kernel_size=5, strides=(2, 2), padding='same', activation=None)(out)
    pred_y = pred_y + inp_y1
    pred_uv = pred_uv + inp_uv1

    print(f'shape of pred_y : {pred_y.shape}')
    print(f'shape of pred_uv : {pred_uv.shape}')

    return keras.Model(inputs=[inp_y, inp_uv], outputs=[pred_y, pred_uv])


def gaussian_blur(img, kernel_size=3, sigma=1):
    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
        return kernel

    gaussian_kernel = gauss_kernel(tf.shape(img)[-1], kernel_size, sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]

    #print('Ee hai gauus kernel :',gaussian_kernel.shape)

    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME')

def getPyramid(img, levels=3):
    #img =tf.expand_dims(img, axis=0)
    pyramid = []
    for i in range(levels):
        #filtered_image = tfa.image.gaussian_filter2d(img, filter_shape=(9, 9), sigma=1.0)
        filtered_image = gaussian_blur(img, kernel_size=3, sigma=1)
        laplacian_image = img - filtered_image
        pyramid.append(laplacian_image)

        img = filtered_image
    
    pyramid.append(img)
    return pyramid

def l1_loss_weighted(img, gt):
  
  par = 3
  weight_map = 1 + par*tf.abs(gt)
  l1 = tf.reduce_sum(input_tensor=tf.abs(weight_map*(img -gt))) 
  return l1

def l1_loss(img, gt):
  l1 = tf.reduce_sum(input_tensor=tf.abs(img -gt)) 
  return l1


class MyModel(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = build_net()
        self.lr = 3*1e-4
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate=self.lr,
			decay_steps=150 * 200,
			decay_rate=0.9,
			staircase=True)
        
    def call(self, inputs):
        inp_y, inp_uv = inputs
        output_y, output_uv = self.model(inp_y, inp_uv)
        return output_y, output_uv
    
	
    def compute_losses(self, output_y,output_uv, y_temp_gt_patch,uv_temp_gt_patch):
        
        levels = 3
        pred_y_pym = getPyramid(output_y, levels)
        pred_uv_pym = getPyramid(output_uv, levels)

        gt_uv_pym = getPyramid(uv_temp_gt_patch, levels)
        gt_y_pym = getPyramid(y_temp_gt_patch, levels)

        level_wise_y_loss = []
        level_wise_uv_loss = []


        weighted_loss = 0

        layer_weights = [3, 15, 30]

        for i in range(levels):
            y_loss, uv_loss = (layer_weights[i])*l1_loss_weighted(pred_y_pym[i], gt_y_pym[i]), (layer_weights[i])*l1_loss(pred_uv_pym[i], gt_uv_pym[i])
            level_wise_y_loss.append(y_loss)
            level_wise_uv_loss.append(uv_loss)

            weighted_loss+=(y_loss+uv_loss)  
        
        # for final gaussian layer:

        y_loss, uv_loss = 0.4*l1_loss(pred_y_pym[levels], gt_y_pym[levels]), 0.4*l1_loss(pred_uv_pym[levels], gt_uv_pym[levels])

        level_wise_y_loss.append(y_loss)
        level_wise_uv_loss.append(uv_loss)

        weighted_loss+=(y_loss+uv_loss)

        loss = {}
        loss['level_wise_y_loss'] = level_wise_y_loss
        loss['level_wise_uv_loss'] = level_wise_uv_loss
        loss['weighted_loss'] = weighted_loss
        return loss
    
    def train_step(self, inp_y, inp_uv, y_gt_scale_1, uv_gt_scale_1):
        with tf.GradientTape() as ae_tape:
            pred_y_scale_1, pred_uv_scale_1= self.model([inp_y, inp_uv])
            loss = self.compute_losses(pred_y_scale_1, pred_uv_scale_1, y_gt_scale_1, uv_gt_scale_1)
        ae_grads = ae_tape.gradient(loss['weighted_loss'], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(ae_grads, self.model.trainable_weights))
        return loss
    
    
    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_schedule)

    def save_weights(self, filepath):
        """While saving the weights, we simply save the weights of the DCE-Net"""
        self.model.save_weights(
            filepath, overwrite=True
        )
    
    def save(self, filepath):
        self.model.save(
            f'{filepath}/my_model.h5', save_format='h5'
        )
    
    def load_weights(self, filepath, by_name=False, skip_mismatch=False):
        """While loading the weights, we simply load the weights of the DCE-Net"""
        self.model.load_weights(
            filepath=filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch
        )
    
    def test_step(self, inputs):
        inp_y, inp_uv = inputs
        pred_y_scale_1, pred_uv_scale_1= self.model([inp_y, inp_uv])
        return pred_y_scale_1, pred_uv_scale_1


if __name__ == "__main__":
      mimo = MyModel()
      mimo.compile()


	    