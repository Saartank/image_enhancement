import tensorflow as tf
# import burst_nets
# import burst_nets_concatenate
# import burst_nets_no_map_fn_no_norm_yuv
import burst_nets_no_map_fn_no_norm_yuv_no_std
# Convert the model
method_name = "Mimo_May_16_all_upscale_concat_to_add"
saved_model_dir = '../saved_model_v2_light_4_res_2_encoder_1_no_map_fn_no_norm/YUV_-20_0_4_mean_2k_no_batch_1_25_y0uv0_cat/%s/'% method_name
model = tf.keras.models.load_model(saved_model_dir, compile=False)
# model=tf.keras.models.load_model('./ED-1536-2048', compile=False)
# model=tf.keras.models.load_model('./ED-720-1280', compile=False)
# model=tf.keras.models.load_model('./ED-3000-4000', compile=False)
# print(model.summary())
# exit()
# model=tf.keras.models.load_model('./ED-720-1280', compile=False)
model.save_weights('../checkpoints_v2_weights/my_checkpoint')

inp_y_scale_1_ev24 = tf.keras.Input(shape=[4096, 3072, 1], dtype=tf.float32, batch_size=1)
inp_uv_scale_1_ev24 = tf.keras.Input(shape=[2048, 1536, 2], dtype=tf.float32, batch_size=1)

inp_y_scale_2_ev0 = tf.keras.Input(shape=[2048, 1536, 1], dtype=tf.float32, batch_size=1)
inp_uv_scale_2_ev0 = tf.keras.Input(shape=[1024, 768, 2], dtype=tf.float32, batch_size=1)
		

inp_y_scale_3_ev4 = tf.keras.Input(shape=[1024, 768, 1], dtype=tf.float32, batch_size=1)
inp_uv_scale_3_ev4 = tf.keras.Input(shape=[512, 384, 2], dtype=tf.float32, batch_size=1)
				

pred_y_scale_1, pred_uv_scale_1 = burst_nets_no_map_fn_no_norm_yuv_no_std.fine_mimo_mod(inp_y_scale_1_ev24, inp_uv_scale_1_ev24, inp_y_scale_2_ev0, inp_uv_scale_2_ev0, inp_y_scale_3_ev4, inp_uv_scale_3_ev4)

model2 =tf.keras.Model(inputs=[inp_y_scale_1_ev24, inp_uv_scale_1_ev24, inp_y_scale_2_ev0, inp_uv_scale_2_ev0, inp_y_scale_3_ev4, inp_uv_scale_3_ev4], outputs=[pred_y_scale_1, pred_uv_scale_1],name='dbp')
model2.summary()
# model = UIC_NET(y_h=1500, y_w=2000, uv_h=750, uv_w=1000).get_UIC_NET()
# model = UIC_NET(y_h=768, y_w=1280, uv_h=384, uv_w=640).get_UIC_NET()
# model.build(input_shape=[(1, 720, 1280, 1), (1, 360, 640, 2)])
model2.load_weights('../checkpoints_v2_weights/my_checkpoint').expect_partial()
model2.save(saved_model_dir+"tflite")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir+"tflite") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open(f'{method_name}.tflite', 'wb') as f:
  f.write(tflite_model)
