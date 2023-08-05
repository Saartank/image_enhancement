from glob import glob
import random
import tensorflow as tf
import os
import pathlib
import numpy as np
import pandas as pd
import cv2
DEBUG = 1
def invnormalize(input_image):
    return ((input_image + 1.0) / (2.0 / 255.0))



class Net_DataLoader(object):

    def __init__(self,train_path, valPath, crop_size_h = -1,crop_size_w=-1, batch_size = 5, max_pixel_val = 255.0,
                 prefetch = 2,normalize=True,
                 strategy=None):

        self.train_path = train_path
        self.valPath = valPath
        self.batch_size = batch_size
        self.crop_size_h = crop_size_h
        self.crop_size_w = crop_size_w
        self.max_pixel_val = max_pixel_val
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.prefetch = prefetch
        self.normalize_flag = normalize
        # self.min_ssim = min_ssim
        # self.max_ssim = max_ssim
        # self.sharpness_factor = sharpness_factor
        self.strategy = strategy
        self.resize = True



    def convert_csvs_2_dataframe(self, trainPath):
        csv_paths = []
        #get all csv files
        for root, dirs, files in os.walk(trainPath):
            for file in files:
                if file.endswith('.csv'):
                    csv_paths.append(root + "/" + file)

        #combine all csvs to single dataframe
        full_df = pd.DataFrame()

        for f in csv_paths:
            p = pathlib.Path(f)
            df = pd.read_csv(f, delimiter=",")
            # make relative filepaths in csv into absolute file paths
            df["sharp_file"] = p.parent.as_posix() + "/" + df["sharp_file"]
            df["blur_file"] = p.parent.as_posix() + "/" + df["blur_file"]
            # df["base_file"] = p.parent.as_posix() + "/" + df["base_file"]
            full_df = pd.concat([full_df, df],ignore_index=True)

        return full_df


    def get_file_paths(self, folder_path):
        image_paths = []
        #get image files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.png',".jpg",".jpeg")):
                    image_paths.append(root + "/" + file)

        return image_paths

    def get_json_paths(self,folder_path):
        folder_paths = []
        #get scene folders
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".json")):
                    folder_paths.append(root + "/")

        return folder_paths

    def get_filtered_files(self, dataframe,min_ssim,max_ssim,sharpness_factor):
        sharp_df = dataframe
        # sharp_df = dataframe[dataframe.sharpness >= sharpness_factor * dataframe.max_sharpness]
        # sharp_df = sharp_df[sharp_df.SSIM >= min_ssim]
        # sharp_df = sharp_df[sharp_df.SSIM < max_ssim]
        sharp_files = sharp_df.sharp_file.tolist()
        blur_files = sharp_df.blur_file.tolist()
        # ssim_list = sharp_df.SSIM.tolist()
        # psnr_list = sharp_df.PSNR.tolist()
        return sharp_files,blur_files #,ssim_list,psnr_list

    def RGB2YUV(self, rgb):
        m = tf.constant([[ 0.29900, -0.16874,  0.50000],
                     [0.58700, -0.33126, -0.41869],
                     [ 0.11400, 0.50000, -0.08131]])
        yuv = tf.tensordot(rgb, m, axes = 1)
        y = yuv[:,:,0]
        u = tf.math.add(yuv[:,:,1],tf.constant(128.0, dtype=tf.float32))
        v = tf.math.add(yuv[:,:,2],tf.constant(128.0, dtype=tf.float32))
        newYUV = tf.stack([y, u, v], axis=2)
        newYUV = tf.clip_by_value(newYUV,0.0,255.0)
        return newYUV

    def YUV444toYUV420(self, yuv444):
        y = yuv444[:,:,0]
        uv = yuv444[:,:,1:]
        uv = (uv[0::2,0::2,:] + uv[0::2,1::2,:] + uv[1::2,0::2,:] + uv[1::2,1::2,:]) * 0.25
        return y, uv

    def normalize(self, input_image):
        return (input_image / (self.max_pixel_val/2.0)) - 1



    def random_crop(self, gt_image,input_images):
        if(gt_image.shape != input_images[0].shape):
            gt_image = tf.image.resize_with_crop_or_pad(gt_image,target_height=3072,target_width=4080)
        images = [gt_image] + input_images
        stacked_image = tf.stack(images, axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[len(images), self.crop_size_h, self.crop_size_w, 3])
        return cropped_image[0], cropped_image[1:]


    def get_files_from_folder(self,folder_path, input_evs):
        files = os.listdir(folder_path)
        files.sort()
        jpgs = []
        for f in files:
            if (b'.jpg' in f):
                jpgs.append(f)

        gts = []
        evs = {}
        for jpg in jpgs:
            if (b'_gt_' in jpg):
                gts.append(jpg)
            elif (b'_ev_' in jpg):
                index = jpg.index(b'_ev_')
                ev_num = jpg[index + 4:-4]
                ev_num = int(float(ev_num))
                if (ev_num not in evs.keys()):
                    evs[ev_num] = []
                    evs[ev_num].append(jpg)
                else:
                    evs[ev_num].append(jpg)
        gts.sort()
        gt_path = folder_path + gts[0]
        inp_ev_paths = []
        for inp_ev, num_frames in input_evs:
            inp_ev_paths.extend(evs[inp_ev][0:num_frames])

        inp_ev_paths = [folder_path + i for i in inp_ev_paths]

        return gt_path, inp_ev_paths

    def transform_train_anv(self,gt_file_path,*inp_ev_paths):
        # gt_file_path, inp_ev_paths = self.get_files_from_folder(folder_path,input_evs=input_evs)

        # extract images
        target_image = tf.io.read_file(gt_file_path)
        target_image = tf.image.decode_image(target_image, channels=3)

        input_images=[]
        for inp_ev_path in inp_ev_paths:
            input_image = tf.io.read_file(inp_ev_path)
            input_image = tf.image.decode_image(input_image, channels=3)
            input_images.append(input_image)

        #random crop
        if(self.crop_size_h != -1):
             target_image, input_images = self.random_crop(target_image,input_images)

        #TODO convert to mat-math
        #Convert to YUV444 and then to YUV420
        input_images_y420 = []
        input_images_uv420 = []
        for input_image in input_images:
            input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
            input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
            if(self.normalize_flag):
                input_image_y420, input_image_uv420 = self.normalize(input_image_y420), self.normalize(input_image_uv420)
            # input_image_y420 = tf.expand_dims(input_image_y420, axis=2)

            input_images_y420.append(input_image_y420)
            input_images_uv420.append(input_image_uv420)


        target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
        target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)
        if(self.normalize_flag):
            target_image_y420, target_image_uv420 =self.normalize(target_image_y420), self.normalize(target_image_uv420)
        # target_image_y420 = tf.expand_dims(target_image_y420,axis=2)


        return target_image_y420,target_image_uv420,input_images_y420,input_images_uv420

    def transform_train(self, input, target):
        file_input = input
        file_target = target

        #Read paths and extract images
        input_image = tf.io.read_file(input)
        input_image = tf.image.decode_image(input_image, channels=3)
        target_image = tf.io.read_file(target)
        target_image = tf.image.decode_image(target_image, channels=3)

        # TODO : change resize and move it outside training process
        # if(self.resize):
        #     input_image = tf.image.resize_with_pad(input_image,target_height=960,target_width=1280)
        #     target_image = tf.image.resize_with_pad(target_image,target_height=960,target_width=1280)

        #random crop
        if(self.crop_size_h != -1):
            input_image, target_image = self.random_crop(input_image, target_image)
            # random horizontal flip
            # if tf.random.uniform(()) > 0.5:
            #     input_image = tf.image.flip_left_right(input_image)
            #     target_image = tf.image.flip_left_right(target_image)
            # # random vertical flip
            # if tf.random.uniform(()) > 0.5:
            #     input_image = tf.image.flip_up_down(input_image)
            #     target_image = tf.image.flip_up_down(target_image)
        #Convert to YUV444 and then to YUV420
        input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
        input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
        target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
        target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)

        #normalize tensors between -1 and 1
        if(self.normalize_flag):
            input_image_y420, input_image_uv420, target_image_y420, target_image_uv420 = self.normalize(input_image_y420), self.normalize(input_image_uv420),\
                                                             self.normalize(target_image_y420), self.normalize(target_image_uv420)
        input_image_y420 = tf.expand_dims(input_image_y420,axis=2)
        target_image_y420 = tf.expand_dims(target_image_y420,axis=2)
        return input_image_y420, input_image_uv420, target_image_y420, target_image_uv420, file_input, file_target

    def transform_val(self, inputCud, inputNonCud):
        fileCUD = inputCud
        fileNonCUD = inputNonCud
        #Read paths and extract images
        inputCud = tf.io.read_file(inputCud)
        cud = tf.image.decode_image(inputCud, channels=3)
        inputNonCud = tf.io.read_file(inputNonCud)
        noncud = tf.image.decode_image(inputNonCud, channels=3)
        #random crop
        if(self.crop_size_h != -1):
            cud, noncud = self.random_crop(cud, noncud)
        #Convert to YUV444 and then to YUV420
        cud444 = self.RGB2YUV(tf.cast(cud, tf.float32))
        cud_y420, cud_uv420 = self.YUV444toYUV420(cud444)
        noncud444 = self.RGB2YUV(tf.cast(noncud, tf.float32))
        noncud_y420, noncud_uv420 = self.YUV444toYUV420(noncud444)
        #normalize tensors between -1 and 1
        if(self.normalize_flag):
            cud_y420, cud_uv420, noncud_y420, noncud_uv420 = self.normalize(cud_y420), self.normalize(cud_uv420),\
                                                             self.normalize(noncud_y420), self.normalize(noncud_uv420)
        cud_y420 = tf.expand_dims(cud_y420,axis=2)
        noncud_y420 = tf.expand_dims(noncud_y420,axis=2)
        return cud_y420, cud_uv420, noncud_y420, noncud_uv420, fileCUD, fileNonCUD

    def transform_test(self, input, target):
        file_input = input
        file_target = target

        #Read paths and extract images
        input_image = tf.io.read_file(input)
        input_image = tf.image.decode_image(input_image, channels=3)
        target_image = tf.io.read_file(target)
        target_image = tf.image.decode_image(target_image, channels=3)
        # out_image = tf.io.read_file(out)
        # out_image = tf.image.decode_image(out_image, channels=3)

        # if(self.resize):
        #     input_image = tf.image.resize_with_pad(input_image,target_height=960,target_width=1280)
        #     target_image = tf.image.resize_with_pad(target_image,target_height=960,target_width=1280)

        #random crop
        # if(self.crop_size_h != -1):
        #     input_image, target_image = self.random_crop(input_image, target_image)
            # random horizontal flip
            # if tf.random.uniform(()) > 0.5:
            #     input_image = tf.image.flip_left_right(input_image)
            #     target_image = tf.image.flip_left_right(target_image)
            # # random vertical flip
            # if tf.random.uniform(()) > 0.5:
            #     input_image = tf.image.flip_up_down(input_image)
            #     target_image = tf.image.flip_up_down(target_image)
        #Convert to YUV444 and then to YUV420
        input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
        input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
        target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
        target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)
        # out_image_444 = self.RGB2YUV(tf.cast(out_image, tf.float32))
        # out_image_y420, out_image_uv420 = self.YUV444toYUV420(out_image_444)

        #normalize tensors between -1 and 1
        if(self.normalize_flag):
            input_image_y420, input_image_uv420, target_image_y420, target_image_uv420 = self.normalize(input_image_y420), self.normalize(input_image_uv420),\
                                                             self.normalize(target_image_y420), self.normalize(target_image_uv420)
            # out_image_y420,out_image_uv420 = self.normalize(out_image_y420), self.normalize(out_image_uv420)
        input_image_y420 = tf.expand_dims(input_image_y420,axis=2)
        target_image_y420 = tf.expand_dims(target_image_y420,axis=2)
        # out_image_y420 = tf.expand_dims(out_image_y420,axis=2)
        return input_image_y420, input_image_uv420, target_image_y420, target_image_uv420, file_input, file_target


    def transform_predict(self,input_image):
        fileImage = input_image
        # Read paths and extract images
        input_image = tf.io.read_file(input_image)
        blur = tf.image.decode_image(input_image, channels=3)

        # random crop
        if (self.crop_size_h != -1):
            blur = tf.image.random_crop(blur,size = [self.crop_size_h,self.crop_size_w,3])
        # # random horizontal flip
        # if tf.random.uniform(()) > 0.5:
        #     cud = tf.image.flip_left_right(cud)
        #     noncud = tf.image.flip_left_right(noncud)
        # # random vertical flip
        # if tf.random.uniform(()) > 0.5:
        #     cud = tf.image.flip_up_down(cud)
        #     noncud = tf.image.flip_up_down(noncud)
        # Convert to YUV444 and then to YUV420
        blur444 = self.RGB2YUV(tf.cast(blur, tf.float32))
        blur_y420, blur_uv420 = self.YUV444toYUV420(blur444)

        # normalize tensors between -1 and 1
        if (self.normalize_flag):
            blur_y420, blur_uv420= self.normalize(blur_y420), self.normalize(blur_uv420)
        blur_y420 = tf.expand_dims(blur_y420, axis=2)

        return blur_y420,blur_uv420, fileImage

    def transform_folder_predict(self,input_image):
        fileImage = input_image
        # Read paths and extract images
        input_image = tf.io.read_file(input_image)
        blur = tf.image.decode_image(input_image, channels=3)
        # blur = tf.image.resize(blur, size=[1536, 2048])
        # blur = tf.image.resize(blur, size=[blur.get_shape()[0] // 2, blur.get_shape()[1] // 2])
        blur444 = self.RGB2YUV(tf.cast(blur, tf.float32))
        blur_y420, blur_uv420 = self.YUV444toYUV420(blur444)

        # normalize tensors between -1 and 1
        if (self.normalize_flag):
            blur_y420, blur_uv420= self.normalize(blur_y420), self.normalize(blur_uv420)
        blur_y420 = tf.expand_dims(blur_y420, axis=2)

        return blur_y420,blur_uv420, fileImage

    def transform_downsample_predict(self,input_image):
        fileImage = input_image
        # Read paths and extract images
        input_image = tf.io.read_file(input_image)
        blur = tf.image.decode_image(input_image, channels=3)

        # random crop
        if (self.crop_size_h != -1):
            blur = tf.image.random_crop(blur,size = [self.crop_size_h,self.crop_size_w,3])
        # # random horizontal flip
        # if tf.random.uniform(()) > 0.5:
        #     cud = tf.image.flip_left_right(cud)
        #     noncud = tf.image.flip_left_right(noncud)
        # # random vertical flip
        # if tf.random.uniform(()) > 0.5:
        #     cud = tf.image.flip_up_down(cud)
        #     noncud = tf.image.flip_up_down(noncud)
        # Convert to YUV444 and then to YUV420
        blur = tf.image.resize(blur,size=[blur.get_shape()[0]//2,blur.get_shape()[1]//2])
        blur444 = self.RGB2YUV(tf.cast(blur, tf.float32))
        blur_y420, blur_uv420 = self.YUV444toYUV420(blur444)

        # normalize tensors between -1 and 1
        if (self.normalize_flag):
            blur_y420, blur_uv420= self.normalize(blur_y420), self.normalize(blur_uv420)
        blur_y420 = tf.expand_dims(blur_y420, axis=2)

        return blur_y420,blur_uv420, fileImage

    def anv_trainDataLoader(self,input_evs):
        folder_paths = self.get_json_paths(self.train_path)

        train_dataset = tf.data.Dataset.from_generator(generator=self.anv_generator,args=[folder_paths,input_evs,self.batch_size], output_types=(tf.float32,tf.float32,
                                                                                                  tf.float32,tf.float32),
                                                       output_shapes=((None,self.crop_size_h,self.crop_size_w,1),(None,self.crop_size_h//2,self.crop_size_w//2,2),
                                                                      (None,self.crop_size_h,self.crop_size_w,3),(None,self.crop_size_h//2,self.crop_size_w//2,6)))
        if(self.strategy):
            train_dataset = train_dataset.batch(self.batch_size * self.strategy.num_replicas_in_sync).prefetch(self.prefetch)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.prefetch(self.prefetch)

        return train_dataset

    def anv_testDataLoader(self,input_evs):
        folder_paths = self.get_json_paths(self.train_path)

        train_dataset = tf.data.Dataset.from_generator(generator=self.anv_generator_test,args=[folder_paths,input_evs,self.batch_size], output_types=(tf.float32,tf.float32,
                                                                                                  tf.float32,tf.float32, tf.string),
                                                       output_shapes=((None,3072,4096,1),(None,3072//2,4096//2,2),
                                                                      (None,3072,4096,3),(None,3072//2,4096//2,6), tf.TensorShape(None)))
        if(self.strategy):
            train_dataset = train_dataset.batch(self.batch_size * self.strategy.num_replicas_in_sync).prefetch(self.prefetch)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.prefetch(self.prefetch)

        return train_dataset

    def trainDataLoader(self):
        input_images = tf.data.Dataset.list_files(self.train_path + "*_input.jpg", shuffle=False)
        gt_images = tf.data.Dataset.list_files(self.train_path + "*_target.jpg", shuffle=False)
        # self.train_dataframe = self.convert_csvs_2_dataframe(self.train_path)
        # print("full dataset size :{}".format(self.train_dataframe.shape))
        # sharp_files,blur_files= self.get_filtered_files(self.train_dataframe,min_ssim=self.min_ssim,max_ssim=self.max_ssim,sharpness_factor=self.sharpness_factor)
        # print("filtered dataset size :{}".format(len(blur_files)))
        # sharp_images = tf.data.Dataset.from_tensor_slices(sharp_files)
        # blur_images = tf.data.Dataset.from_tensor_slices(blur_files)
        pairImages = tf.data.Dataset.zip((input_images, gt_images))
        train_dataset = pairImages.shuffle(tf.data.experimental.cardinality(pairImages).numpy(), reshuffle_each_iteration=True)
        train_dataset = train_dataset.map(self.transform_train, num_parallel_calls=self.AUTOTUNE)
        if(self.strategy):
            train_dataset = train_dataset.batch(self.batch_size * self.strategy.num_replicas_in_sync).prefetch(self.prefetch)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.batch(self.batch_size).prefetch(self.prefetch)
        return train_dataset

    #TODO update test datalaoder to be different from train dataloader
    def testDataLoader(self):
        input_images = tf.data.Dataset.list_files(self.train_path + "*_input.jpg", shuffle=False)
        gt_images = tf.data.Dataset.list_files(self.train_path + "*_target.jpg", shuffle=False)
        # self.train_dataframe = self.convert_csvs_2_dataframe(self.train_path)
        # print("full dataset size :{}".format(self.train_dataframe.shape))
        # sharp_files,blur_files= self.get_filtered_files(self.train_dataframe,min_ssim=self.min_ssim,max_ssim=self.max_ssim,sharpness_factor=self.sharpness_factor)
        # print("filtered dataset size :{}".format(len(blur_files)))
        # sharp_images = tf.data.Dataset.from_tensor_slices(sharp_files)
        # blur_images = tf.data.Dataset.from_tensor_slices(blur_files)
        pairImages = tf.data.Dataset.zip((input_images, gt_images))
        train_dataset = pairImages.shuffle(tf.data.experimental.cardinality(pairImages).numpy(), reshuffle_each_iteration=False)
        train_dataset = train_dataset.map(self.transform_test, num_parallel_calls=self.AUTOTUNE)
        if(self.strategy):
            train_dataset = train_dataset.batch(self.batch_size * self.strategy.num_replicas_in_sync).prefetch(self.prefetch)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.batch(self.batch_size).prefetch(self.prefetch)
        return train_dataset


    def folderDataLoader(self):
        files_list = self.get_file_paths(self.valPath)
        pimages = tf.data.Dataset.from_tensor_slices(files_list)
        predict_dataset = pimages.map(self.transform_folder_predict,num_parallel_calls=self.AUTOTUNE)
        predict_dataset = predict_dataset.batch(self.batch_size)
        return predict_dataset

    def anv_generator(self,folder_paths, input_evs,batch_size):
        DEBUG = 1
        i=0
        np.random.shuffle(folder_paths)
        while True:
            if i*batch_size >= len(folder_paths):
                i=0
                return
            else:
                batch_chunk = folder_paths[i*batch_size:(i+1)*batch_size]
                target_batch_y = []
                target_batch_uv = []
                input_batch_y = []
                input_batch_uv = []
                if(DEBUG):
                    print(b'batch_size: '+ str(len(batch_chunk)).encode())
                for path in batch_chunk:
                    if(DEBUG):
                        print(b'folder path: '+path)
                    gt_file_path, inp_ev_paths = self.get_files_from_folder(path, input_evs=input_evs)

                    # extract images
                    target_image = tf.io.read_file(gt_file_path)
                    target_image = tf.image.decode_image(target_image, channels=3)

                    input_images = []
                    for inp_ev_path in inp_ev_paths:
                        input_image = tf.io.read_file(inp_ev_path)
                        input_image = tf.image.decode_image(input_image, channels=3)
                        input_images.append(input_image)

                    # random crop
                    if (self.crop_size_h != -1):
                        target_image, input_images = self.random_crop(target_image, input_images)

                    # TODO convert to mat-math
                    # Convert to YUV444 and then to YUV420
                    input_images_y420 = []
                    input_images_uv420 = []
                    for input_image in input_images:
                        input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
                        input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
                        if (self.normalize_flag):
                            input_image_y420, input_image_uv420 = self.normalize(input_image_y420), self.normalize(
                                input_image_uv420)
                        input_image_y420 = tf.expand_dims(input_image_y420, axis=2)

                        input_images_y420.append(input_image_y420)
                        input_images_uv420.append(input_image_uv420)

                    #concat input images along channels
                    input_images_y420 = np.concatenate(np.asarray(input_images_y420),axis=-1)
                    input_images_uv420 = np.concatenate(np.asarray(input_images_uv420),axis=-1)

                    target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
                    target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)
                    if (self.normalize_flag):
                        target_image_y420, target_image_uv420 = self.normalize(target_image_y420), self.normalize(
                            target_image_uv420)
                    target_image_y420 = tf.expand_dims(target_image_y420,axis=2)
                    target_batch_y.append(target_image_y420)
                    target_batch_uv.append(target_image_uv420)
                    input_batch_y.append(input_images_y420)
                    input_batch_uv.append(input_images_uv420)


                target_batch_y = np.asarray(target_batch_y)#.reshape((-1,1024,1024,1))
                target_batch_uv = np.asarray(target_batch_uv)#.reshape((-1,512,512,2))
                input_batch_y = np.asarray(input_batch_y)#.reshape((-1,3,-1,-1,-1))
                input_batch_uv = np.asarray(input_batch_uv)#.reshape((-1,3,-1,-1,-1))

                yield target_batch_y, target_batch_uv , input_batch_y, input_batch_uv, 
                i += 1
    def anv_generator_test(self,folder_paths, input_evs,batch_size):
        DEBUG = 1
        i=0
        np.random.shuffle(folder_paths)
        while True:
            if i*batch_size >= len(folder_paths):
                i=0
                return
            else:
                batch_chunk = folder_paths[i*batch_size:(i+1)*batch_size]
                target_batch_y = []
                target_batch_uv = []
                input_batch_y = []
                input_batch_uv = []
                if(DEBUG):
                    print(b'batch_size: '+ str(len(batch_chunk)).encode())
                for path in batch_chunk:
                    if(DEBUG):
                        print(b'folder path: '+path)
                    gt_file_path, inp_ev_paths = self.get_files_from_folder(path, input_evs=input_evs)

                    # extract images
                    target_image = tf.io.read_file(gt_file_path)
                    target_image = tf.image.decode_image(target_image, channels=3)
                    target_image = np.array(cv2.resize(np.array(target_image), (4096, 3072), interpolation=cv2.INTER_AREA))
                    input_images = []
                    for inp_ev_path in inp_ev_paths:
                        input_image = tf.io.read_file(inp_ev_path)
                        input_image = tf.image.decode_image(input_image, channels=3)
                        input_image = cv2.resize(np.array(input_image), (4096, 3072), interpolation=cv2.INTER_AREA)
                        input_images.append(np.array(input_image))

                    # random crop
                    if (self.crop_size_h != -1):
                        target_image, input_images = self.random_crop(target_image, input_images)

                    # TODO convert to mat-math
                    # Convert to YUV444 and then to YUV420
                    input_images_y420 = []
                    input_images_uv420 = []
                    for input_image in input_images:
                        input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
                        input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
                        if (self.normalize_flag):
                            input_image_y420, input_image_uv420 = self.normalize(input_image_y420), self.normalize(
                                input_image_uv420)
                        input_image_y420 = tf.expand_dims(input_image_y420, axis=2)

                        input_images_y420.append(input_image_y420)
                        input_images_uv420.append(input_image_uv420)

                    #concat input images along channels
                    input_images_y420 = np.concatenate(np.asarray(input_images_y420),axis=-1)
                    input_images_uv420 = np.concatenate(np.asarray(input_images_uv420),axis=-1)

                    target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
                    target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)
                    if (self.normalize_flag):
                        target_image_y420, target_image_uv420 = self.normalize(target_image_y420), self.normalize(
                            target_image_uv420)
                    target_image_y420 = tf.expand_dims(target_image_y420,axis=2)
                    target_batch_y.append(target_image_y420)
                    target_batch_uv.append(target_image_uv420)
                    input_batch_y.append(input_images_y420)
                    input_batch_uv.append(input_images_uv420)


                target_batch_y = np.asarray(target_batch_y)#.reshape((-1,1024,1024,1))
                target_batch_uv = np.asarray(target_batch_uv)#.reshape((-1,512,512,2))
                input_batch_y = np.asarray(input_batch_y)#.reshape((-1,3,-1,-1,-1))
                input_batch_uv = np.asarray(input_batch_uv)#.reshape((-1,3,-1,-1,-1))

                #print(f'type(path) : {type(path)}')

                name = str(path).split('/')[-2]
                #print(f'name : {name}')
                res = bytes(name, 'utf-8')

                test_path = tf.constant(res, dtype=tf.string)


                #test_path = tf.constant(path.split('/')[-2].encode('utf-8'), dtype=tf.string)
                
                #test_path = tf.convert_to_tensor(path.split('/')[-2], dtype=tf.string) 

                yield target_batch_y, target_batch_uv , input_batch_y, input_batch_uv, test_path 
                i += 1
    def get_dnz_scene_files_from_folder(self,folder_path, input_evs):
        files = os.listdir(folder_path)
        files.sort()
        jpgs = []
        for f in files:
            if ('.jpg' in f):
                jpgs.append(f)

        gts = []
        evs = {}
        for jpg in jpgs:
            if ('_gt_' in jpg):
                gts.append(jpg)
            # elif ('_ev' in jpg and 'dnz' in jpg):
            elif ('_ev_' in jpg):
                index = jpg.index('_ev')
                endex = jpg.index('_',index+1)
                # ev_num = jpg[index + 3:endex]
                ev_num = jpg[endex+1:-4]
                ev_num = int(float(ev_num))
                if (ev_num not in evs.keys()):
                    evs[ev_num] = []
                    evs[ev_num].append(jpg)
                else:
                    evs[ev_num].append(jpg)
        gts.sort()
        gt_path = folder_path + gts[0]
        inp_ev_paths = []
        for inp_ev, num_frames in input_evs:
            inp_ev_paths.extend(evs[inp_ev][0:num_frames])

        inp_ev_paths = [folder_path + i for i in inp_ev_paths]

        return gt_path, inp_ev_paths
    def test_dnz_scene_anv_generator(self,folder_paths,input_evs,batch_size):
        DEBUG = 1
        i=0
        while True:
            if i*batch_size >= len(folder_paths):
                i=0
                return
            else:
                batch_chunk = folder_paths[i*batch_size:(i+1)*batch_size]
                target_batch_y = []
                target_batch_uv = []
                input_batch_y = []
                input_batch_uv = []
                if(DEBUG):
                    print(b'batch_size: '+ str(len(batch_chunk)).encode())
                for path in batch_chunk:
                    if(DEBUG):
                        print(b'folder path: '+path)
                    gt_file_path, inp_ev_paths = self.get_dnz_scene_files_from_folder(path, input_evs=input_evs)

                    # extract images
                    target_image = tf.io.read_file(gt_file_path)
                    target_image = tf.image.decode_image(target_image, channels=3)

                    input_images = []
                    for inp_ev_path in inp_ev_paths:
                        input_image = tf.io.read_file(inp_ev_path)
                        input_image = tf.image.decode_image(input_image, channels=3)
                        input_images.append(input_image)

                    # random crop
                    # if (self.crop_size_h != -1):
                    #     target_image, input_images = self.random_crop(target_image, input_images)

                    # TODO convert to mat-math
                    # Convert to YUV444 and then to YUV420
                    input_images_y420 = []
                    input_images_uv420 = []
                    for input_image in input_images:
                        input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
                        input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
                        if (self.normalize_flag):
                            input_image_y420, input_image_uv420 = self.normalize(input_image_y420), self.normalize(
                                input_image_uv420)
                        input_image_y420 = tf.expand_dims(input_image_y420, axis=2)

                        input_images_y420.append(input_image_y420)
                        input_images_uv420.append(input_image_uv420)

                    #concat input images along channels
                    input_images_y420 = np.concatenate(np.asarray(input_images_y420),axis=-1)
                    input_images_uv420 = np.concatenate(np.asarray(input_images_uv420),axis=-1)

                    target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
                    target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)
                    if (self.normalize_flag):
                        target_image_y420, target_image_uv420 = self.normalize(target_image_y420), self.normalize(
                            target_image_uv420)
                    target_image_y420 = tf.expand_dims(target_image_y420,axis=2)
                    target_batch_y.append(target_image_y420)
                    target_batch_uv.append(target_image_uv420)
                    input_batch_y.append(input_images_y420)
                    input_batch_uv.append(input_images_uv420)


                target_batch_y = np.asarray(target_batch_y)#.reshape((-1,1024,1024,1))
                target_batch_uv = np.asarray(target_batch_uv)#.reshape((-1,512,512,2))
                input_batch_y = np.asarray(input_batch_y)#.reshape((-1,3,-1,-1,-1))
                input_batch_uv = np.asarray(input_batch_uv)#.reshape((-1,3,-1,-1,-1))

                yield  input_batch_y, input_batch_uv,target_batch_y, target_batch_uv
                i += 1

    def dnz_anv_generator(self,folder_paths,input_evs,batch_size):
        DEBUG = 1
        i=0
        np.random.shuffle(folder_paths)
        while True:
            if i*batch_size >= len(folder_paths):
                i=0
                return
            else:
                batch_chunk = folder_paths[i*batch_size:(i+1)*batch_size]
                target_batch_y = []
                target_batch_uv = []
                input_batch_y = []
                input_batch_uv = []
                # if(DEBUG):
                #     print('batch_size: '+ str(len(batch_chunk)).encode())
                for path in batch_chunk:
                    # if(DEBUG):
                    #     print('folder path: '+path)
                    gt_file_path, inp_ev_paths = self.get_dnz_scene_files_from_folder(path, input_evs=input_evs)

                    # extract images
                    target_image = tf.io.read_file(gt_file_path)
                    target_image = tf.image.decode_image(target_image, channels=3)

                    input_images = []
                    for inp_ev_path in inp_ev_paths:
                        input_image = tf.io.read_file(inp_ev_path)
                        input_image = tf.image.decode_image(input_image, channels=3)
                        input_images.append(input_image)

                    # random crop
                    if (self.crop_size_h != -1):
                        target_image, input_images = self.random_crop(target_image, input_images)

                    # TODO convert to mat-math
                    # Convert to YUV444 and then to YUV420
                    input_images_y420 = []
                    input_images_uv420 = []
                    for input_image in input_images:
                        input_image_444 = self.RGB2YUV(tf.cast(input_image, tf.float32))
                        input_image_y420, input_image_uv420 = self.YUV444toYUV420(input_image_444)
                        if (self.normalize_flag):
                            input_image_y420, input_image_uv420 = self.normalize(input_image_y420), self.normalize(
                                input_image_uv420)
                        input_image_y420 = tf.expand_dims(input_image_y420, axis=2)

                        input_images_y420.append(input_image_y420)
                        input_images_uv420.append(input_image_uv420)

                    #concat input images along channels
                    input_images_y420 = np.concatenate(np.asarray(input_images_y420),axis=-1)
                    input_images_uv420 = np.concatenate(np.asarray(input_images_uv420),axis=-1)

                    target_image_444 = self.RGB2YUV(tf.cast(target_image, tf.float32))
                    target_image_y420, target_image_uv420 = self.YUV444toYUV420(target_image_444)
                    if (self.normalize_flag):
                        target_image_y420, target_image_uv420 = self.normalize(target_image_y420), self.normalize(
                            target_image_uv420)
                    target_image_y420 = tf.expand_dims(target_image_y420,axis=2)
                    target_batch_y.append(target_image_y420)
                    target_batch_uv.append(target_image_uv420)
                    input_batch_y.append(input_images_y420)
                    input_batch_uv.append(input_images_uv420)


                target_batch_y = np.asarray(target_batch_y).reshape((-1,1024,1024,1))
                target_batch_uv = np.asarray(target_batch_uv).reshape((-1,512,512,2))
                input_batch_y = np.asarray(input_batch_y).reshape((-1,3,1024,1024,1))
                input_batch_uv = np.asarray(input_batch_uv).reshape((-1,3,512,512,2))

                yield target_batch_y, target_batch_uv , input_batch_y, input_batch_uv
                i += 1

    def dnz_anv_trainDataLoader(self,input_evs):
        folder_paths = self.get_json_paths(self.train_path)

        train_dataset = tf.data.Dataset.from_generator(generator=self.dnz_anv_generator,args=[folder_paths,input_evs,self.batch_size], output_types=(tf.float32,tf.float32,
                                                                                                  tf.float32,tf.float32),
                                                       output_shapes=((None,self.crop_size_h,self.crop_size_w,1),(None,self.crop_size_h//2,self.crop_size_w//2,2),
                                                                      (None,self.crop_size_h,self.crop_size_w,3),(None,self.crop_size_h//2,self.crop_size_w//2,6)))
        if(self.strategy):
            train_dataset = train_dataset.batch(self.batch_size * self.strategy.num_replicas_in_sync).prefetch(self.prefetch)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.prefetch(self.prefetch)

        return train_dataset

    def dnz_anv_testDataLoader(self,input_evs,path):
        folder_paths = self.get_json_paths(path)
        folder_paths.sort()
        train_dataset = tf.data.Dataset.from_generator(generator=self.test_dnz_scene_anv_generator,args=[folder_paths,input_evs,1], output_types=(tf.float32,tf.float32,
                                                                                                  tf.float32,tf.float32),
                                                       output_shapes=((None,None,None,3),(None,None,None,6),
                                                                      (None,None,None,1),(None,None,None,2)))
        if(self.strategy):
            train_dataset = train_dataset.batch(self.batch_size * self.strategy.num_replicas_in_sync).prefetch(self.prefetch)
            train_dataset = self.strategy.experimental_distribute_dataset(train_dataset)
        else:
            train_dataset = train_dataset.prefetch(self.prefetch)

        return train_dataset

if __name__ == "__main__":
    '''
    dataloader = Net_DataLoader(train_path="../dataset/Competition_testing_10_10_jpg_only/",valPath="../dataset/Competition_testing_10_10_jpg_only/",crop_size_h=1024,crop_size_w=1024,
                                batch_size=1,normalize=True)

    folder_paths = dataloader.get_json_paths("../dataset/Competition_testing_10_10_jpg_only/")
    train_loader = dataloader.dnz_anv_generator(folder_paths=folder_paths ,input_evs=[(-20,1),(0,1),(4,1)],batch_size=1)

    train_loader = dataloader.anv_trainDataLoader(input_evs=[(-20,1),(0,1),(4,1)])
    #
    for data in train_loader:
        print(data[3].shape)
    pass
    '''

    dataloader = Net_DataLoader(train_path="../rear_dataset/test_rear/",
							valPath="../rear_dataset/test_rear/", crop_size_h=-1,
							crop_size_w=-1,
							batch_size=1, normalize=True)

    test_loader = dataloader.anv_testDataLoader(input_evs=[(-20,1),(0,1),(4,1)])

    i = 1
    for data in test_loader:
        #print(f'type(data[4]) : {type(data[4])}')
        string_value = data[4].numpy()
        #print(f'string_value = {string_value}')
        #print(f'string_value[0] = {string_value[0]}')

        decoded_path = string_value.decode('utf-8')
        #print(f'type(decoded_path) : {type(decoded_path)}')

        print(f'Opened file {i} : {decoded_path}')
        i = i+1


