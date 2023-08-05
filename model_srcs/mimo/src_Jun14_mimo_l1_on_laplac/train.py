import os, time
import tensorflow as tf
import numpy as np
from model import MyModel
from net_dataloader import *
import matplotlib.pyplot as plt



method_name = "Mimo_June_14_L-1_loss_on_laplacian_weighted_sum"
checkpoint_dir = '../checkpoint/%s/' % method_name
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
saved_model_dir='../saved_models/%s/'% method_name

batch_size = 16

global_h = 512
global_w = 512

dataloader = Net_DataLoader(train_path="../rear_dataset/panang_rear_dataset/",
							valPath="../rear_dataset/panang_rear_dataset/", crop_size_h=global_h,
							crop_size_w=global_w,
							batch_size=batch_size, normalize=True)

folder_paths = dataloader.get_json_paths("../rear_dataset/panang_rear_dataset/")

save_freq =1
num_epochs = 180


levels = 3
losses = {}
losses['level_wise_y_loss'] = []
losses['level_wise_uv_loss'] = []
losses['weighted_loss'] = []


def list_add(l1, l2):
    new = []
    for i in range(len(l1)):
        new.append(l1[i] + l2[i])

    return new

        
def plot_weighted_loss(w_losses):
        plt.figure(figsize=(12,8))
        plt.plot(w_losses)
        plt.xlabel("Epochs")
        plt.ylabel('Weighted Loss')
        title = f"Train - Weighted Loss Over Epochs"
        plt.title(title, fontsize=14)
        plt.grid()
        plt.savefig(f'./plots/Weighted_Loss.png')
        print(f'./plots/Weighted_Loss.png')
        plt.clf()


def level_wise_loss():
        
        y_levels =[]
        for i in range(levels+1):
            y_levels.append([temp[i] for temp in losses['level_wise_y_loss']])
             
        
        uv_levels =[]
        for i in range(levels+1):
            uv_levels.append([temp[i] for temp in losses['level_wise_uv_loss']])
        
        plt.figure(figsize=(12,8))
        for i in range(levels+1):
            plt.plot(y_levels[i], label=f'level-{i+1}',marker='*')
        plt.xlabel("Epochs")
        plt.ylabel('Y channel loss')
        title = f"Y channel loss for different Laplacian levels"
        plt.title(title, fontsize=14)
        plt.grid()
        plt.legend()
        plt.savefig(f'./plots/y_loss.png')
        print(f'./plots/y_loss.png')
        plt.clf()

        plt.figure(figsize=(12,8))
        for i in range(levels+1):
            plt.plot(uv_levels[i], label=f'level-{i+1}',marker='*')
        plt.xlabel("Epochs")
        plt.ylabel('UV channel loss')
        title = f"UV channel loss for different Laplacian levels"
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid()
        plt.savefig(f'./plots/uv_loss.png')
        print(f'./plots/uv_loss.png')
        plt.clf()

def train():

    my_model = MyModel()
    my_model.compile()

    for epoch in range(num_epochs):
        
        if epoch % save_freq == 0:
            print("Saving model..")
            my_model.save(saved_model_dir)
            #my_model.save_weights(saved_model_dir)
        
        train_loader = dataloader.anv_trainDataLoader(input_evs=[(0,1),(-20,1),(4,1)])

        weighted_loss_sum = 0

        level_wise_y_loss_sum = [0] * (levels+1)
        level_wise_uv_loss_sum = [0] * (levels+1)


        image_index = 0
        for data in train_loader:
            print(f'Training on image {image_index}')
            image_index += 1
				
            st = time.time()

            # Pre-processing Normalization
            y_temp = data[2]
            uv_temp = data[3]
            uv_temp = np.float32(uv_temp)
            uv_temp_patches = np.minimum(uv_temp, 1.0)
            y_temp = np.float32(y_temp)
            y_temp_patches = np.minimum(y_temp, 1.0)
            y_temp = data[0]
            uv_temp = data[1]
            uv_temp_gt = np.float32(uv_temp)
            uv_gt_scale_1 = np.minimum(uv_temp_gt, 1.0)

            y_temp_gt = np.float32(y_temp)
            y_gt_scale_1 = np.minimum(y_temp_gt, 1.0)

            #print('shapes : ',inp_y_scale_1_ev20.shape, inp_uv_scale_1_ev20.shape, inp_y_scale_1_ev0.shape, inp_uv_scale_1_ev0.shape, inp_y_scale_1_ev4.shape, inp_uv_scale_1_ev4.shape, y_gt_scale_1.shape, uv_gt_scale_1.shape)
            #print('shapes : ',y_temp_patches.shape, uv_temp_patches.shape)
            loss = my_model.train_step(y_temp_patches, uv_temp_patches, y_gt_scale_1, uv_gt_scale_1)
            weighted_loss_sum+=loss['weighted_loss']

            level_wise_y_loss_sum = list_add(level_wise_y_loss_sum, loss['level_wise_y_loss'])
            level_wise_uv_loss_sum = list_add(level_wise_uv_loss_sum, loss['level_wise_uv_loss'])

            print(f'Epoch = {epoch}, Image Index = {image_index}, Current Weighted loss = {loss["weighted_loss"]:.4f}, Average weighted Loss = {weighted_loss_sum/image_index:.4f}, Time take = {(time.time() - st):.4f}sec')
			

        print(f'Epoch {epoch} completed! Total Images = {image_index}, Average weighted Loss = {weighted_loss_sum/image_index:.4f}')

        losses['level_wise_y_loss'].append([i/image_index for i in level_wise_y_loss_sum])
        losses['level_wise_uv_loss'].append([i/image_index for i in level_wise_uv_loss_sum])
        
        losses['weighted_loss'].append(weighted_loss_sum/image_index)

        plot_weighted_loss(losses['weighted_loss'])
        level_wise_loss()


if __name__ == "__main__":
    train()