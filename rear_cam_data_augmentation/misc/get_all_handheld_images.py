import os
import shutil as sh
import time


dataset_path = "/home/ubuntu/evaluation_data/evaluation_data"
result_dir = '/home/ubuntu/dataset_augmentation/all_handheld_dataset'

if not os.path.isdir(f'{result_dir}'):
    os.makedirs(f'{result_dir}') 

i=0

for root, dirs, files in os.walk(dataset_path):
        
        #if i>9: break

        folder_name = root.split('/')[-1] 

        if folder_name == 'Handheld':
            for dir in dirs:
                sh.copytree(f'{root}/{dir}', f'{result_dir}/{dir}')
                print('Copying : ', f'{root}/{dir}')     
            i+=1    

print(f'i = {i}')