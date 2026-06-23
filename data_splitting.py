import os
import numpy as np
from glob import glob
from tqdm import tqdm
class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']
n_images_per_class = len(os.listdir(f'./{class_names[0]}'))
train_dir = './train'
valid_dir = './valid'
test_dir = './test'
for directory in [train_dir, valid_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
for name in class_names:
    for directory in [train_dir, valid_dir, test_dir]:
        class_path = os.path.join(directory, name)
        if not os.path.exists(class_path):
            os.makedirs(class_path)
all_class_paths = [glob(f'./{name}/*') for name in class_names]
total_size = sum([len(paths) for paths in all_class_paths])
train_ratio = 0.97
valid_ratio = 0.02
test_ratio = 0.01
train_size = int(total_size * train_ratio)
valid_size = int(total_size * valid_ratio)
test_size = int(total_size * test_ratio)
train_images_per_class = int(n_images_per_class * train_ratio)
valid_images_per_class = int(n_images_per_class * valid_ratio)
test_images_per_class = int(n_images_per_class * test_ratio)
print('Total Data Size  :   {}'.format(total_size))
print('Training Size    :   {}'.format(train_size))
print('Validation Size  :   {}'.format(valid_size))
print('Testing Size     :   {}\n'.format(test_size))
for paths in all_class_paths:
    np.random.shuffle(paths)
train_images = [(path, os.path.join(train_dir, path.split('/')[-2], path.split('/')[-1])) for paths in all_class_paths for path in paths[:train_images_per_class]]
valid_images = [(path, os.path.join(valid_dir, path.split('/')[-2], path.split('/')[-1])) for paths in all_class_paths for path in paths[train_images_per_class:train_images_per_class + valid_images_per_class]]
test_images = [(path, os.path.join(test_dir, path.split('/')[-2], path.split('/')[-1])) for paths in all_class_paths for path in paths[train_images_per_class + valid_images_per_class:train_images_per_class + valid_images_per_class + test_images_per_class]]
for images, data_type in [(train_images, 'Training'), (valid_images, 'Validation'), (test_images, 'Testing')]:
    for old_path, new_path in tqdm(images, desc=data_type + ' Data'):
        os.rename(old_path, new_path)
for directory in class_names:
    os.rmdir('./' + directory)
print('ALL DONE!!')
