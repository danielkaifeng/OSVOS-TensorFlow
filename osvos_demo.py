from __future__ import print_function
"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset
os.chdir(root_folder)
from sys import argv


# User defined parameters
seq_name = argv[1]
#seq_name = "car-shadow"
gpu_id = 0
train_model = True
result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS', seq_name)

# Train parameters
parent_path = os.path.join('models', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = os.path.join('models', seq_name)
max_training_iters = 500

# Define Dataset
test_frames = sorted(os.listdir(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name)))
test_imgs = [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
if train_model:
    train_imgs = []
    anno_path = os.path.join('DAVIS', 'Annotations', '480p', seq_name)
    anno_names = os.listdir(anno_path)
    if len(anno_names) > 3:
        select_names = np.random.choice(anno_names, 3)
    else:
        select_names = anno_names[0]

    for name in select_names:
        ID = name.split('.')[0]
        train_imgs += [os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, '%s.jpg' % ID)+' '+
                  os.path.join(anno_path, name)]
    dataset = Dataset(train_imgs, test_imgs, './', data_aug=True)
else:
    dataset = Dataset(None, test_imgs, './')

# Train the network
if train_model:
    # More training parameters
    learning_rate = 1e-8
    save_step = max_training_iters
    side_supervision = 3
    display_step = 10
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join('models', seq_name, seq_name+'.ckpt-'+str(max_training_iters))
        osvos.test(dataset, checkpoint_path, result_path)

# Show results
overlay_color = [255, 0, 0]
transparency = 0.6
plt.ion()
for img_p in test_frames:
    frame_num = img_p.split('.')[0]
    img = np.array(Image.open(os.path.join('DAVIS', 'JPEGImages', '480p', seq_name, img_p)))
    mask = np.array(Image.open(os.path.join(result_path, frame_num+'.png')))
    mask = mask//np.max(mask)
    im_over = np.ndarray(img.shape)
    im_over[:, :, 0] = (1 - mask) * img[:, :, 0] + mask * (overlay_color[0]*transparency + (1-transparency)*img[:, :, 0])
    im_over[:, :, 1] = (1 - mask) * img[:, :, 1] + mask * (overlay_color[1]*transparency + (1-transparency)*img[:, :, 1])
    im_over[:, :, 2] = (1 - mask) * img[:, :, 2] + mask * (overlay_color[2]*transparency + (1-transparency)*img[:, :, 2])

    im_over = im_over.astype(np.uint8)
    img = Image.fromarray(im_over)

    os.makedirs(os.path.join("img_output", seq_name), exist_ok=True)
    img.save(os.path.join("img_output", seq_name, frame_num+'.png'))

    plt.imshow(im_over)
    plt.axis('off')
    plt.show()
    plt.pause(0.01)
    plt.clf()
