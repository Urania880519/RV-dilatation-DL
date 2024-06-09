import tensorflow as tf
import numpy as np
import csv 
import pandas as pd
import os
from PIL import Image
import random
import glob
import matplotlib.pyplot as plt
import itertools
import io
from scipy.ndimage import rotate
# #echo data shape: frame, height, width, channel
import warnings
import cv2
from ast import literal_eval
warnings.filterwarnings("error")



def Get_clean_info():
    all_df= pd.read_csv(r"C:\Users\User\Desktop\echo_RV\clean_data\clean_a4c_0310.csv") 
    all_echo= []
    for i in range(len(all_df)):
        lid= all_df.iloc[i]['lid']
        _set= all_df.iloc[i]['train or val']
        label= all_df.iloc[i]['label']
        echo_tuple= lid, label, _set
        all_echo.append(echo_tuple)
    
    return all_echo

def Get_dirty_info():
    all_df= pd.read_csv(r"C:\Users\User\Desktop\echo_RV\clean_data\dirty_data_1137.csv") 
    all_echo= []
    for i in range(len(all_df)):
        lid= all_df.iloc[i]['lid']
        _set= all_df.iloc[i]['train or val']
        label= all_df.iloc[i]['label']
        echo_tuple= lid, label, _set
        all_echo.append(echo_tuple)
    return all_echo

def Get_views_info():
    
    all_df= pd.read_csv(r"C:\Users\User\Desktop\echo_RV\clean_data\clean_a4c_rvi_sax_0310.csv") 
    all_echo= []
    
    for i in range(len(all_df)):
        lid= all_df.iloc[i]['lid']
        _set= all_df.iloc[i]['train or val']
        label= all_df.iloc[i]['label']
        echo_tuple= lid, label, _set
        all_echo.append(echo_tuple)
    return all_echo



# +
def get_array(data_dir, path, lid):
    if os.path.exists(data_dir):
        echo= np.load(data_dir)["arr_0"]
    else:
        print(data_dir)
        res= [f for f in glob.glob(os.path.join(path, "*.npz")) if lid in f]
        filename= random.choice(res)
        data_dir= os.path.join(path, filename)
        try: 
            echo= np.load(data_dir)["arr_0"]
        except ValueError:
            print(filename)
    if echo.shape[0]!= 20:
        echo= pad_along_axis(echo)
    return echo

def get_echo(lid):
    path= "F:\\Echo\\A4C_data"
    filename= lid+".npz"
    data_dir= os.path.join(path, filename)
    echo= get_array(data_dir, path, lid)
        
    return echo

def get_rvi_echo(lid):
    path= "F:\\Echo\\RV_inflow_data"
    filename= lid+".npz"
    data_dir= os.path.join(path, filename)
    echo= get_array(data_dir, path, lid)
    return echo

def get_sax_echo(lid):
    path= "F:\\Echo\\SApm_data"
    filename= lid+".npz"
    data_dir= os.path.join(path, filename)
    echo= get_array(data_dir, path, lid)
    return echo

# -



def get_echo_seg_aug(lid, seg_array, hflip, vflip, noise, brightness, trans, rotate):
    path= "F:\\Echo\\A4C_data"
    filename= lid+".npz"
    data_dir= os.path.join(path, filename)
    echo= get_array(data_dir, path, lid) #20, 128, 128, 3
    echo= aug_methods(echo).horizontal_flip(hflip)
    echo= aug_methods(echo).vertical_flip(vflip)
    echo= aug_methods(echo).add_noise(noise)
    echo= aug_methods(echo).add_brightness(brightness)
    echo= aug_methods(echo).translation(trans)
    echo= aug_methods(echo).rotation(rotate)
    echo_tensor= tf.convert_to_tensor(echo, dtype= tf.float32)
    
    echo_tensor= tf.concat([echo_tensor, seg_array], axis= -1)
    echo_tensor= tf.squeeze(echo_tensor)
    echo= echo_tensor.numpy()
    return echo



def make_gif(video, path, filename):
    video= video*255
    imgs = [Image.fromarray(np.uint8(img)) for img in video] 
    filepath= os.path.join(path, filename+ ".gif")
    imgs[0].save(filepath, save_all=True, append_images=imgs[1:], duration=50, loop=0)

class aug_methods():
    def __init__(self, video):
        self.video= video
        self.video= self.video.astype(np.float32)
    def horizontal_flip(self, prob):
        choice= np.random.choice([True, False], p=[prob, (1-prob)])
        if choice:
            return np.array([np.fliplr(img) for img in self.video])
        else:
            return self.video
    def vertical_flip(self, prob):
        choice= np.random.choice([True, False], p=[prob, (1-prob)])
        if choice:
            return np.array([np.flipud(img) for img in self.video])
        else:
            return self.video
    def add_noise(self, prob):
        choice= np.random.choice([True, False], p=[prob, (1-prob)])
        if choice:
            channel= self.video.shape[-1]
            noise= np.random.normal(0, 0.1, (128, 128, channel))
            return np.array([(img+ noise) for img in self.video])
        else:
            return self.video
    def add_brightness(self, prob):
        choice= np.random.choice([True, False], p=[prob, (1-prob)])
        if choice:
            videos=[]
            for img in self.video:
                #contrasts = [30/255, 20/255, 10/255] 
                #brightnesses = [50/255, 40/255, 30/255]
                #contrast=random.choice(contrasts)
                #brightness= random.choice(brightnesses)
                contrast= 20/255
                brightness= 50/255
                output = img * (contrast/127 + 1) - contrast + brightness
                videos.append(output)
            return np.array(videos)
        else:
            return self.video
    def translation(self, prob):
        choice= np.random.choice([True, False], p=[prob, (1-prob)])
        if choice:
            videos=[]
            for img in self.video:
                #H1= np.float32([[1, 0, 10],[0, 1, -5]])
                #H2= np.float32([[1, 0, 5],[0, 1, -10]])
                #H3= np.float32([[1, 0, 3],[0, 1, -8]])
                #H= random.choice([H1, H2, H3])
                H= np.float32([[1, 0, 10],[0, 1, -5]])
                res= cv2.warpAffine(img, H, (128, 128))
                videos.append(res)
            return np.array(videos)
        else:
            return self.video
    def rotation(self, prob):
        choice= np.random.choice([True, False], p=[prob, (1-prob)])
        if choice:
            angle= random.choice([5, -5])
            return np.array([rotate(img, angle= angle, reshape=False) for img in self.video])
        else:
            return self.video


def pad_along_axis(array, target_length=20, axis= 0):

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


def plot_confusion_matrix(cm, filename, class_names=['0', '1']):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    #labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    labels= cm


    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)
    return figure


#positive samples
#negative samples
#loss: -loss(p)-omega* loss(n)
def f1_reweight_loss(echo_batch, label, prob):
    bce_sum = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.SUM)
    pos_idx= tf.where(tf.equal(label[:, 0], 1))
    neg_idx= tf.where(tf.equal(label[:, 0], 0))
    pos_prob= tf.gather_nd(prob, pos_idx)
    neg_prob= tf.gather_nd(prob, neg_idx)
    pos_loss_sum= bce_sum(tf.gather_nd(label, pos_idx), pos_prob)
    neg_loss_sum= bce_sum(tf.gather_nd(label, neg_idx), neg_prob)
    batch_size= echo_batch.shape[0]
    p1= tf.reduce_sum(tf.cast(pos_prob[:,0]>=0.5, tf.int32)).numpy()
    p2= tf.reduce_sum(tf.cast(neg_prob[:,0]<0.5, tf.int32)).numpy()
    try:
        neg_weight=  p1 / (batch_size-p2)
    except RuntimeWarning:
        neg_weight= p1
    reweighted_loss=(pos_loss_sum+ neg_weight* neg_loss_sum)/batch_size
    
    return reweighted_loss 

def get_prepared_seg(lid):
    path= "F:\\Echo\\A4C_data"
    data_dir= os.path.join(path, lid+ '.npz' )
    if os.path.exists(data_dir):
        echo= np.load(data_dir)["arr_0"]
        seg_dir= os.path.join("F:\\Echo\\A4C_seg", data_dir[17:])
        seg_array= np.load(seg_dir)["arr_0"]
    else:
        res= [f for f in glob.glob(os.path.join(path, "*.npz")) if lid in f]
        filename= random.choice(res)
        data_dir= os.path.join(path, filename)
        try: 
            echo= np.load(data_dir)["arr_0"]
            seg_dir= os.path.join("F:\\Echo\\A4C_seg", data_dir[17:])
            try:
                seg_array= np.load(seg_dir)["arr_0"]
            except:
                print(seg_dir)
        except ValueError:
            print(filename)
    
    #only use LV and RV seg
    lv= np.expand_dims(seg_array[:, :, :, 1], axis= -1)
    rv= np.expand_dims(seg_array[:, :, :, 3], axis= -1)
    seg_array= np.concatenate([lv, rv], axis= -1)

    echo= tf.convert_to_tensor(echo, dtype= tf.float32)
    if echo.shape[0]!= 20:
        echo= pad_along_axis(echo)
    seg_array= tf.convert_to_tensor(seg_array, dtype= tf.float32)
    if seg_array.shape[0]!= 20:
        seg_array= pad_along_axis(seg_array)
    
    concat_array= tf.concat([echo, seg_array], axis= -1)
    
    return np.array(concat_array)

def get_more_views(lid, rvi= False, sax= False):
    a4c_path= "F:\\Echo\\A4C_data"
    rvi_path= "F:\\Echo\\RV_inflow_data"
    sax_path= "F:\\Echo\\SApm_data"
    df= pd.read_csv(r"C:\Users\User\Desktop\echo_RV\clean_data\clean_a4c_rvi_71.csv")
    
    a4c_lid= df.loc[df['lid']==lid]['a4c_lid']
    a4c_lid= literal_eval(a4c_lid.item())
    if len(a4c_lid)> 1:
        a4c_file= random.choice(a4c_lid)+'.npz'
    else:
        a4c_file= str(a4c_lid[0])+'.npz'
    a4c_echo= np.load(os.path.join(a4c_path, a4c_file))['arr_0']
    a4c_echo= tf.convert_to_tensor(a4c_echo, dtype= tf.float32)
    if a4c_echo.shape[0]!= 20:
        a4c_echo= pad_along_axis(a4c_echo)
    
    if rvi and sax:
        df= pd.read_csv(r"C:\Users\User\Desktop\echo_RV\clean_data\clean_a4c_rvi_sax_71.csv")
        rvi_lid= df.loc[df['lid']==lid]['rvi_lid']
        rvi_lid= literal_eval(rvi_lid.item())
        if len(rvi_lid)> 1:
            rvi_file= random.choice(rvi_lid)+'.npz'
        else:
            rvi_file= str(rvi_lid[0])+'.npz'
        rvi_echo= np.load(os.path.join(rvi_path, rvi_file))['arr_0']
        
        sax_lid= df.loc[df['lid']==lid]['sax_lid']
        sax_lid= literal_eval(sax_lid.item())
        if len(sax_lid)> 1:
            sax_file= random.choice(sax_lid)+'.npz'
        else:
            sax_file= str(sax_lid[0])+'.npz'
        sax_echo= np.load(os.path.join(sax_path, sax_file))['arr_0']
    
        rvi_echo= tf.convert_to_tensor(rvi_echo, dtype= tf.float32)
        if rvi_echo.shape[0]!= 20:
            rvi_echo= pad_along_axis(rvi_echo)
        sax_echo= tf.convert_to_tensor(sax_echo, dtype= tf.float32)
        if sax_echo.shape[0]!= 20:
            sax_echo= pad_along_axis(sax_echo)
        concat_echo= tf.concat([a4c_echo, rvi_echo, sax_echo], axis= 0)
    elif rvi:
        rvi_lid= df.loc[df['lid']==lid]['rvi_lid']
        rvi_lid= literal_eval(rvi_lid.item())
        if len(rvi_lid)> 1:
            rvi_file= random.choice(rvi_lid)+'.npz'
        else:
            rvi_file= str(rvi_lid[0])+'.npz'
        rvi_echo= np.load(os.path.join(rvi_path, rvi_file))['arr_0']
        rvi_echo= tf.convert_to_tensor(rvi_echo, dtype= tf.float32)
        if rvi_echo.shape[0]!= 20:
            rvi_echo= pad_along_axis(rvi_echo)
        concat_echo= tf.concat([a4c_echo, rvi_echo], axis= 0)
    
    return concat_echo