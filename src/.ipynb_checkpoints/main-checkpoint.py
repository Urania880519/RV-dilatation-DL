import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
from utils import get_echo, aug_methods, plot_confusion_matrix, get_echo_seg, get_prepared_seg,  get_echo_seg_aug, f1_reweight_loss
from utils import get_more_views, Get_views_info
import logging
import random
from resnet3d import Model
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import TensorBoard
import os
import sklearn.metrics as metrics
import datetime
from keras.models import model_from_json
from keras.models import load_model
from Seg_model import seg_model, SegLayer
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
logging.basicConfig()
logger= logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--batch-size", help='batch_size for training', default=32, type=int)
    parser.add_argument("--num-workers", help='number of workers for dataloading', default=1, type=int)
    parser.add_argument("--epochs", help='number of epochs', default=1, type=int)
    parser.add_argument("--lr", help='learning rate', default= 1e-4)
    parser.add_argument("--use-gpu", help='use gpu for training or not', default=False, action='store_true')
    parser.add_argument("--resume", help='resume the training steps from last time', default=False,  action='store_true')
    parser.add_argument("--balance", help='ratio of balancing(neg/pos)', default=False,  action='store_true')
    parser.add_argument("--ratio-int", help='ratio of balancing(neg/pos)', default=0, type=int )
    parser.add_argument("--aug-bool", help='do data augmentation', default=False, action='store_true')
    parser.add_argument("--model-path", help='path to save model', default=('./model/'), type=str)
    parser.add_argument("--hflip-prob", help='probability of horizontal flip', default=0, type=float)
    parser.add_argument("--vflip-prob", help='probability of vertical flip', default=0, type=float)
    parser.add_argument("--addnoise-prob", help='probability of adding up Gaussian noise', default=0, type=float)
    parser.add_argument("--brightness-prob", help='probability of adding brightness', default=0, type=float)
    parser.add_argument("--trans-prob", help='probability of tranlating', default=0, type=float)
    parser.add_argument("--rotate-prob", help='probability of rotating', default=0, type=float)
    parser.add_argument("--log-dir", help='tensdorboard log', default=('./runs/{}').format(datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')))
    parser.add_argument("--seg", help='adding segmentation model', default=False,  action='store_true')
    parser.add_argument("--f1-loss", help='reweight the loss', default=False,  action='store_true')
    parser.add_argument("--datetime", default= datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S'))
    parser.add_argument("--rvi", help='add RV inflow', default=False,  action='store_true')
    parser.add_argument("--sax", help='add SAXpm inflow', default=False,  action='store_true')
    #add some more arguments about augmentation
    args= parser.parse_args()
    return args

def get_info(label_csv, fd, train= True):
    label_df= pd.read_csv(label_csv)
    #train_df= label_df.loc[label_df['train or val']!=0]
    train_df= label_df
    if train!=True:
        wanted_data= train_df.loc[train_df['train or val']==fd]
    else:
        wanted_data= train_df.loc[train_df['train or val']!=fd]
    
    
    wanted_data.reset_index(drop= True, inplace= True)
    all_echo=[]
    for i in range(len(wanted_data)):
        lid= wanted_data.iloc[i]['lid']
        _set= wanted_data.iloc[i]['train or val']
        label= wanted_data.iloc[i]['label']
        echo_tuple= lid, label, _set
        all_echo.append(echo_tuple)
    return all_echo

class EchoDataGen(tf.keras.utils.Sequence):
    def __init__(self,
                 all_echo, # all echo_tuple= lid, label, _set
                 balance,
                 ratio_int,
                 aug_bool,
                 batch_size,
                 rvi= False,
                 sax= False,
                 seg= False,
                 hflip_prob= None, 
                 vflip_prob= None, 
                 addnoise_prob= None, 
                 brightness_prob= None, 
                 trans_prob= None,
                 rotate_prob= None,
                 ):
        self.rvi= rvi
        self.sax= sax
        self.seg= seg
        self.batch_size= batch_size
        self.aug_bool= aug_bool
        self.balance= balance
        self.ratio_int= ratio_int
        self.hflip= hflip_prob
        self.vflip= vflip_prob
        self.noise= addnoise_prob
        self.brightness= brightness_prob
        self.trans= trans_prob
        self.rotate= rotate_prob

        self.data_list= all_echo
        self.pos_list= list(filter(lambda x:x[1]==1, self.data_list))
        self.neg_list= list(filter(lambda x:x[1]==0, self.data_list))
        
    def do_augmentation(self, x):
        aug= aug_methods(x).horizontal_flip(self.hflip)
        aug= aug_methods(aug).vertical_flip(self.vflip)
        aug= aug_methods(aug).add_noise(self.noise)
        aug= aug_methods(aug).add_brightness(self.brightness)
        aug= aug_methods(aug).translation(self.trans)
        aug= aug_methods(aug).rotation(self.rotate)
        #if only applying horizontal_flip, set self.vflip to 0
        #if aug_bool=False, self.hflip and self.vflip==0
        return aug
    def __len__(self):
        if self.balance:
            #return 6000//self.batch_size
            return 2*len(self.neg_list)//self.batch_size
        else:
            return len(self.pos_list+ self.neg_list)//self.batch_size
    def pos_weight(self):
        #count only using the training set
        pos_count= len(self.pos_list)
        neg_count= len(self.neg_list)
        pos_weight= (1/pos_count)* ((pos_count+neg_count)/2)
        return pos_weight
    
    def random_choice(self):
        #random choosing some data for training and validating
       
        self.neg_list= random.sample(self.neg_list, k= 1500)
        self.data_list= self.neg_list+ self.pos_list
        logger.info("{!r}: random sampling {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.data_list),
            "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.balance else 'unbalanced'
            ))
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.balance:
            self.pos_indexes= np.arange(len(self.pos_list))
            self.neg_indexes= np.arange(len(self.neg_list))
            np.random.shuffle(self.pos_indexes)
            np.random.shuffle(self.neg_indexes)
        else:
            self.data_indexes= np.arange(len(self.data_list))
            np.random.shuffle(self.data_indexes)
    def from_tuple(self, _tuple):
        label= _tuple[1] # 0 or 1
        label= np.array([label, not label]) #one hot

        if not self.seg:
            if self.rvi or self.sax:
                echo_array= get_more_views(_tuple[0], self.rvi, self.sax)
            else:
                echo_array= get_echo(_tuple[0])

            if self.aug_bool:
                echo_array= self.do_augmentation(echo_array)
            else:
                self.hflip= 0
                self.vflip= 0
                self.noise= 0
                self.brightness= 0
                self.trans= 0
                self.rotate= 0
                
        else: 
            #not yet adding choices if other views with seg are required
            if self.aug_bool and (_tuple[1]==1): #only augment the positive samples
                echo_array= get_echo_seg_aug(_tuple[0], self.hflip, self.vflip, self.noise, self.brightness, self.trans, self.rotate)
                print(type(echo_array))
            else:
                echo_array= get_prepared_seg(_tuple[0])
                self.hflip= 0
                self.vflip= 0
                self.noise= 0
                self.brightness= 0
                self.trans= 0
        
        echo_tensor= tf.convert_to_tensor(echo_array, dtype= tf.float32)
        label_tensor= tf.convert_to_tensor(label, dtype=tf.float32)

        return echo_tensor, label_tensor, _tuple[0]
    
    def __getitem__(self, index):
        #index= index of batches
        start_index= (index)* self.batch_size
        end_index= (index+1)* self.batch_size
        batch_list=[]
        if self.balance:
            for idx in range(start_index, end_index):
                pos_ndx = idx // (self.ratio_int + 1)
                if idx % (self.ratio_int + 1):
                    neg_ndx = idx - 1 - pos_ndx
                    neg_ndx %= len(self.neg_list)
                    info_tup = self.neg_list[self.neg_indexes[neg_ndx]]
                    batch_list.append(info_tup)
                else:
                    pos_ndx %= len(self.pos_list)
                    info_tup = self.pos_list[self.pos_indexes[pos_ndx]]
                    batch_list.append(info_tup)
        else:
            for i in range(start_index, end_index):
                batch_list.append(self.data_list[self.data_indexes[i]])
        
        with ThreadPoolExecutor(max_workers= 16) as executor:
            futures=[]
            echo_tensors=[]
            label_tensors=[]
            lids=[]
            for i in range(len(batch_list)):
                future = executor.submit(self.from_tuple, batch_list[i]) #tf tensors
                futures.append(future)
            for future in as_completed(futures):
                echo_tensor, label_tensor, lid= future.result()
                echo_tensors.append(echo_tensor)
                label_tensors.append(label_tensor)
                lids.append(lid)
            
              
            echo_batch= tf.stack(echo_tensors)
            label_batch= tf.stack(label_tensors)

        return echo_batch, label_batch, lids

args= parse_args()

class training_loop:
    def __init__(self, height, width, frames):
        self.optimizer= tf.keras.optimizers.Adam(learning_rate= args.lr)
        
    def lr_scheduler(self, epoch, lr):
        if epoch < 10:
            lr= lr
        else:
            lr= lr * tf.math.exp(-0.1)
        
        return lr
    def train_step(self, train_gen, pos_weight, lr):
        weight_dict={}
        metrics_list= []
        self.optimizer= tf.keras.optimizers.Adam(learning_rate= lr)
        for batch_number, (echo_batch, label, lids) in enumerate(train_gen):
            metrics= np.zeros((3, args.batch_size))
            with tf.GradientTape() as tape:
                out, prob= self.model(echo_batch, training=True)
               
                #loss_array= self.loss_func(label[:, 0], out[:, 0]).numpy()
                #loss_mean= self.loss_mean_func(label[:, 0], out[:, 0])
                
                if args.f1_loss:
                    loss_mean= f1_reweight_loss(echo_batch, label, prob)
                else:
                    loss_array= tf.nn.weighted_cross_entropy_with_logits(label[:, 0], out[:, 0], pos_weight)
                    loss_mean= tf.reduce_mean(loss_array)
                metrics[0]= label[:, 0]
                metrics[1]= prob[:, 0]
                metrics[2]= loss_mean
                metrics_list.append(metrics)      
                    
            gradients= tape.gradient(loss_mean, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            trn_metrics= np.concatenate(metrics_list, axis=1)
        for layer in self.model.layers:
            weight_dict[str(layer.name)]= layer.get_weights()
        return trn_metrics, weight_dict
    
    def val_step(self, val_gen, pos_weight):
        def cal(echo_batch, label, pos_weight):
            metrics= np.zeros((3, args.batch_size))
            out, prob= self.model(echo_batch)
            if args.f1_loss:
                loss_mean= f1_reweight_loss(echo_batch, label, prob)
            else:
                loss_array= tf.nn.weighted_cross_entropy_with_logits(label[:, 0], out[:, 0], pos_weight)
                loss_mean= tf.reduce_mean(loss_array)
            metrics[0]= label[:, 0]
            metrics[1]= prob[:, 0]
            metrics[2]= loss_mean
            return metrics

        with ThreadPoolExecutor(max_workers=16) as executor:
            metrics_list=[]
            for batch_number, (echo_batch, label, lids) in enumerate(val_gen):
                future= executor.submit(cal, echo_batch, label, pos_weight)
                metrics_list.append(future.result())         
        val_metrics= np.concatenate(metrics_list, axis=1)
        return val_metrics
    
    def summary_writer(self, mode):
        writer= tf.summary.create_file_writer(args.log_dir+'_'+ mode)
        return writer

    def log_metrics(self, mode, epoch_index, metrics_np, threshold=0.5):
        #label, prob, loss
        logger.info("epoch {} {}".format(epoch_index, type(self).__name__,))
        pred_mask=  metrics_np[1] > threshold
        try:
            roc_auc_score= metrics.roc_auc_score(metrics_np[0,:], metrics_np[1,:])
        except ValueError:
            roc_auc_score= 0
        accuracy= metrics.accuracy_score(metrics_np[0], pred_mask)*100
        f1_score= metrics.f1_score(metrics_np[0], pred_mask, zero_division= np.nan)
        precision= metrics.precision_score(metrics_np[0], pred_mask, zero_division= np.nan)
        recall= metrics.recall_score(metrics_np[0], pred_mask, zero_division= np.nan)
        loss= metrics_np[2,:].mean()
        cm = metrics.confusion_matrix(metrics_np[0], pred_mask, normalize= 'true')
        
        if epoch_index== args.epochs:
            if mode== 'val':
                filename= ('./confusion_matrix/{}').format(args.datetime)
                figure = plot_confusion_matrix(cm, filename+'.png')
                

        logger.info(
                ("epoch{} {:8} {:.4f} loss, "
                     + "{:-5.1f}% correct, "
                     + "{:.4f} precision, "
                     + "{:.4f} recall, "
                     + "{:.4f} f1_score, "
                     + "{:.4f} roc_auc_score"
                ).format(
                    epoch_index,
                    mode,
                    loss,
                    accuracy,
                    precision,
                    recall,
                    f1_score,
                    roc_auc_score,
                )
            )
        
        #writer scalar
        with self.summary_writer(mode).as_default() as writer:
            tf.summary.scalar("accuracy", accuracy, step= epoch_index)
            tf.summary.scalar("loss", loss, step= epoch_index)
            tf.summary.scalar("f1 score", f1_score, step= epoch_index)
            tf.summary.scalar("precision", precision, step= epoch_index)
            tf.summary.scalar("recall", recall, step= epoch_index)
            tf.summary.scalar("roc_auc_score", roc_auc_score, step= epoch_index)
            writer.flush()

    def main(self, args):
        logger.info("Starting {}, {}".format(type(self).__name__, args))
        
        label_csv= r"C:\Users\User\Desktop\echo_RV\clean_data\five_fold\clean_a4c_0310.csv"
        if args.sax:
            label_csv= r"C:\Users\User\Desktop\echo_RV\clean_data\five_fold\clean_a4c_rvi_sax_0310.csv"
        elif args.rvi:
            label_csv= r"C:\Users\User\Desktop\echo_RV\clean_data\five_fold\clean_a4c_rvi_0310.csv"
        folds= range(0,5)
        for s, fd in enumerate(folds):
            train_folds= list(folds[:])
            train_folds.remove(fd)
            train_gen= EchoDataGen(
                get_info(label_csv, fd),
                balance= args.balance,
                ratio_int= args.ratio_int,
                aug_bool= args.aug_bool,
                batch_size= args.batch_size,
                rvi= args.rvi,
                sax= args.sax,
                seg= args.seg,
                hflip_prob= args.hflip_prob,
                vflip_prob= args.vflip_prob,
                addnoise_prob= args.addnoise_prob, 
                brightness_prob= args.brightness_prob, 
                trans_prob= args.trans_prob,
                rotate_prob= args.rotate_prob,
            )
            pos_weight= train_gen.pos_weight()
            val_gen= EchoDataGen(
                get_info(label_csv, fd, train= False),
                balance= False,
                ratio_int= None,
                aug_bool= False,
                batch_size= args.batch_size,
                rvi= args.rvi,
                sax= args.sax,
                seg= args.seg,
            )
            #reinitialize the model
            self.model= Model(height, width, frames)
            
            #if args.resume:
            #    if args.seg:
            #        self.model.build(input_shape= (args.batch_size, frames, height, width, 5))
            #        self.model.load_weights(r"./model/A4C_SEG_resnet3d_0.h5")

            #    else: 
            #        self.model.build(input_shape= (args.batch_size, frames, height, width, 3))
            #        self.model.load_weights(r"./model/A4C_SEG_resnet3d_0.h5")
            lr= args.lr
            for epoch in range(1, args.epochs+1):
                logger.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(
                    epoch,
                    args.epochs,
                    len(train_gen),
                    len(val_gen),
                    args.batch_size,
                    len(list(filter(lambda x:x.device_type=='GPU', device_lib.list_local_devices()))) if (len(tf.config.list_physical_devices('GPU'))>0) else 1
                ))
                train_gen.on_epoch_end()
                val_gen.on_epoch_end()
                lr= self.lr_scheduler(epoch, lr)
                trn_metrics, weight_dict= self.train_step(train_gen, pos_weight, lr)
                self.log_metrics('trn', epoch, trn_metrics)
                val_metrics= self.val_step(val_gen, pos_weight)
            
                self.log_metrics('val', epoch, val_metrics)
                if epoch== args.epochs:
                    
                    model_name= "{}_resnet3d".format(args.datetime)
                    if not os.path.exists(args.model_path):
                        os.makedirs(args.model_path)
                
                    self.model.save_weights(os.path.join(args.model_path, model_name+ f'_{fd}.h5'))
                
                
                    f = open(os.path.join(args.model_path, model_name+ f'_{fd}.pickle'), 'wb')
                    pickle.dump(weight_dict, f)
                    f.close()   

if __name__ == "__main__":
    height= 128
    width= 128
    frames= 20
    training_loop(height= height, width=width, frames=frames).main(args= args)

