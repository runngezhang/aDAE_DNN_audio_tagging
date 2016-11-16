import sys
sys.path.append('your_dir/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import os
import config_fb40 as cfg
from Hat.preprocessing import reshape_3d_to_4d
import prepare_data_1ch_MFC as pp_data
#from prepare_data import load_data


import keras

from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import h5py
from keras.optimizers import SGD,Adam

# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, fea_dim*agg_num) )




# hyper-params
fe_fd = cfg.dev_fe_mel_fd
#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 7        # concatenate frames
hop = 1            # step_len
n_hid = 100
n_out = len( cfg.labels )
print n_out
fold = 1            # can be 0, 1, 2, 3, 4
fea_dim=40

# prepare data
scaler = pp_data.GetScaler( fe_fd, fold )
#tr_X, tr_y, te_X, te_y = pp_data.GetAllData_NAT( fe_fd, agg_num, hop, fold, scaler,fea_dim)
tr_X, tr_y, te_X, te_y = pp_data.GetAllData( fe_fd, agg_num, hop, fold, scaler,fea_dim)
#tr_X, tr_y, te_X, te_y = pp_data.GetAllData_noMVN( fe_fd, agg_num, hop, fold,fea_dim)



tr_X=np.concatenate((tr_X,te_X),axis=0)

tr_y=tr_X[:,((agg_num-1)/2)*fea_dim:((agg_num-1)/2+1)*fea_dim] ### arrive at 4*fea_dim-1
te_y=te_X[:,((agg_num-1)/2)*fea_dim:((agg_num-1)/2+1)*fea_dim]
#tr_y=tr_X
#te_y=te_X


print tr_X.shape, tr_y.shape
print te_X.shape, te_y.shape




###build model by keras
input_audio=Input(shape=(agg_num*fea_dim,))
encoded = Dropout(0.1)(input_audio)
encoded = Dense(500,activation='relu')(encoded)
encoded = Dense(50,activation='relu')(encoded)

decoded = Dense(500,activation='relu')(encoded)
#decoded = Dense(fea_dim*agg_num,activation='linear')(decoded)
decoded = Dense(fea_dim,activation='linear')(decoded)

autoencoder=Model(input=input_audio,output=decoded)

autoencoder.summary()

sgd = SGD(lr=0.01, decay=0, momentum=0.9)
autoencoder.compile(optimizer=sgd,loss='mse')

dump_fd=cfg.scrap_fd+'/Md/dae_keras_Relu50_1outFr_7inFr_dp0.1_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto') 

autoencoder.fit(tr_X,tr_y,nb_epoch=100,batch_size=100,shuffle=True,validation_data=(te_X,te_y), callbacks=[eachmodel])



