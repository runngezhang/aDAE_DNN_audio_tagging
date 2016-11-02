import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import os
import config_dae as cfg
from Hat.preprocessing import reshape_3d_to_4d
import prepare_data_1ch_MFC as pp_data
#from prepare_data import load_data


import keras

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
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
agg_num = 91        # concatenate frames
hop = 7            # step_len
n_hid = 1000
n_out = len( cfg.labels )
print n_out
fold = 0         # can be 0, 1, 2, 3, 4
fea_dim=50

# prepare data
scaler = pp_data.GetScaler( fe_fd, fold )
tr_X, tr_y, te_X, te_y = pp_data.GetAllData_NAT( fe_fd, agg_num, hop, fold, scaler, fea_dim)
#tr_X, tr_y, te_X, te_y = pp_data.GetAllData( fe_fd, agg_num, hop, fold, scaler)
#tr_X, tr_y, te_X, te_y = pp_data.GetAllData_noMVN( fe_fd, agg_num, hop, fold)

#tr_X, te_X=reshapeX(tr_X), reshapeX(te_X) 
print tr_X.shape, tr_y.shape
print te_X.shape, te_y.shape

#m_value=np.mean(tr_X,axis=0)
#std_value=np.std(tr_X,axis=0)
#print m_value,std_value
#sys.exit()

###build model by keras
model = Sequential()

#model.add(Flatten(input_shape=(agg_num,fea_dim)))
model.add(Dropout(0.1,input_shape=(agg_num*fea_dim+fea_dim,)))
#model.add(Dropout(0.1,input_shape=(agg_num*fea_dim,)))

model.add(Dense(1000,input_dim=agg_num*fea_dim))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(n_out))
model.add(Activation('sigmoid'))

model.summary()

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='mse', optimizer='adam') ### sth wrong here
sgd = SGD(lr=0.005, decay=0, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#model.compile(loss='mse', optimizer=sgd)

dump_fd=cfg.scrap_fd+'/Md/dnn_daeRelu50out1frNoDP_fold0_fr91_bcCOST_keras_weights.{epoch:02d}-{val_loss:.2f}.hdf5'

eachmodel=ModelCheckpoint(dump_fd,monitor='val_loss',verbose=0,save_best_only=False,save_weights_only=False,mode='auto')      

model.fit(tr_X, tr_y, batch_size=100, nb_epoch=51,
              verbose=1, validation_data=(te_X, te_y), callbacks=[eachmodel]) #, callbacks=[best_model])
#score = model.evaluate(te_X, te_y, show_accuracy=True, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

