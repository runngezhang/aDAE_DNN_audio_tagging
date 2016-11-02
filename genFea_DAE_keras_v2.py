import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import os
import config_fb40 as cfg
from Hat.preprocessing import reshape_3d_to_4d
import prepare_data_1ch_MFC as pp_data
import matplotlib.pyplot as plt
from matplotlib.pyplot import matshow, colorbar,clim,title
from keras.models import load_model
#from prepare_data import load_data
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d

import keras
import shutil
from keras.models import Sequential, Graph, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
import h5py
from keras.optimizers import SGD,Adam
import csv
import cPickle
from keras import backend as K



# resize data for fit into CNN. size: (batch_num*color_maps*height*weight)
def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, fea_dim*agg_num) )



debug=1
# hyper-params
fe_fd = cfg.dev_fe_mel_fd
#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 7        # concatenate frames
hop = 1            # step_len
n_hid = 100
n_out = len( cfg.labels )
print n_out
fold = 1           # can be 0, 1, 2, 3, 4
fea_dim=40

if not debug:
    if os.path.exists('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_1ch_mfcc/Fe/dae_linear200_7outFr_dp0.1'):
        shutil.rmtree('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_1ch_mfcc/Fe/dae_linear200_7outFr_dp0.1')
        print 'rm all done!'
    if not os.path.exists('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_1ch_mfcc/Fe/dae_linear200_7outFr_dp0.1'):
        os.makedirs('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_1ch_mfcc/Fe/dae_linear200_7outFr_dp0.1')
        print 'creat a new dir done!'

dae=load_model('/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_1ch_mfcc/Results/saved_models/DAE_all_7frout_1frout_relu50_DP0.1/dae_keras_relu50_1outFr_dp0.1_weights.99-0.01.hdf5')

scaler = pp_data.GetScaler( fe_fd, fold )

def recognize():
    
    
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        nl=0
        # read one line
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            
            nl=nl+1
            #if fold==curr_fold:           
            if 1==1:
                fe_path = fe_fd + '/' + na + '.f'
                print na
                print nl
                X = cPickle.load( open( fe_path, 'rb' ) )

                X = scaler.transform( X )

                X3d = mat_2d_to_3d( X, agg_num, hop )
                print X3d.shape
                X3d= reshapeX(X3d)
                
                if debug:
                    pred = dae.predict( X3d )


                get_3rd_layer_output = K.function([dae.layers[0].input, K.learning_phase()], [dae.layers[3].output])
                layer_output = get_3rd_layer_output([X3d, 0])[0]
                print layer_output.shape
                out_path = '/vol/vssp/msos/yx/chime_home/DCASE2016_task4_scrap_1ch_mfcc/Fe/dae_linear200_7outFr_dp0.1' + '/' + na + '.f' #### change na[0:-4]
                if not debug:
                    cPickle.dump( layer_output, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
                    print 'write done!'              
                #sys.exit()
                if debug:
                    print layer_output.shape
                    #layer_output1=layer_output[5,:]
                    imgplot=plt.matshow(layer_output[:,:].T, origin='lower', aspect='auto')
                    plt.ylabel('Feature dims')
                    plt.xlabel('Frames')
                    title('Learned new feature: Non-negative representations')               
                    plt.colorbar()
                    plt.show()
                    #sys.pause()
                if debug:
                #if nl==1:
                    fig=plt.figure()
                    ax=fig.add_subplot(2,1,1)
                    nfr=3
                    ax.matshow(pred.T, origin='lower', aspect='auto')
                    #ax.matshow(pred[:,nfr*fea_dim:(nfr+1)*fea_dim].T, origin='lower', aspect='auto')
                    plt.ylabel('Frequency bins')
                    plt.xlabel('Frames')
                    #title('Reconstructed Fbank features')
                    ax=fig.add_subplot(2,1,2)
                    ax.matshow(X3d[:,nfr*fea_dim:(nfr+1)*fea_dim].T, origin='lower', aspect='auto')
                    plt.ylabel('Frequency bins')
                    plt.xlabel('Frames')
                    #title('Original Fbank features')
                    plt.show()
                    pause

if __name__ == '__main__':
    recognize()
