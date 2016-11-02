
import sys
sys.path.append('/user/HS103/yx0001/Downloads/Hat')
from Hat.preprocessing import mat_2d_to_3d, reshape_3d_to_4d, mat_concate_multiinmaps4in
import numpy as np
from scipy import signal
import cPickle
import os
import sys
import matplotlib.pyplot as plt
from scipy import signal
import wavio
import librosa
import config_fb40 as cfg
import csv
import scipy.stats
from sklearn import preprocessing
import scikits.talkbox.features.mfcc as mfcc
import htkmfc


def reshapeX( X ,fea_dim,agg_num):
    N = len(X)
    return X.reshape( (N, agg_num*fea_dim) )
    
### readwav
def readwav( path ):
    Struct = wavio.read( path )
    wav = Struct.data.astype(float) / np.power(2, Struct.sampwidth*8-1)
    fs = Struct.rate
    return wav, fs

# calculate mel feature
def GetMel( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.wav') ]
    extlen=len('16kHz.wav')+1
    #print extlen
    #sys.exit()
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        wav, fs = readwav( path )
        #print wav.shape
        #print fs
        #sys.exit()
        if ( wav.ndim==2 ): 
            wav_m = np.mean( wav, axis=-1 ) # mean
        assert fs==cfg.fs
        ham_win = np.hamming(cfg.win)
        
        #[f_m, t_m, X_m] = signal.spectral.spectrogram( wav_m, window=ham_win, nperseg=cfg.win, noverlap=0, detrend=False, return_onesided=True, mode='magnitude' )
        #X_m = X_m.T
        [ceps, mspec, spec]=mfcc(wav_m, nwin=cfg.win, overlap=cfg.win/2, nfft=512, fs=16000, nceps=24)
        #mspec: Log-spectrum in the mel-domain;; ceps: Mel-cepstrum coefficients
        X_m=ceps
        #print ceps.shape, mspec.shape, spec.shape #(399, 24) (399, 40) (399, 512), why is 512 for spec ?
        #(99, 24) (99, 40) (99, 512)
        #sys.exit()

        # DEBUG. print mel-spectrogram
        #plt.matshow(mspec.T, origin='lower', aspect='auto')
        #plt.show()
        #pause
        
        out_path = fe_fd + '/' + na[0:-extlen] + '.f'
        cPickle.dump( X_m, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )


# calculate mel feature
def GetHTKfea( wav_fd, fe_fd, n_delete ):
    names = [ na for na in os.listdir(wav_fd) if na.endswith('.16kHz.fb40') ]
    extlen=len('16kHz.fb40')+1
    names = sorted(names)
    for na in names:
        print na
        path = wav_fd + '/' + na
        print path
        mfc_reader=htkmfc.open(path,mode='rb')
        X=mfc_reader.getall()
        X = X[:, n_delete:]
	print X.shape # (1291,40)
        
        out_path = fe_fd + '/' + na[0:-extlen] + '.f' #### change na[0:-4]
        cPickle.dump( X, open(out_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL )
          
### format label
# get tags
def GetTags( info_path ):
    with open( info_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    tags = lis[-2][1]
    return tags
            
# tags to categorical, shape: (n_labels)
def TagsToCategory( tags ):
    y = np.zeros( len(cfg.labels) )
    for ch in tags:
        y[ cfg.lb_to_id[ch] ] = 1
    return y


# get chunk data, size: N*agg_num*n_in
def GetAllData_noMVN( fe_fd, agg_num, hop, fold,fea_dim):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        #print info_path
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        # aggregate data
        #print X_l.shape #(nframe=125,ndim=257)
        X3d = mat_2d_to_3d( X, agg_num, hop )
        X3d= reshapeX(X3d,fea_dim, agg_num) 
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ] * len( X3d )
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ] * len( X3d )

    return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ),\
           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist )

###
# get chunk data, size: N*agg_num*n_in
def GetAllData( fe_fd, agg_num, hop, fold , scaler,fea_dim):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        #print info_path
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        X = scaler.transform( X )

        # aggregate data
        #print X_l.shape #(nframe=125,ndim=257)
        X3d = mat_2d_to_3d( X, agg_num, hop )
        X3d= reshapeX(X3d,fea_dim, agg_num)
        #print X3d_l.shape # (nsampelPERutt=10,contextfr=33,ndim=257)
        # reshape 3d to 4d
        #X4d_l = reshape_3d_to_4d( X3d_l)
        #X4d_r = reshape_3d_to_4d( X3d_r)
        #X4d_m = reshape_3d_to_4d( X3d_m)
        #X4d_d = reshape_3d_to_4d( X3d_d)
        # concatenate
        #X4d=mat_concate_multiinmaps4in(X3d_l, X3d_r, X3d_m, X3d_d)
        #print X4d.shape      
        #sys.exit()       
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ] * len( X3d )
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ] * len( X3d )

    return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ),\
           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist )

# get chunk data, size: N*agg_num*n_in
def GetAllData_NAT( fe_fd, agg_num, hop, fold , scaler, fea_dim):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    line_n=0
    for li in lis:
        na = li[1]
        line_n=line_n+1
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        #print info_path
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        #print X.shape
        X = scaler.transform( X )
        #print X.shape
        X_1=X[:6,:]
        #print X_1.shape
        X_n = np.mean(X_1,axis=0)
        #print X_n.shape
        # aggregate data
        #print X_l.shape #(nframe=125,ndim=257)
        X3d = mat_2d_to_3d( X, agg_num, hop )
        X3d= reshapeX(X3d,fea_dim,agg_num)
        #print X3d.shape
        X_n=np.tile(X_n,(len(X3d),1))
        #print X_n.shape
        X_in=np.concatenate((X3d, X_n),axis=1)
        #print X_in.shape # (nsampelPERutt=10,contextfr=33,ndim=257)
        #sys.exit()
        # reshape 3d to 4d
        #X4d_l = reshape_3d_to_4d( X3d_l)
        #X4d_r = reshape_3d_to_4d( X3d_r)
        #X4d_m = reshape_3d_to_4d( X3d_m)
        #X4d_d = reshape_3d_to_4d( X3d_d)
        # concatenate
        #X4d=mat_concate_multiinmaps4in(X3d_l, X3d_r, X3d_m, X3d_d)
        #print X4d.shape      
        #sys.exit()       
        
        if curr_fold==fold:
            te_Xlist.append( X_in )
            te_ylist += [ y ] * len( X_in )
        else:
            tr_Xlist.append( X_in )
            tr_ylist += [ y ] * len( X_in )
    print line_n
    return np.concatenate( tr_Xlist, axis=0 ), np.array( tr_ylist ),\
           np.concatenate( te_Xlist, axis=0 ), np.array( te_ylist )
               
# size: n_songs*n_chunks*agg_num*n_in
def GetSegData( fe_fd, agg_num, hop, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )    
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ]
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ]

    return np.array( tr_Xlist ), np.array( tr_ylist ), \
           np.array( te_Xlist ), np.array( te_ylist )
           
def GetScaler( fe_fd, fold ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist = []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        X = cPickle.load( open( fe_path, 'rb' ) )
        if curr_fold!=fold:
            tr_Xlist.append( X )
            
    Xall = np.concatenate( tr_Xlist, axis=0 )
    scaler = preprocessing.StandardScaler( with_mean=True, with_std=True ).fit( Xall )

    return scaler
           
def GetScalerSegData( fe_fd, agg_num, hop, fold, scaler ):
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
        
    tr_Xlist, tr_ylist = [], []
    te_Xlist, te_ylist = [], []
        
    # read one line
    for li in lis:
        na = li[1]
        curr_fold = int(li[2])
        
        # get features, tags
        fe_path = fe_fd + '/' + na + '.f'
        info_path = cfg.dev_wav_fd + '/' + na + '.csv'
        tags = GetTags( info_path )
        y = TagsToCategory( tags )
        X = cPickle.load( open( fe_path, 'rb' ) )
        if scaler is not None:
            X = scaler.transform( X )
        
        # aggregate data
        X3d = mat_2d_to_3d( X, agg_num, hop )
        
        
        if curr_fold==fold:
            te_Xlist.append( X3d )
            te_ylist += [ y ]
        else:
            tr_Xlist.append( X3d )
            tr_ylist += [ y ]

    return np.array( tr_Xlist ), np.array( tr_ylist ), \
           np.array( te_Xlist ), np.array( te_ylist )
    
###
# create an empty folder
def CreateFolder( fd ):
    if not os.path.exists(fd):
        os.makedirs(fd)
            
if __name__ == "__main__":
    CreateFolder( cfg.scrap_fd + '/Fe' )
    CreateFolder( cfg.scrap_fd + '/Fe/htk_fb40_eval' )
    CreateFolder( cfg.scrap_fd + '/Results' )
    CreateFolder( cfg.scrap_fd + '/Md' )
    #GetMel( cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0 )
    GetHTKfea( cfg.dev_wav_fd, cfg.dev_fe_mel_fd, n_delete=0 )
