
import sys
sys.path.append('your_dir/Hat')
import pickle
import numpy as np
np.random.seed(1515)
import scipy.stats

import keras
from keras.models import load_model

import config_dae as cfg
import prepare_data_1ch_MFC_eval_dae as pp_data
import csv
from Hat.preprocessing import reshape_3d_to_4d
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d
from Hat.preprocessing import sparse_to_categorical, mat_2d_to_3d, reshape_3d_to_4d, mat_concate_multiinmaps4in
from Hat.metrics import prec_recall_fvalue
import cPickle
import eer
#from main_cnn import fe_fd, agg_num, hop, n_hid, fold
np.set_printoptions(threshold=np.nan, linewidth=1000, precision=2, suppress=True)



def reshapeX( X ):
    N = len(X)
    return X.reshape( (N, agg_num*feadim) )

# hyper-params
n_labels = len( cfg.labels )
fe_fd = cfg.dev_fe_mel_fd
#fe_fd_ori = cfg.dev_fe_mel_fd_ori
agg_num = 91        # concatenate frames
hop = 1        # step_len
n_hid = 1000
fold = 0  # can be 0, 1, 2, 3;;;  9 for eva set
feadim=40

# load model
#md = serializations.load( cfg.scrap_fd + '/Md/md20.p' )
md=load_model('your_dir/model.hdf5')

scaler = pp_data.GetScaler( fe_fd, fold )

def recognize():
    
    # do recognize and evaluation
    thres = 0.4     # thres, tune to prec=recall, if smaller, make prec smaller
    n_labels = len( cfg.labels )
    
    gt_roll = []
    pred_roll = []
    result_roll = []
    y_true_binary_c = []
    y_true_file_c = []
    y_true_binary_m = []
    y_true_file_m = []
    y_true_binary_f = []
    y_true_file_f = []
    y_true_binary_v = []
    y_true_file_v = []
    y_true_binary_p = []
    y_true_file_p = []
    y_true_binary_b = []
    y_true_file_b = []
    y_true_binary_o = []
    y_true_file_o = []
    pred_roll_c=[]
    gt_roll_c=[]
    pred_roll_m=[]
    gt_roll_m=[]
    pred_roll_f=[]
    gt_roll_f=[]
    pred_roll_v=[]
    gt_roll_v=[]
    pred_roll_p=[]
    gt_roll_p=[]
    pred_roll_b=[]
    gt_roll_b=[]
    pred_roll_o=[]
    gt_roll_o=[]
    with open( cfg.dev_cv_csv_path, 'rb') as f:
        reader = csv.reader(f)
        lis = list(reader)
    
        # read one line
        line_n=0
        for li in lis:
            na = li[1]
            curr_fold = int(li[2])
            
            if fold==curr_fold:
                line_n=line_n+1
                print line_n
                # get features, tags
                fe_path = fe_fd + '/' + na + '.f'
                info_path = cfg.dev_wav_fd + '/' + na + '.csv'
                #info_path = '/vol/vssp/msos/yx/chime_home/chunk_annotations/annotations' + '/' + na + '.csv'
                #print na
                tags = pp_data.GetTags( info_path )
                #print tags
                y = pp_data.TagsToCategory( tags )
                #print y
                #sys.exit()
                X = cPickle.load( open( fe_path, 'rb' ) )

                X = scaler.transform( X )
                X_1=X[:6,:]
                X_n = np.mean(X_1,axis=0)
                
                # aggregate data
                X3d = mat_2d_to_3d( X, agg_num, hop )
     	        ## reshape 3d to 4d
       	        #X4d = reshape_3d_to_4d( X3d)
                #X4d=np.swapaxes(X4d,2,3) # or np.transpose(x,(1,0,2))  1,0,2 is axis
                #X4d=reshapeX( X)
                X3d= reshapeX(X3d)
                #print X3d.shape
                X_n=np.tile(X_n,(len(X3d),1))
                X_in=np.concatenate((X3d, X_n),axis=1)

                p_y_pred = md.predict( X_in )
                p_y_pred = np.mean( p_y_pred, axis=0 )     # shape:(n_label)
                pred = np.zeros(n_labels)
                pred[ np.where(p_y_pred>thres) ] = 1
                ind=0
                for la in cfg.labels:
                    if la=='S':
                        break
                    elif la=='c':
                        y_true_file_c.append(na)
                        y_true_binary_c.append(y[ind])
                        pred_roll_c.append( pred[ind] )
                        gt_roll_c.append( y[ind] )
                    elif la=='m':
                        y_true_file_m.append(na)
                        y_true_binary_m.append(y[ind])
                        pred_roll_m.append( pred[ind] )
                        gt_roll_m.append( y[ind] )
                    elif la=='f':
                        y_true_file_f.append(na)
                        y_true_binary_f.append(y[ind])
                        pred_roll_f.append( pred[ind] )
                        gt_roll_f.append( y[ind] )
                    elif la=='v':
                        y_true_file_v.append(na)
                        y_true_binary_v.append(y[ind])
                        pred_roll_v.append( pred[ind] )
                        gt_roll_v.append( y[ind] )
                    elif la=='p':
                        y_true_file_p.append(na)
                        y_true_binary_p.append(y[ind])
                        pred_roll_p.append( pred[ind] )
                        gt_roll_p.append( y[ind] )
                    elif la=='b':
                        y_true_file_b.append(na)
                        y_true_binary_b.append(y[ind])
                        pred_roll_b.append( pred[ind] )
                        gt_roll_b.append( y[ind] )
                    elif la=='o':
                        y_true_file_o.append(na)
                        y_true_binary_o.append(y[ind])
                        pred_roll_o.append( pred[ind] )
                        gt_roll_o.append( y[ind] )
                    result=[na,la,p_y_pred[ind]]
                    result_roll.append(result)
                    ind=ind+1
                
                
                pred_roll.append( pred )
                gt_roll.append( y )
    
    pred_roll = np.array( pred_roll )
    gt_roll = np.array( gt_roll )
    pred_roll_c = np.array( pred_roll_c )
    gt_roll_c= np.array( gt_roll_c )
    pred_roll_m = np.array( pred_roll_m )
    gt_roll_m = np.array( gt_roll_m )
    pred_roll_f = np.array( pred_roll_f )
    gt_roll_f = np.array( gt_roll_f )
    pred_roll_v = np.array( pred_roll_v )
    gt_roll_v = np.array( gt_roll_v )
    pred_roll_p = np.array( pred_roll_p )
    gt_roll_p = np.array( gt_roll_p )
    pred_roll_b = np.array( pred_roll_b )
    gt_roll_b = np.array( gt_roll_b )
    pred_roll_o = np.array( pred_roll_o )
    gt_roll_o = np.array( gt_roll_o )
    #write csv for EER computation
    csvfile=file('result.csv','wb')
    writer=csv.writer(csvfile)
    #writer.writerow(['fn','label','score'])
    writer.writerows(result_roll)
    csvfile.close()
    
    # calculate prec, recall, fvalue
    prec, recall, fvalue = prec_recall_fvalue( pred_roll, gt_roll, thres )
    prec_c, recall_c, fvalue_c = prec_recall_fvalue( pred_roll_c, gt_roll_c, thres )
    prec_m, recall_m, fvalue_m = prec_recall_fvalue( pred_roll_m, gt_roll_m, thres )
    prec_f, recall_f, fvalue_f = prec_recall_fvalue( pred_roll_f, gt_roll_f, thres )
    prec_v, recall_v, fvalue_v = prec_recall_fvalue( pred_roll_v, gt_roll_v, thres )
    prec_p, recall_p, fvalue_p = prec_recall_fvalue( pred_roll_p, gt_roll_p, thres )
    prec_b, recall_b, fvalue_b = prec_recall_fvalue( pred_roll_b, gt_roll_b, thres )
    prec_o, recall_o, fvalue_o = prec_recall_fvalue( pred_roll_o, gt_roll_o, thres )
    # EER for each tag : [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
    EER_c=eer.compute_eer('result.csv', 'c', dict(zip(y_true_file_c, y_true_binary_c)))
    EER_m=eer.compute_eer('result.csv', 'm', dict(zip(y_true_file_m, y_true_binary_m)))
    EER_f=eer.compute_eer('result.csv', 'f', dict(zip(y_true_file_f, y_true_binary_f)))
    EER_v=eer.compute_eer('result.csv', 'v', dict(zip(y_true_file_v, y_true_binary_v)))
    EER_p=eer.compute_eer('result.csv', 'p', dict(zip(y_true_file_p, y_true_binary_p)))
    EER_b=eer.compute_eer('result.csv', 'b', dict(zip(y_true_file_b, y_true_binary_b)))
    EER_o=eer.compute_eer('result.csv', 'o', dict(zip(y_true_file_o, y_true_binary_o)))
    EER=(EER_c+EER_m+EER_v+EER_p+EER_f+EER_b+EER_o)/7.0
    prec2=(prec_c+prec_m+prec_f+prec_v+prec_p+prec_b+prec_o)/7.0
    recall2=(recall_c+recall_m+recall_f+recall_v+recall_p+recall_b+recall_o)/7.0
    fvalue2=(fvalue_c+fvalue_m+fvalue_f+fvalue_v+fvalue_p+fvalue_b+fvalue_o)/7.0
    print prec, recall, fvalue
    print prec2, recall2, fvalue2
    print EER_c,EER_m,EER_f,EER_v,EER_p,EER_b,EER_o
    print prec_c,prec_m,prec_f,prec_v,prec_p,prec_b,prec_o
    print recall_c,recall_m,recall_f,recall_v,recall_p,recall_b,recall_o
    print fvalue_c,fvalue_m,fvalue_f,fvalue_v,fvalue_p,fvalue_b,fvalue_o
    print EER

if __name__ == '__main__':
    recognize()
