

# development
ori_dev_root = 'your_dir/chime_home'

dev_root = 'your_dir/Fe'
dev_wav_fd = dev_root + '/DCASE2016_task4_16k_W20msS10ms_fbank40'

# temporary data folder
scrap_fd = "your_dir/DCASE2016_task4_scrap_1ch_mfcc"
dev_fe_mel_fd = scrap_fd + '/Fe/htk_fb40'
dev_cv_csv_path = ori_dev_root + '/development_chunks_refined_crossval_dcase2016.csv'



# evaluation
'''
eva_csv_path = root + '/evaluation_chunks_refined.csv'
fe_mel_eva_fd = 'Fe_eva/Mel'
'''

labels = [ 'c', 'm', 'f', 'v', 'p', 'b', 'o', 'S' ]
lb_to_id = { lb:id for id, lb in enumerate(labels) }
id_to_lb = { id:lb for id, lb in enumerate(labels) }

fs = 16000.
win = 320.
