
AA = zeros(7,5);
SEER = zeros(5,1);
for foldindex = 1:5,
    
    %--- prediction and labels of the devlopment data ---%
    %result_filename = ['temp_instancelevel/Results_dev_instancelevel_mil_mfcc24_fold' num2str(foldindex) '_eval'];
    %label_name  = ['/user/HS103/qh0001/QH/t4_gmm_mfcc/fold' num2str(foldindex) '_evaluate_label.txt'];


    %--- prediction and labels of the evaluation data ---%
    result_filename = 'temp_instancelevel/Results_train_eval_instancelevel_mil_mfcc24';  
    label_name  = 'dt4_eval_label.txt';

	

	ii = 0;
	Sum_eer = 0;
	for label='bcfmopv'
    
            ii = ii + 1;
	    label_assignments = [];
	    label_assignments_filelist = {};
    
	    F = fopen(label_name, 'r');
	    L = fgetl(F);
	    while ischar(L)
        	S = strsplit(L,',');
        	label_assignments_filelist = cat(1,label_assignments_filelist, S{1});
        	label_assignments = cat(1, label_assignments, any(strfind(S{2}, label)));
        	L = fgetl(F);
    	    end
    	    fclose(F);    
    
    	    [EER, pre,rec,f1] = compute_eer(result_filename, label, label_assignments, label_assignments_filelist);
    	    Sum_eer = Sum_eer + EER;

	    %AA(ii,foldindex) = EER;    

    	    fprintf('Label %s: EER %f, Pre %f, Rec %f F1 %f\n', label, EER, pre, rec, f1);
    	end
    	Sum_eer/7
end


