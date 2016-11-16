
AA = zeros(7,5);
SEER = zeros(5,1);
for foldindex = 1:5,
    

    %--- predictions and labels of the development set
    %result_filename = ['temp_framelevel/Results_W20_framelevel_mil_mfcc24_fold' num2str(foldindex) '_eval_3'];
    %label_name  = ['/user/HS103/qh0001/QH/t4_gmm_mfcc/fold' num2str(foldindex) '_evaluate_label.txt'];

    %--- predictions and labels of the evaluation set
    result_filename = 'temp_framelevel/Results_W20_framelevel_mil_mfcc24_train_eval_5';
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


