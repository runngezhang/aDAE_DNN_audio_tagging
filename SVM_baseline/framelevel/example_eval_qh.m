clear

Class = 'bcfmopv';
%for i = 1:5,

        data_name = ['temp_framelevel/train_eval_frame_bags_targets_5'];
	% load the data
	load(data_name);

	train_data = train_bags;
	test_data = test_bags;

	% get the predictions of the evaluation data
	[test_outputs,test_labels]=MIMLfast(train_data,train_targets,test_data);
    
    
        [R,C] = size(test_labels);
    
        filename = ['temp_framelevel/Results_W20_framelevel_mil_mfcc24_train_eval_5'];
    
        fid = fopen(filename, 'w');
        for j = 1:R,
            for k =1:7,
                fprintf(fid,'%d,%s,%6.4f\n',j,Class(k),test_outputs(j,k));
            end
        end
    
        fclose(fid);
%end
