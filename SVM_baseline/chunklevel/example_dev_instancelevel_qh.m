clear;
Class = 'bcfmopv';

for i = 1:5,

        data_name = ['temp_instancelevel/feature_dev_fold' num2str(i) '_instance_level'];    
	% load the data
	load(data_name);

	train_data = train_bags;
	test_data = test_bags;

	% get the predictions 
	[test_outputs,test_labels]=MIMLfast(train_data,train_target,test_data);
    
    
    	[R,C] = size(test_outputs);
    
    	filename = ['temp_instancelevel/Results_dev_instancelevel_mil_mfcc24_fold' num2str(i) '_eval'];
    
    	fid = fopen(filename, 'w');
    	for j = 1:R,
        	for k =1:7,
        	    fprintf(fid,'%d,%s,%6.4f\n',j,Class(k),test_outputs(j,k));
        	end
    	end
    
    	fclose(fid);
end
