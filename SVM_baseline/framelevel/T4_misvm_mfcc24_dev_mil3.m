%------------------------------------------------------------------------%

%add_paths;
addpath /user/HS103/qh0001/QH/t4_gmm_mfcc/voicebox;


file_path = '/vol/vssp/msos/qh/data/chime_home/feat_norm_w20_h10/';
Result_fold = cell(1,5);

%--- five folds ---%
for i= 1:5,    
    i    
    %dev data
    crossname = ['/user/HS103/qh0001/QH/t4_gmm_mfcc/fold' num2str(i) '_dev_mat'];
    AA = load(crossname);
    A = AA(:,1:end);
    [r,c] = size(A);
    Name = cell(1,r);
    
    filename = ['fold' num2str(i) '_dev_mil_filelist'];
    fid = fopen(filename, 'r'); 
    
    index = 1;
    while 1
        tline = fgetl(fid);
        Name{index} = tline;
        index = index + 1;
        if ~ischar(tline), break, end
    end
    fclose(fid);
	
	%------- Generate training instance vector --------%
	N = 0;
	Dev_bags = cell(r,1);
	train_targets = zeros(r,c);
	ii = 0;
	r
	index
	for j = 1:index-2,
		%j
		dev_data_name = [file_path Name{j}];
		load(dev_data_name);
		Data = Data';
		[Rd, Cd] = size(Data);
		
        	LA = length(A(j,:));
		A_pos = find(A(j,:)>0);
		L_pos = length(A_pos);
		
		if (L_pos>0)
			Instance_vector = zeros(L_pos,Cd);
               		ii = ii + 1;
	
               		Dev_bags{ii} = Data'; 
               
			train_targets(ii,:) = 2*(A(j,:)>0)-1;
		end
	end
	train_bags = Dev_bags(1:ii);
	train_targets = train_targets(1:ii,:);
	size(train_targets)
	size(train_bags)

	
    crossname_eval = ['/user/HS103/qh0001/QH/t4_gmm_mfcc/fold' num2str(i) '_eval_mat'];
    AA_eval = load(crossname_eval);
    A_eval = AA_eval(:,2:8);
    [r_eval,c_eval] = size(A_eval);
    Name_eval = cell(1,r_eval);
    
    filename = ['fold' num2str(i) '_evaluate_mil_filelist'];
    fid = fopen(filename, 'r'); 
    
    index = 1;
    while 1
        tline = fgetl(fid);
        Name_eval{index} = tline;
        index = index + 1;
        if ~ischar(tline), break, end
    end
    fclose(fid);
    
	r_eval
    ii = 0;
    test_bags = cell(r_eval,1);
    test_targets = zeros(r_eval,c);
    for j = 1:r_eval,
        Mat_file = [file_path Name_eval{j}];
        load(Mat_file);
        
        test_bags{j} = Data;
        test_targets(j,:) = 2*(A_eval(j,:)>0)-1;
          
        
    end

    train_test_name = ['/vol/vssp/msos/qh/mil/mimlfast/temp_framelevel/train_test_fold' num2str(i) 'bags_targets_3'];
    save(train_test_name, 'train_bags', 'train_targets','test_bags','test_targets');
    
end
