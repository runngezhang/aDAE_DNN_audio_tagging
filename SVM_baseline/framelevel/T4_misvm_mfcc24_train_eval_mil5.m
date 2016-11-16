%------------------------------------------------------------------------%

%add_paths;
addpath ./voicebox;


file_path = 'feat_norm_w20_h10/';
file_eval_path = 'feat_eval_norm_w20_h10/';



%--- five folds ---%
%for i= 1:5,    
%    i    
    %dev data
    crossname = 'dt4_train_classlist';
    AA = load(crossname);
    A = AA(:,1:end);
    [r,c] = size(A);
    Name = cell(1,r);
    
    filename = ['dt4_train_matlist'];
    fid = fopen(filename, 'r'); 
    
    index = 1;
    while 1
        tline = fgetl(fid);
        Name{index} = tline;
        index = index + 1;
        if ~ischar(tline), break, end
    end
    fclose(fid);
	
	%------- Generate training  vector --------%
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

	
	
	%---  Generate evaluate vector ---%
	
    crossname_eval = 'dt4_eval_classlist';
    AA_eval = load(crossname_eval);
    A_eval = AA_eval(:,2:8);
    [r_eval,c_eval] = size(A_eval);
    Name_eval = cell(1,r_eval);
    
    filename = ['dt4_eval_matlist'];
    fid = fopen(filename, 'r'); 
    
    index = 1;
    while 1
        tline = fgetl(fid);
        Name_eval{index} = tline;
        index = index + 1;
        if ~ischar(tline), break, end
    end
    fclose(fid);

    test_bags = cell(r_eval,1);
    test_targets = zeros(r_eval,c);
    for j = 1:r_eval,
        Mat_file = [file_eval_path Name_eval{j}];
        load(Mat_file);
	
	    test_targets = 2*(A_eval(j,:)>0)-1;	
	    test_bags{j} = Data; 
    end
	
	
    train_test_name = ['temp_framelevel/train_eval_frame_bags_targets_5'];
    save(train_test_name, 'train_bags', 'train_targets','test_bags','test_targets');
    
%end
