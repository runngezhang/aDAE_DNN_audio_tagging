%------------------------------------------------------------------------%

addpath ./voicebox;


file_path = 'feat_norm_w20_h10/';
file_eval_path = 'feat_eval_norm_w20_h10/';

Result_fold = cell(1,5);

%for i= 1:5,    
    %i   
    
    %dev data
    crossname = ['dt4_train_classlist'];
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
		
    N = 0;
    Dev_data = cell(1, index);
	
    for j = 1:index-2,
		%j
		dev_d = [file_path Name{j}];
		load(dev_d);
		Dev_data{j} = Data';

    end

    [rr,cc] =size(Dev_data{1})
   
    
    % evaluation data
    crossname_eval = ['dt4_eval_classlist'];
    AA_eval = load(crossname_eval);
    A_eval = AA_eval(:,2:end);
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
    
    Eval_data = cell(1,r_eval);
    for j = 1:r_eval,
         Mat_file = [file_eval_path Name_eval{j}];
         load(Mat_file);
         Eval_data{j} = Data'; 
    end
    
    
    % -------------- instance based training and test features using One-GMM  ------------%
    % training data
    train_bags = cell(r,1);
    Num_mix = 1;
    sprintf('----------- begin  -----------\n') 
    for j = 1:r,
        [m1,v1,w1,g1,f1,pp1,gg1]=gaussmix(Dev_data{j},[],100.001,Num_mix,'hp');
        Feat_m = reshape(m1,1,Num_mix*cc);
        Feat_v = reshape(v1,1,Num_mix*cc);
       
        train_bags{j} = [Feat_m Feat_v];
    end
    
    % test data
    test_bags = cell(r_eval,1);
    for j = 1:r_eval,
        [m1,v1,w1,g1,f1,pp1,gg1]=gaussmix(Eval_data{j},[],100.001,Num_mix,'hp');
        Feat_m = reshape(m1,1,Num_mix*cc);
        Feat_v = reshape(v1,1,Num_mix*cc);
       
        test_bags{j} = [Feat_m Feat_v];
    end
    
    train_target = (2*(AA>0)-1);
    test_target = (2*(A_eval(:,1:c)>0)-1);
    
    Feat_filename = ['feature_train_eval_target_instance_level'];
    
    save(Feat_filename,'train_bags','test_bags','train_target','test_target');
    
%end

