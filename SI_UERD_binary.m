clear all
clc
QF = 75;
CAPA = 40;
MSG_SEED = 518;
SEED = 438;
H = 10;      % constraint height - default is 10 - drives the complexity/quality tradeof
Testing_ERROR = zeros(2,length(CAPA));
dir_jpg = 'D:\guolinjie\suwenkang\data\BOSS10000_mod_Q75';
cover_fea = 'D:\guolinjie\suwenkang\feature\bossbase10000_mod\cover\cover_ccJRM';
% my_ccJRM_mex(dir_jpg,cover_fea,QF);  %提取隐写体的ccJRM值

% dirSource = 'D:\guolinjie\suwenkang\data\BOSS10000_mod\BossBase-1.01-cover';             % input dir
dirSource = 'C:\Users\Administrator\Desktop';
Output_path = 'C:\Users\Administrator\Desktop\test';                              % output dir of all schemes 

for xxx = 1:length(CAPA)
    CAPACITY = CAPA(xxx);    
    if exist(Output_path,'dir'); rmdir(Output_path,'s'); end
    if ~exist(Output_path,'dir'); mkdir(Output_path); end
    stego_fea = ['D:\guolinjie\suwenkang\feature\bossbase10000_mod\SI\SI-UERD_binary_ccJRM_' num2str(CAPACITY)];
    
    files=dir([dirSource '\*.pgm']);
    length(files)
    for w=1:length(files)               
            fprintf('processing   %s: ',files(w).name);
            full_image_file_name=[dirSource '\' files(w).name];
            stego_name =  [Output_path '\' files(w).name];
            stego_name =  [stego_name(1:end-4) '.jpg'];
            
            obj = load('default_gray_jpeg_obj');
            default_gray_jpeg_obj = obj.default_gray_jpeg_obj;
            quan_table=jpeg_qtable(QF);   % quantization table
            default_gray_jpeg_obj.quant_tables{1} = quan_table;
            
            img = imread(full_image_file_name); % load precover
            jpg_dct = bdct(double(img)-128); % dct
            dct_v = quantize(jpg_dct,quan_table); % quantization
            dct_coef = round(dct_v); % rounding
            [img_h, img_w] = size(dct_v);        
            default_gray_jpeg_obj.image_width = img_w;
            default_gray_jpeg_obj.image_height = img_h;
            
            dct_coef2 = dct_coef;
            dct_v2 = dct_v;
            % remove DC coefs;
            dct_coef2(1:8:end,1:8:end) = 0;
            dct_v2(1:8:end,1:8:end) = 0;
            
            nz_index = find(dct_coef < 10000000); % use all dct coefficients
            nz_number = nnz(dct_coef2); % number of non zero ac coefficients

            rand('state',SEED); % Pseudo-random Permutation for cover elements
            r_index = randperm(length(nz_index));
            nz_dct_coef = dct_coef(nz_index(r_index));
            nz_dct_v = dct_v(nz_index(r_index));
            rand('state',MSG_SEED); % Pseudo-random Permutation for message
            
            hidden_message = double(rand(ceil(max(CAPACITY)*nz_number/100+1),1)>0.5);
            
            distortion =  nz_dct_coef - nz_dct_v; % rounding error   nz_dct_coef：舍入；nz_dct_v： 未舍入
            adjust = (distortion <=0)*2-1 - (distortion <=0).*(nz_dct_coef==-1)*2 + (distortion >0).*(nz_dct_coef==1)*2;

            %%% for 1/2-coefficients    avoid zero distortion
            if QF == 75                
                distortion(distortion>0.4999) = distortion(distortion>0.4999) - 0.01;
                distortion(distortion<-0.4999) = distortion(distortion<-0.4999) + 0.01; 
            elseif QF == 95                
                distortion(distortion>0.4999) = distortion(distortion>0.4999) - 0.1;
                distortion(distortion<-0.4999) = distortion(distortion<-0.4999) + 0.1; 
            end        
            
            q_tab = quan_table;
            q_tab(1,1) = 0.5*(q_tab(2,1)+q_tab(1,2));
            q_matrix = repmat(q_tab,[64 64]);
            
            %%% energy of each block
%             fun = @(block_struct) sum(sum(abs(q_tab.*block_struct.data)))*ones(8);
%             J = blockproc(dct_v2,[8 8],fun);
            dct_v2 = im2col(q_matrix.*dct_v2,[8 8],'distinct');
            J2 = sum(abs(dct_v2));
            J = ones(64,1)*J2;
            J = col2im(J,[8 8], [512 512], 'distinct');   
            
%             decide = q_matrix./J; % version 1
            
            pad_size = 8;
            im2 = padarray(J,[pad_size pad_size],'symmetric');  % energies of eight-neighbor blocks
            size2 = 2*pad_size;
            im_l8 = im2(1+pad_size:end-pad_size,1:end-size2);
            im_r8 = im2(1+pad_size:end-pad_size,1+size2:end);
            im_u8 = im2(1:end-size2,1+pad_size:end-pad_size);
            im_d8 = im2(1+size2:end,1+pad_size:end-pad_size);
            im_l88 = im2(1:end-size2,1:end-size2);
            im_r88 = im2(1+size2:end,1+size2:end);
            im_u88 = im2(1:end-size2,1+size2:end);
            im_d88 = im2(1+size2:end,1:end-size2);

            decide = q_matrix./(J+0.25*(im_l8+im_r8+im_u8+im_d8)+0.25*(im_l88+im_r88+im_u88+im_d88)); % version 2
            decide = decide(nz_index(r_index));
            decide = decide/min(decide);
            
            quant_rand=q_matrix(nz_index(r_index)); %量化矩阵的处理
            D = 0.5-abs(distortion);
            PE = (distortion.*adjust<=0).*D+(distortion.*adjust>0)*2;
            costs = decide.*PE;  % capa = 10 ,err = 15
%             costs = decide.*(quant_rand.*PE);
            c_nz_dct_lsb=mod(nz_dct_coef,2);
            %--------------STCs embedding----------%
            [dist, s_nz_dct_lsb] = stc_embed(uint8(c_nz_dct_lsb), uint8(hidden_message), costs, H); % embed message
            diff_message_index = xor(s_nz_dct_lsb,c_nz_dct_lsb); 
            s_nz_dct=nz_dct_coef+ adjust.*diff_message_index;
            s_dct=dct_coef;
            s_dct(nz_index(r_index))=s_nz_dct;
            write2jpeg(default_gray_jpeg_obj,s_dct,stego_name);       % from Guo                                          

    end
    show_s_dif(dct_coef,s_dct);
    message_ext = stc_extract(uint8(mod(s_nz_dct,2)), length(hidden_message), H); % extract message
    if(message_ext==hidden_message)
       disp('All messages are extracted exactly!');
    else
       disp('There must be something wrong here!');
    end

%     my_ccJRM_mex(Output_path,stego_fea,QF);  %提取隐写体的ccJRM值
%     error = my_ensemble(cover_fea,stego_fea);%ensenmble1.0分类   %%有时会出现identifier,需要将当前路径设位ensemple的路径
%     Testing_ERROR(1,xxx) = CAPACITY;
%     Testing_ERROR(2,xxx) = error;
end



% save('Testing_ERROR','Testing_ERROR');

