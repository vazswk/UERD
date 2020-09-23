clear all
close all
warning off
clc


QFs = 75;
CAPA = 40;
scheme = 'SI-UERD';
MSG_SEED = 518;
SEED = 438;
H = 10;      % constraint height - default is 10 - drives the complexity/quality tradeof

dirSource = 'C:\Users\Administrator\Desktop';            % input dir
output_dir = 'C:\Users\Administrator\Desktop\test';                             % output dir of all schemes 
for www = 1:length(QFs)
    QF = QFs(www);
for xxx = 1:length(CAPA)
    CAPACITY = CAPA(xxx);    
    Output_path = [output_dir '\Q' num2str(QF) '\' scheme '\' num2str(CAPACITY)];  % output dir of each scheme
    if ~exist(Output_path,'dir'); mkdir(Output_path); end
    
    files=dir([dirSource '\*.pgm']);
    length(files);
    for w=1:length(files)        
        if mod(w,1000)==0 
            disp(w)
        end
        if files(w).isdir==0
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
            [img_h img_w] = size(dct_v);        
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
            %%% for 1/2-coefficients    avoid zero distortion
            if QF == 75                
                distortion(distortion>0.4999) = distortion(distortion>0.4999) - 0.01;
                distortion(distortion<-0.4999) = distortion(distortion<-0.4999) + 0.01; 
            elseif QF == 95                
                distortion(distortion>0.4999) = distortion(distortion>0.4999) - 0.1;
                distortion(distortion<-0.4999) = distortion(distortion<-0.4999) + 0.1; 
            end
            % 调整原则：distortion >0 时，-1的代价小；distortion < 0 时，+1的代价小；
            distortion_m =  distortion - 1; % additional rounding error of -1 embedding
            distortion_p =  distortion + 1; % additional rounding error of +1 embedding
            
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
            
            costs = zeros(3, length(nz_index), 'single'); % for each pixel, assign cost of being changed
            costs(1,:) = decide.*abs(abs(distortion)-abs(distortion_m));       % cost of changing the first cover pixel by -1, 0, +1
            costs(3,:) = decide.*abs(abs(distortion)-abs(distortion_p));       % cost of changing the first cover pixel by -1, 0, +1

            [d stego n_msg_bits l] = stc_pm1_pls_embed(int32(nz_dct_coef)', costs, uint8(hidden_message)', H); % ternary STC embedding 
            % extr_msg = stc_ml_extract(stego, n_msg_bits, H);
            % sum(uint8(hidden_message)'~=extr_msg);      

            em_dct_coef = dct_coef;
            em_dct_coef(nz_index(r_index)) =stego;
        
            default_gray_jpeg_obj.coef_arrays{1} = em_dct_coef;
            jpeg_write(default_gray_jpeg_obj,stego_name);    % generate stego image

        end
    end
end
end

show_s_dif(dct_coef,em_dct_coef);