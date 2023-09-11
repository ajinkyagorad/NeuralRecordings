%% Show spike rate and input video
% goal is to analyze the input output behaviour using linear model

%% Configuration
addpath C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\MatlabInclude
movie_dir = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\NeuralRecordings\data\crcns-ringach-data\movie_frames'
movie_frame_fmt = 'movie<movie_id>_<segment_id>.images\movie<movie_id>_<segment_id>_<frame_id>.jpeg';
nframe_fmt = 'movie<movie_id>_<segment_id>.images\nframes';
video_dir = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\NeuralRecordings\data\crcns-ringach-data\processed_video';
video_fmt = 'NR_<movie_id>_<segment_id>';
%%
%% Load raster from RES
%eload ('data\crcns-ringach-data\neurodata\ac1\RES_ac1_u006_004.mat');
load ('data\crcns-ringach-data\neurodata\ac1\RES_ac1_u005_007.mat');
load ('data\crcns-ringach-data\neurodata\ac1\ac1_u005_007.mat');
%% Loop over all data
for RES_id = 1:4:numel(RES)
    res = RES(RES_id);
    fprintf('Processing %i\r\n',RES_id);
    %%
    
   
    
    %% Load corresponding movie
    movie_id = RES(RES_id).movie_id;
    segment_id = RES(RES_id).segment_id;
    frame_fmt = strrep(movie_frame_fmt,'<movie_id>',sprintf('%03i',movie_id));
    frame_fmt = strrep(frame_fmt,'<segment_id>',sprintf('%03i',segment_id));
    nframe_file = strrep(nframe_fmt,'<movie_id>',sprintf('%03i',movie_id));
    nframe_file = strrep(nframe_file,'<segment_id>',sprintf('%03i',segment_id));
    video_file = strrep(video_fmt,'<movie_id>',sprintf('%03i',movie_id));
    video_file = strrep(video_file,'<segment_id>',sprintf('%03i',segment_id));
    
    % get all frames in that movie
    nframe_file_ID = fopen([movie_dir filesep nframe_file]);
    Nframes = textscan(nframe_file_ID,'%d'); Nframes = Nframes{1};
    fclose(nframe_file_ID);
    %% Show Spike rate and movie simultaneously
    % each frame was shown for 33.33 ms (30Hz), sampling of signal at 30KHz->
    % 1000 points from S along time samples per frame
    [K1,N] =size(res.S);
    K = double(Nframes);
    %K = 900; % custom
    % read image files beforehand and process all data
    fprintf('Reading all images');tic();
    I= uint8([]); Sx = []; Ix = []; Nx = [];
    res.S(1000*K,N) = 0; % append with zeros
    for k = 1:K
        im_file =strrep(frame_fmt,'<frame_id>',sprintf('%03i',k-1));
        I(:,:,:,k) =  (imread([movie_dir filesep im_file]));
        Nx(:,k) = sum(res.S([1:1000]+1000*(k-1),:),1);
    end
    Nx2 = conv2(Nx,ones(1,20)/20,'same'); % smooth over suitable number of frames
    subplot(1,5,[1 2 3 4]);imagesc(Nx2);  h = line([0 0],[1 25],'LineStyle','--','Color',[1 1 1 0.5],'LineWidth',2);
    subplot(1,5,5); for k = 1:K; imagesc(I(:,:,:,k)); h.XData =[k k]; drawnow; end;
    toc();
    drawnow;
    
    %% Display all data & get handles
   
    
end