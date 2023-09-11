% Show movie with corresponding raster spikes with frame by frame
clear all;
addpath C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\MatlabInclude
%% Load raster from RES
%eload ('data\crcns-ringach-data\neurodata\ac1\RES_ac1_u006_004.mat');
load ('data\crcns-ringach-data\neurodata\ac1\RES_ac1_u005_007.mat');

RES_id = 50;
res = RES(RES_id);

%% Load corresponding movie
movie_id = RES(RES_id).movie_id;
segment_id = RES(RES_id).segment_id;
movie_dir = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\NeuralRecordings\data\crcns-ringach-data\movie_frames'
movie_frame_fmt = 'movie<movie_id>_<segment_id>.images\movie<movie_id>_<segment_id>_<frame_id>.jpeg';
frame_fmt = strrep(movie_frame_fmt,'<movie_id>',sprintf('%03i',movie_id));
frame_fmt = strrep(frame_fmt,'<segment_id>',sprintf('%03i',segment_id));
%% Show Spike raster and movie simultaneously
% each frame was shown for 33.33 ms (30Hz), sampling of signal at 30KHz->
% 1000 points from S along time samples per frame
[K,N] =size(res.S);
%Colors=distinguishable_colors(N,'w');
Colors = othercolor('Cat_12',N); 
K = K/1000;
K = K-2; % custom adjust
% read image files beforehand and process all data
fprintf('Reading all images');tic(); 
for k = 1:K
    
     im_file =strrep(frame_fmt,'<frame_id>',sprintf('%03i',k-1));
    I(:,:,:,k) =  (imread([movie_dir filesep im_file]));
    Sx(k) = full(sum(sum(res.S([1:1000]+1000*(k-1),:),1)));
end
Ix = diff(I,1,4);
Ix = squeeze(gather(sum(sum(sum(Ix,1),2),3)));
toc();
%% display all data
for k = 1:K
   
    subplot(3,2,[1 3]); imshow((I(:,:,:,k))); title(sprintf('t=%.1f',k/30));
    subplot(3,2,2); hold on; 
    [b,a] = find(res.S([1:1000]+1000*(k-1),:)); 
    h =plot(([1;1].*b'+1000*(k-1))/res.Fs,[a a+1]','-b','MarkerSize',0.1);%,'Color',Colors(a,:));
    for i = 1:length(h); h(i).Color = Colors(a(i),:); end;
    xlabel('time(s)')
    if(k>1)
        subplot(3,2,[5 6]);
        yyaxis left; plot([1:k]/30,conv(Ix(1:k),0.1*ones(1,3),'same'));
        yyaxis right; plot([1:k]/30,conv(Sx(1:k),0.1*ones(1,3),'same')); 
    end
    drawnow;
end