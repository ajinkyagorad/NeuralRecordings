%% Generate visual file (video) of raster data in compact video format for HD
% & Show movie with corresponding raster spikes with frame by frame
clear all;
SAVE_VIDEO =1;
SPEAK = 1;
LOGTXT = 0;
%% CONFIG
if(SPEAK)
    tts('Application Started');
end
%%
if(LOGTXT)
    LogFile = fopen('VisualRecording.log','w'); % save log of console
end
%% Movie Parameters
% ref : http://kawahara.ca/matlab-make-a-movievideo/

%%
addpath C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\MatlabInclude
movie_dir = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\NeuralRecordings\data\crcns-ringach-data\movie_frames'
movie_frame_fmt = 'movie<movie_id>_<segment_id>.images\movie<movie_id>_<segment_id>_<frame_id>.jpeg';
nframe_fmt = 'movie<movie_id>_<segment_id>.images\nframes';
video_dir = 'C:\Users\Ajinkya\Desktop\Tt\SEM8\Summer\Neuromorphic\NeuralRecordings\data\crcns-ringach-data\processed_video';
video_fmt = 'NR_<movie_id>_<segment_id>';
%% Load raster from RES
%eload ('data\crcns-ringach-data\neurodata\ac1\RES_ac1_u006_004.mat');
load ('data\crcns-ringach-data\neurodata\ac1\RES_ac1_u005_007.mat');
load ('data\crcns-ringach-data\neurodata\ac1\ac1_u005_007.mat');
%% Loop over all data
for RES_id = 81:numel(RES);
    res = RES(RES_id);
    fprintf('Processing %i\r\n',RES_id);
    %%
    
    %% Figure Handles
    close all;
    fig_handle = figure('name','neuralrecording', 'MenuBar', 'none','ToolBar', 'none');
    set(fig_handle, 'Position', [0, 0, 1280 720]); % window size
    text_subplot = axes('Position',[0 0 0.2 1],'Visible','off');infoText=text(0.05,0.55,' ','FontSize',8);
    input_img = axes('Position',[0.1 0.6 0.2 0.4]);
    spike_nd = axes('Position',[0.35 0.6 0.2 0.4]);
    spike_full = axes('Position',[0.02 0.5 0.95 0.1]);
    activity_full = axes('Position',[0.02 0.37 .95 0.1]);
    spike_live = axes('Position',[0.02 0.15 0.95 0.15]);
    activity_live = axes('Position',[0.02 0.05 .95 0.1]);
    
    % spike_old = axes('Position',[.8 0.6 .15 .35]);
    set(gcf,'Color','k');
    set(findobj(gcf,'type','axes'),'FontName','Consolas','FontSize',8, 'LineWidth', 1,'Color','k','XColor','w','YColor','w');
    set(spike_nd,'Visible','off');
    set(input_img,'Visible','off');
    set(spike_full,'YTick',[],'XTick',[]);set(spike_live,'XTick',[]);
    %set(activity_full);
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
    %% Show Spike raster and movie simultaneously
    % each frame was shown for 33.33 ms (30Hz), sampling of signal at 30KHz->
    % 1000 points from S along time samples per frame
    [K1,N] =size(res.S);
    %Colors=distinguishable_colors(N,'w');
    Colors = othercolor('Cat_12',N);
    K = double(Nframes);
    %K = 900; % custom
    % read image files beforehand and process all data
    fprintf('Reading all images');tic();
    I= uint8([]); Sx = []; Ix = []; Nx = [];
    res.S(1000*K,N) = 0; % append with zeros
    for k = 1:K
        im_file =strrep(frame_fmt,'<frame_id>',sprintf('%03i',k-1));
        I(:,:,:,k) =  (imread([movie_dir filesep im_file]));
        Nx(:,k) = sum(res.S([1:1000]+1000*double(k-1),:),1);
    end
    Nx2 = conv2(Nx,ones(1,20)/20,'same');
    Sx  = full(sum(Nx,1));
    Ix = diff(I,1,4);
    Ix = squeeze(gather(sum(sum(sum(Ix,1),2),3)));
    toc();
    
    %% Display all data & get handles
    [b,a] = find(res.S);
    subplot(spike_live);hold on;
    h =plot([1;1].*b'/res.Fs,[a a+1]','LineWidth',0.01); %,'Color',Colors(a,:));
    for i = 1:length(h); h(i).Color = Colors(a(i),:); end;
    subplot(spike_full);hold on;
    h =plot([1;1].*b'/res.Fs,[a a+1]','LineWidth',0.01); %,'Color',Colors(a,:));
    for i = 1:length(h); h(i).Color = Colors(a(i),:); end;
    xlim([0 30]);
    subplot(activity_live);
    yyaxis left; plot([1:K]/30,conv([0;Ix],0.1*ones(1,3),'same'));
    yyaxis right; plot([1:K]/30,conv(Sx,0.1*ones(1,3),'same'));
    subplot(activity_full);
    yyaxis left; plot([1:K]/30,conv([0;Ix],ones(1,3)/3,'same')); ylabel('Image Diff');
    yyaxis right; plot([1:K]/30,conv(Sx,ones(1,3)/3,'same')); ylabel('Spike rate');
    xlim([0 30]);
     %% slider patch handles
    if(1)
        xp = [-2.5 2.5];
        yp = [0 N+1];
        x_patch = [xp(1) xp(2) xp(2) xp(1)];
        y_patch = [yp(1) yp(1) yp(2) yp(2)];
        
        subplot(spike_full);h_slider1 = patch(x_patch,y_patch,'w','FaceAlpha',0.2); h_center1 = line(xp(1)*[1 1],yp,'LineWidth',2,'Color',[0.8 0.8 0.8 0.5]);
        subplot(activity_full);h_slider2 = patch(x_patch,y_patch,'w','FaceAlpha',0.2); h_center2 = line(xp(1)*[1 1],yp,'LineWidth',2,'Color',[0.8 0.8 0.8 0.5]);
        subplot(activity_live);h_center2_live = line(xp(1)*[1 1],yp,'LineWidth',2,'Color',[0.8 0.8 0.8 0.5]);
        subplot(spike_live);h_center1_live= line(xp(1)*[1 1],yp,'LineWidth',2,'Color',[0.8 0.8 0.8 0.5]);
    end
   
    %% get image handle
    subplot(input_img);img_live = imshow(I(:,:,:,1));
    %%
    if(SAVE_VIDEO)
        writerObj = VideoWriter([video_dir filesep video_file],'MPEG-4');
        open(writerObj);
    end
    %% Create Nd to 2D transform
    T = exp(-sqrt(-1)*([0:N-1]'*2*pi/N));
    subplot(spike_nd); hold on;
    h=plot(2*[T*0 T]',':');    for i = 1:length(h); h(i).Color = Colors(N-i+1,:); end;
    h_nd=scatter(real(T.*Nx2(:,1)),imag(T.*Nx2(:,1)),1000,'CData',Colors,'Marker','.','MarkerFaceAlpha',0.4);
    daspect([1 1 1]);

    %% Loop live
    fprintf(' -- Graphing start ...');tic();
    for k1 = 1:K
        k = double(k1);
        %update video frame
        subplot(input_img);
        img_live.CData = I(:,:,:,k);
        title(sprintf('t=%.1f',k/30),'Color','w');
        % update ndplot
         NDx = T.*Nx2(:,k);
        h_nd.XData  = real(NDx);
        h_nd.YData = imag(NDx);
        % update spike box lim
        subplot(spike_live);        axis([k/30-2.5 k/30+2.5 0 N+1]);
        subplot(activity_live);   axis([k/30-2.5 k/30+2.5 0 N+1]);
        
        xp = [k/30-2.5 k/30+2.5];
        %yp = [0 N+1];
        x_patch = [xp(1) xp(2) xp(2) xp(1)];
        %y_patch = [yp(1) yp(1) yp(2) yp(2)];
        h_slider1.XData  = x_patch;   % h_slider1.YData = y_patch;
        h_slider2.XData  = x_patch;   % h_slider2.YData = y_patch;
        h_center1.XData =k/30*[1 1];
        h_center2.XData =k/30*[1 1];
        h_center1_live.XData = k/30*[1 1];
        h_center2_live.XData = k/30*[1 1];
        drawnow;
        % Add each frame to video
        if(SAVE_VIDEO)
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
        end
    end
    toc();
    %% Close files
    if(SAVE_VIDEO)
        close(writerObj);
    end
    fprintf('Done %i\r\n',RES_id);
    
end
if(LOGTXT)
    close(LogFile);
end

