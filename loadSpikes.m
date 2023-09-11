%% Load spikes from file
clear all;
GETMFR = 0;
%% Load Data
% load from gui
if(0)
    [FileName,PathName] = uigetfile(['RES*.mat'],['Select the *.mat file'],'MultiSelect', 'on');
    if(isequal(FileName,0))
        error(['Specify *_' FILETYPE '.mat File!']);
    elseif(~iscellstr(FileName))
        FileName = cellstr(FileName);
    end
    % load all files of specific format from a directory
else
    directory = '..\data\crcns-ringach-data\neurodata\ac1'
    DirList = dir([directory '\RES*.mat']);
    FileName =  {DirList(:).name}; 
    PathName = DirList.folder;
end
%% Start
mFR = []; % mean Firing Rate
for file_id = 1:length(FileName)
    file = FileName{file_id};
    load([PathName filesep file]);
       
    for i =1:length(RES)
            %[a,b] = find(RES(i).S); plot(a,b,'.'); drawnow;
            % get binned spikes
            s=[];
            for k=1:(30*30-1) % bin into each frame, 30 frames / sec (30 sec long), 30K sampling rate
                s(:,k)=sum(RES(i).S((k-1)*1000+1:k*1000,:))';
            end
            imagesc(conv2(s,ones(1,50)/50)); drawnow;
            %imagesc(conv2(full((RES(i).S')),ones(1,5000)/5000)); drawnow;
            
            if(GETMFR); mFR = [mFR mean(RES(i).S)*RES(i).Fs]; end; % get mean firing rate
    end
    
    
end   
%%
if(GETMFR)
% Get distribution of Mean Firing Rate
[a,b] = hist(mFR,1000);
% fit exponential y = A*exp(lambda*x), x=b,y=a
id_end =max(find(a>5));
fitRes = log(a(1:id_end))/[b(1:id_end); ones(1,id_end)];
lambda = -fitRes(1); A = exp(fitRes(2));
figure('name','ExpFit'); hold on;
plot(b,a);plot(b,A*exp(-lambda*b),'-','LineWidth',2);
legend('Raw Data distribution',sprintf('Exponential Fit ({\\lambda}=%.3f)',lambda));
xlabel('meanFiringRate(Hz)'); ylabel('Frequency(#)');
end