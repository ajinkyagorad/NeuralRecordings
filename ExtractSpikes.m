%% Load neural data & extract raster plot for the channels
clear all;
[FileName,PathName] = uigetfile(['.mat'],['Select the *.mat file'],'MultiSelect', 'on');
if(isequal(FileName,0))
    error(['Specify *_' FILETYPE '.mat File!']);
elseif(~iscellstr(FileName))
    FileName = cellstr(FileName);
end
%%
for file_id = 1:length(FileName)
    %file = 'ac1_u004_001'
    file = FileName{file_id};
    load([PathName filesep file]);
    RES = [];
    NRes = length(pepANA.listOfResults);
    fprintf('Processing file %s\r\n',file);
    for RES_id = 1:NRes
        
        fs = 30E3;
        res = pepANA.listOfResults{RES_id};
        repeat = res.repeat{1}; % take only first occurance 
        NCh = length(repeat.data);
        T = []; CH = [];
        
        for CH_id = 1:NCh
            data = repeat.data{CH_id}; % take from 1 electrode
            t = data{1}; q = data{2}; % take its time and data
            
            if(0)
                figure('name','svd analysis');
                [u,s,v] = svd(double(q));
                plot(v(:,1),v(:,2),'.','MarkerSize',10); hold on; plot(0,0,'r.','MarkerSize',25);
            end
            %% separate points using kmeans
            if(length(q(1,:))<2)
                warning(sprintf('No data at locn FileID(%i),ResID(%i),ChID(%i)',file_id,RES_id,CH_id));
            else
                idx = kmeans(double(q'),2);
                id1 = (idx==1); id2 = (idx==2);
                q1 = q(:,idx==1); q2 = q(:,idx==2);close
                mq1 = mean(q1'); mq2 = mean(q2');
                % separate noisy and actual signal
                if(max(mq1)-min(mq1)>max(mq2)-min(mq2)) % check peak height difference in mean over ensemble
                    ids = id1; idn = id2;
                else
                    ids = id2; idn = id1;
                end
                if(0)
                    %figure('name','clutering results');hold on;
                    %errorbar(mq1,std(double(q1')),'g');
                    %errorbar(mq2,std(double(q2')),'b');
                    %figure(1);hold on;
                    plot(mean(q(:,ids)')); hold on; plot(mean(q(:,idn)')); hold off; drawnow; pause(0.2);
                end
                %h = plot(q)ckl; for i = 1:N; h(i).Color = double([idx(i)==1 0 idx(i)==2]); end;
                T = [T t(ids)]; % append spike times (save master)
                CH = [CH CH_id*ones(1,length(find(ids)))]; % append channel number (save master)
            end
        end
        %% Not correct waveforms
        RES(RES_id).S = sparse(ceil(T*fs),CH,logical(1));
        RES(RES_id).Fs = fs;
        RES(RES_id).elec_list = pepANA.elec_list;
        if(strcmp(res.symbols{1},'movie_id'))
        RES(RES_id).movie_id = res.values{1};
        RES(RES_id).segment_id = res.values{2};
        elseif(strcmp(res.symbols{1},'rseed'))
        RES(RES_id).rseed = res.values{1};
        else
            warning('No classified res.value property\r\n');
        end
        RES(RES_id).tzero = repeat.tzero;
        RES(RES_id).timeTag = repeat.timeTag;
    end
    %%
    save([PathName 'RES_' file ],'RES');
    if(0)
        for i =1:length(RES)
            [a,b] = find(RES(i).S); plot(a,b,'.'); drawnow;
        end
    end
end
fprintf('DONE\r\n');