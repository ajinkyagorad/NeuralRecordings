
%% LSM
% GeneralLSM code function
% can run in both direct and indirect method
%% Add details of script here
%% Updated : added weight update record spike statistics
% activity of the network
% total spikes of readout neuron
% spikes of readout neuron when there is weight change
% 2:32 AM ISTT{[1:5 14],'AccuracyTestAvg'}
%% added modified weight update (2nd sept,18) 1:09AM IST
%% added probabilistic wt update (3rd sept,18) 8:32PM IST
%% changed sample_i to sample_ii (13th sept,18)
%% store all network parameters in single object file
versionLSM='v1.1';
PARAM1 = ws2struct();
fprintf('RESULT/LOG will be stored as %s.mat/LOG_*.txt',savefilename);
%% CONFIG
%addpath(['..' filesep 'MatlabInclude' filesep ]);
fprintf('Running version %s\r\n',versionLSM);
%%
if(LOGTXT)
    LogFile = fopen(['LOG_' savefilename '.txt'],'w'); % save log of console
end
%% Movie Parameters
% ref : http://kawahara.ca/matlab-make-a-movievideo/
if(SAVE_VIDEO)
    writerObj = VideoWriter([savefilename '.mp4']);
    open(writerObj);
end
%% Load Data
% Put AudioSpikes.m code here or any data generationn code
addpath ../SpeechDataset/
addpath ../SpeechDataset/AuditoryToolbox
%Training Data
dataset=load(DATASET); % load corresponding structure
field_dataset=fieldnames(dataset);field_dataset=field_dataset{1};
if(isfield(dataset.(field_dataset),'class'));class='class'; info='info';else class='digit';info='subject';end
tempDATAfile = 'tempDatasetProc.mat'; % Change this if current data needs to be saved
if(LOAD_EXISTING_INPUT)
    %load 0-9digits-10x16x4_new.mat
    load(tempDATAfile);
else
    rng(RNG.rng.Seed); % for reproducibility
    DATA = struct([]);
    
    %fs = digitdata(1).Fs;  % assume 8KHz
    dt = 1/out_fs;  
    tic();
    parfor i = 1:numel(dataset.(field_dataset))
        fprintf('%i',dataset.(field_dataset)(i).(class));
        %[filts,freqs] = DesignLyonFilters(fs);
        s = LyonPassiveEar(dataset.(field_dataset)(i).sig,fs,df,earQ,stepfactor,differ,agcf,tauf);
        s = min(s,max_AGC);
        s = s/max_AGC;
        S = BSA(s,BSAfilterFac*BSA_filter(out_fs,BSAtau,BSAtau2));
        DATA(i).type = dataset.(field_dataset)(i).(class);
        DATA(i).spk = str2num(dataset.(field_dataset)(i).(info){2}{12,2}(2));
        DATA(i).S =sparse(logical(S));
        DATA(i).info = dataset.(field_dataset)(i).(info);
        %subplot(211);imagesc(s);title(DATA(i).type);
        %subplot(212);imagesc(S); drawnow;
    end
    toc();
    save(tempDATAfile,'DATA');
end
%dgt =[ DATA.type]; % sort accordingly
%[~,id] = sort(dgt);
%DATA = DATA(id);
%DATA = reshape(DATA,samplesPerClass,Nout); % shape accordingly
%DATA = DATA';
%
% if(0)
% % show dataset speaker vs 
% spk=[DATA.spk];type=[DATA.type];
% SMP=zeros(max(spk),max(type)+1);
% end
fprintf('\r\n... Data Preprocessed \r\n');
%% PARAM2
% General parameters
Nstride = numel(DATA)/Nfold;  % Training
Ke = 1/(tauv1e-tauv2e); Ke1 = 1/tauv1e;
Ki = 1/(tauv1i-tauv2i); Ki1 = 1/tauv1i;
alphaV = (1-dt/tauV); alphaC = (1-dt/tauC);
alphav1e = (1-dt/tauv1e);alphav2e = (1-dt/tauv2e);
alphav1i = (1-dt/tauv1i);alphav2i = (1-dt/tauv2i);
RefracPeriodSamples = floor(RefracPeriod/dt);
%nRP = ceil(RF/dt);
%resting potential = 0, refractory period  = 2ms
% END PARAM2
%% No parameters should be declared after this
if(LOGTXT) % LOG all the parameters at start of file
    s_template = {'%Input Parameters (Auditory Toolbox)'
        'fs = <fs>; out_fs = <out_fs>; df = <df>; earQ = <earQ>; stepfactor =<stepfactor>; differ=<differ>; agcf=<agcf>; tauf=<tauf>; max_AGC = <max_AGC>; BSAtau =<BSAtau>; BSAtau2 = <BSAtau2>;BSAfilterFac = <BSAfilterFac>;  appendS = <appendS>;'
        '% Neuron Parameters'
        'tauV = <tauV>; tauC = <tauC>; tauv1e = <tauv1e>; tauv2e = <tauv2e>; tauv1i = <tauv1i>; tauv2i = <tauv2i>; RefracPeriod = <RefracPeriod>; Vth = <Vth>;'
        '% Reservoir Parameters'
        'resSize = <resSize>; Wres = <Wres>; r0 = <r0>; Kres = <Kres>; f_inhibit = <f_inhibit>; GinMag = <GinMag>;  InputFanout  = <InputFanout>;'
        '% Classifier Parameters'
        'Nout = <Nout>; dW0 = <dW0>; Wlim = <Wlim>; Cth = <Cth>; DeltaC= <DeltaC>; dC =<dC>; Iinf = <Iinf>'
        'dt = <dt>; '};
    s = s_template;
    s = strrep(s,'<fs>',num2str(fs));           s = strrep(s,'<out_fs>',num2str(out_fs));           s = strrep(s,'<df>',num2str(df));
    s = strrep(s,'<earQ>',num2str(earQ));       s = strrep(s,'<stepfactor>',num2str(stepfactor));   s = strrep(s,'<differ>',num2str(differ));
    s = strrep(s,'<agcf>',num2str(agcf));       s = strrep(s,'<tauf>',num2str(tauf));               s = strrep(s,'<max_AGC>',num2str(max_AGC));
    s = strrep(s,'<BSAtau>',num2str(BSAtau));   s = strrep(s,'<BSAtau2>',num2str(earQ));            s = strrep(s,'<BSAfilterFac>',num2str(BSAfilterFac));
    s = strrep(s,'<appendS>',num2str(appendS));
    
    s = strrep(s,'<tauV>',num2str(tauV));       s = strrep(s,'<tauv1e>',num2str(tauv1e));           s = strrep(s,'<tauv2e>',num2str(tauv2e));
    s = strrep(s,'<tauv1i>',num2str(tauv1i));   s = strrep(s,'<tauv2i>',num2str(tauv2i));           s = strrep(s,'<RefracPeriod>',num2str(RefracPeriod));
    s = strrep(s,'<Vth>',num2str(Vth));
    
    s = strrep(s,'<resSize>',sprintf('[%i %i %i]', resSize));    s = strrep(s,'<Wres>',sprintf('[%f %f; %f %f]', Wres));
    s = strrep(s,'<r0>',num2str(r0));                           s = strrep(s,'<Kres>',sprintf('[%f %f; %f %f]', Kres));
    s = strrep(s,'<f_inhibit>',num2str(f_inhibit));             s = strrep(s,'<GinMag>',num2str(GinMag));
    s = strrep(s,'<InputFanout>',num2str(InputFanout));
    
    s = strrep(s,'<Nout>',num2str(Nout));   s = strrep(s,'<dW0>',num2str(dW0));         s = strrep(s,'<Wlim>',num2str(Wlim));
    s = strrep(s,'<Cth>',num2str(Cth));     s = strrep(s,'<DeltaC>',num2str(DeltaC));   s = strrep(s,'<dC>',num2str(dC));
    s = strrep(s,'<Iinf>',num2str(Iinf));       s = strrep(s,'<dt>',num2str(dt));
    for i =1:length(s)
        fprintf(LogFile,'--%s\r\n',s{i});
    end
    fprintf('--%s\r\n--\r\n',s);
end
%%  Reservoir
rng(RNG.rng.Seed) % reset it for producing reservoir
[X,Xn,T,G,R,E] = createNetworkM(resSize,Wres,r0,Kres,f_inhibit,1E-3);
%[X,Xn,G,R,E] = get2DneighbourCrossConnectedNetwork(resSize,Wres,f_inhibit);
Nres = length(E); Nsyn = length(X);

alphav1=zeros(Nres,1); alphav1(E>0)=alphav1e; alphav1(E<0)=alphav1i;
alphav2=zeros(Nres,1); alphav2(E>0)=alphav2e; alphav2(E<0)=alphav2i;
K0 = zeros(Nres,1); K0(E>0) =Ke; K0(E<0) =Ki;
K01 = zeros(Nres,1); K01(E>0) =Ke1; K01(E<0) =Ki1;
G = sparse(X,Xn,G,Nres,Nres);
Gres=G;

% Input
Nin = length(DATA(1).S(:,1)); % Input Neurons giving out spike trains
Ain = double(sparse(applied_current_matrix(Nin,Nres,InputFanout)));
Gin = GinMag*Ain.*sign(rand(size(Ain))-0.5);
NsynIn = length(find(Ain~=0));

PARAM2 = struct('G',G,'Nin',Nin,'Ain',Ain,'Gin',Gin,'E',E);

% append gap in data
if(appendS>0)
    for sample_i = 1:numel(DATA)
        DATA(sample_i).S = logical([DATA(sample_i).S zeros(Nin,appendS)]);
    end
    
end
%% RESERVOIR
 SpikesRes.List = {};
    SpikesRes.Info = {};
if(RESERVOIR)
    SpikesRes.List = {'RES','RES0'};
    SpikesRes.Info = {'Reservoir','Reservoir-NoProp'};
    SpikesRes.G{1} = G; SpikesRes.G{2}=G*1E-10;
    SpikesRes.E{1} = E; SpikesRes.E{2} = E; % redundant in this scenario yet
    fprintf('Processing  reservoir spikes..\r\n');
    NinR =Nin;
    for iRes = 1:numel(SpikesRes.List)
        tic();
        spikeResVar = SpikesRes.List{iRes};
        G = SpikesRes.G{iRes};
        E = SpikesRes.E{iRes};
        if(~exist('spikeResVar','var')); spikeResVar='RES'; end;
        DATA(1).(spikeResVar)=[];
        parfor sample_i = 1:numel(DATA)
            
            sample = DATA(sample_i);
            jmax = length(sample.S(1,:)); % time length of input
            sample_label = sample.type+1;
            fprintf('%i',sample.type);
            spikedLog = [];
            k = 0;
            % Res
            I = zeros(Nres,2); % synaptic delay = 1;
            RP = -Inf*ones(Nres,1);
            V = zeros(Nres,1);
            v = zeros(Nres,SYNAPSE_ORDER);
            %Input
            vin = zeros(NinR,SYNAPSE_ORDER);
            for j = 1:jmax
                k = k+1;
                if(SYNAPSE_ORDER==0)
                    Iapp = sample.S(:,j)/dt;
                elseif(SYNAPSE_ORDER==1)
                    vin(:,1) = alphav1e*vin(:,1)+sample.S(:,j);
                    Iapp = Ke1*vin(:,1);
                else
                    vin(:,1) = alphav1e*vin(:,1)+sample.S(:,j);
                    vin(:,2) = alphav2e*vin(:,2)+sample.S(:,j);
                    Iapp = Ke*(vin(:,1)-vin(:,2));
                end
                
                if(RESERVOIR)
                    V = alphaV*V+G'*I(:,mod(k-1,2)+1)*dt+Gin*Iapp*dt;
                    V(k-RP<=RefracPeriodSamples) = 0;
                    spiked = V>Vth ; V(spiked)=0; V(V<0) = 0;
                    RP(spiked) = k;
                    if(SYNAPSE_ORDER==0)
                        I(:,mod(k,2)+1) = spiked/dt;
                    elseif(SYNAPSE_ORDER==1)
                        v(:,1) = alphav1.*v(:,1)+spiked;
                        I(:,mod(k,2)+1) = K01.*v(:,1);
                        
                    else
                        v(:,1) = alphav1.*v(:,1)+spiked;
                        v(:,2) = alphav2.*v(:,2)+spiked;
                        I(:,mod(k,2)+1) = K0.*(v(:,1)-v(:,2));
                    end
                    spikedLog(:,end+1) = spiked;
                end
            end
            %figure(2); subplot(5,2,sample_label); [a,b] = find(spikedLog); plot(b,a,'.'); drawnow;%xlabel(num2str(sample_label));drawnow;
            DATA(sample_i).(spikeResVar) = (logical(spikedLog));
            %figure(2); subplot(211); [a,b] = find(sample.S);plot(b,a,'.','MarkerSize',0.1);subplot(212); [a,b] = find(spikedLog);plot(b,a,'.','MarkerSize',0.1);title(num2str(sample_label));drawnow;
        end
        fprintf('\r\nReservoir Raster Mapping Done\r\n');
        toc();
    end
end
%% Save DATA file
save([savefilename '_DATA.mat'],'DATA');
%% Get Distribution
if(GET_DIST)
    
    [inDist,inSort,resDist,resSort]=showSpikesDist(DATA,SpikesRes.List,SpikesRes.Info,1); drawnow;
    DIST = struct('inDist',inDist,'inSort',inSort,'resDist',resDist,'resSort',resSort,'SpikesRes',SpikesRes);
    PARAM2.DIST = DIST;
end
%% CLASSIFIER
fprintf('Classifying reservoir spikes (Training and simultaneous testing of training and testing data)\r\n');
k = 0;
tic();
RESULT = struct([]);

if(RESERVOIR==0)
    NinC = Nin;
    spikeClassify = 'S';
    ActiveNeuronID=flip(inSort);
else
    NinC = Nres; % input to the classifier
    spikeClassify = 'RES';
    ActiveNeuronID=flip(resSort(:,1));
end
ConnectedNeuronMask = zeros(NinC,1);

if(REVERSE_PRUNE)
    RevActiveNeuronID=flip(ActiveNeuronID);
    ConnectedNeuronMask(RevActiveNeuronID(1:min(NinC,ceil(fracResIn*NinC))))   =1;
else
ConnectedNeuronMask(ActiveNeuronID(1:min(NinC,ceil(fracResIn*NinC))))   =1;
PARAM2.ConnectedNeuronMask = ConnectedNeuronMask;
end
parfor kfold = 1:Nfold
%for kfold=Nfold:-1:1
    tic();
    
    
    % Input
    I = zeros(NinC,2); % synaptic delay = 1;
    RP = -Inf*ones(NinC,1);
    V = zeros(NinC,1);
    v = zeros(NinC,SYNAPSE_ORDER);
    
    % Output
    Vout = zeros(Nout,1);
    VoutR = zeros(Nout,1);
    Cout = zeros(Nout,1);
    Iteach = zeros(Nout,1);
    desired = zeros(Nout,1);
    Winit = zeros(Nout,NinC);
    Prob = zeros(Nout,NinC);
    W = Winit;
    WR = W; % for testing old data
    RPOut = -Inf*ones(Nout,1);
    %
    spikedLog=[];
    
    k =0;
    testOn = (kfold-1)*Nstride+(1:Nstride);
    trainOn  = setdiff(1:numel(DATA),testOn);
    typetrn = [DATA(trainOn).type]+1;
    typetest = [DATA(testOn).type]+1;
    [~,sortIdTrain] = sort(typetrn);
    [~,sortIdTest] = sort(typetest);
    Imerit = zeros(Nout,length([trainOn testOn]));
    robustnessMetric=[]; Rmetric = [];
    CoutLog = []; SpikeOutLog = []; IteachLog=[];
    wmean = []; accuracyTrain=[]; numCorrectTrain=[];
    accuracyTest=[]; numCorrectTest=[];
    numWchanges = 0; readoutSpikesTraining =0; readoutSpikesTestingAll=0; % total weight changes and spikes in each epoch
    
    for epoch = 1:Nepochs
        % script train & test epoch
        s = sprintf('Training epoch:%i (@t=%fs)',epoch,toc);
        fprintf('%s\t',s);if(LOGTXT);fprintf(LogFile,s);end;
        Imerit(:)=0;
        sample_ii=0;
        spikeSampleCountTrain = zeros(Nout,0); % each epoch, in sequence of trainOn 
        fprintf('\t Training|Testing...\t');
        for sample_i = [trainOn testOn] 
            sample_ii=sample_ii+1;
            sample = DATA(sample_i);
            jmax = length(sample.(spikeClassify)(1,:)); % time lenggth of input
            
            sample_label = sample.type+1;
            if(sample_ii==length(trainOn)+1); fprintf('\t Testing:');end
            fprintf('%i',sample.type);
            spikeSampleCountTrain(:,end+1)=zeros(Nout,1);% for reading testing data
            if(RESET_VAR_PER_SAMPLE)   VoutR(:)=0; Vout(:)=0;v(:)=0;Cout(:)=0; end
            for j = 1:jmax
                k = k+1;
                
                if(SYNAPSE_ORDER==0)
                    Iapp = sample.(spikeClassify)(:,j)/dt;
                elseif(SYNAPSE_ORDER==1)
                    v(:,1) = alphav1e.*v(:,1)+sample.(spikeClassify)(:,j);
                    %vin(:,2) = alphav2e*vin(:,2)+sample.S(:,j);
                    Iapp = Ke1.*v(:,1);
                else
                    v(:,1) = alphav1e*v(:,1)+sample.(spikeClassify)(:,j);
                    v(:,2) = alphav2e*v(:,2)+sample.(spikeClassify)(:,j);
                    Iapp = Ke*(v(:,1)-v(:,2));
                end
                
                
                if(MODIFIED_WT_UPDATE<2)
                    if(sample_ii<=length(trainOn))
                        Iteach(:) =0;
                        Iteach(Cout>Cth-dC) = -Iinf;
                        if(Cout(sample_label)<Cth+dC); Iteach(sample_label) = Iinf;
                        else Iteach(sample_label) = 0;
                        end
                    end
                end
                
                Imerit(:,sample_ii) = Imerit(:,sample_ii)+W*Iapp;
                Vout = alphaV*Vout+W*Iapp*dt+Iteach*dt;
                VoutR = alphaV*VoutR+WR*Iapp*dt;
                outputSpiked = Vout>Vth;Vout(Vout<0)=0;  Vout(outputSpiked) = 0;
                outputSpikedR = VoutR>Vth;VoutR(VoutR<0)=0;  VoutR(outputSpikedR) = 0;
                readoutSpikesTestingAll = readoutSpikesTestingAll+ numel(find(outputSpikedR~=0));%LOG
                
                if(sample_ii<=length(trainOn))
                    Cout = alphaC*Cout+Cfactor*outputSpiked;
                    signC = ((Cout-Cth)>0 & (Cout-Cth)<DeltaC)-((Cout-Cth)<0 & (Cout-Cth)>-DeltaC);
                    sample.spike = sample.(spikeClassify)(:,j).*ConnectedNeuronMask;
                    if(MODIFIED_WT_UPDATE==0)
                        deltaW = dW0*(signC*sample.spike');
                    elseif(MODIFIED_WT_UPDATE==1)
                        Cw = abs(signC).*((Cth-Cout)+signC.*DeltaC)/(DeltaC-dC);
                        deltaW = dW0*(Cw*sample.spike');
                    else
                        desired(:) = -1; desired(sample_label) = 1;
                        Cw = ((Cout<Cth+DeltaC).*(desired>0)-(Cout>Cth-DeltaC).*(desired<0));
                        deltaW = dW0*(Cw*sample.spike');
                    end
                    
                    if(PROBABILISTIC_WT_UPDATE)
                        Prob(:)= probN; Prob(sample_label,:) = probP;
                        deltaW=deltaW.*(rand(size(deltaW))<Prob);
                    end
                    W_old=W;
                    W = W+deltaW;
                    W(W>Wlim) = Wlim;
                    W(W<-Wlim) = -Wlim;
                    numWchanges = numWchanges+numel(find(abs(W_old-W)~=0)); % LOG
                    readoutSpikesTraining = readoutSpikesTraining+numel(find(outputSpiked~=0)); %LOG
                    if(j==jmax)
                        wmean(:,end+1) = sum(W,2)/NinC;
                    end
                end
                spikeSampleCountTrain(:,end) = spikeSampleCountTrain(:,end)+outputSpikedR;
                %CoutLog(:,k) = Cout;
                %spikeOutLog(:,k) = outputSpiked;
                %IteachLog(:,k) = Iteach;
            end
        end
        %% END of Trainind and testing of Train & Test data
        
        Imerit = Imerit/jmax;
        % Get robustness metric
        for i = 1:Nout
            Rmetric(:,i) = sum(Imerit(:,typetrn==i),2);
        end
        DRmetric = Inf*ones(Nout,Nout);
        for i =1:Nout; for j = 1:i-1
                metricDirn = zeros(Nout,1);metricDirn(i)=1;metricDirn(j)=-1;
                DRmetric(i,j) = sum((Rmetric(:,i)-Rmetric(:,j)).*metricDirn);
            end; end;
        DRmetricMin = min(DRmetric,[],2);
        robustnessMetric(end+1) = sum(DRmetricMin(2:end));
        
        %if(Nout==3)
        %    set(h_Imerit,'XData',Imerit(1,:),'YData',Imerit(2,:),'ZData',Imerit(3,:));
        %end
        
        WR = W;
        [M,recognized] = max(spikeSampleCountTrain);
        Y = spikeSampleCountTrain./repmat(M,Nout,1);
        %recognizedTrain = recognized(trainOn);
        %recognizedTest = recognized(testOn);
        
        % TRAIN accuracy
        %numCorrectTrain = length(find((typetrn-recognizedTrain)==0));
        %accuracyTrain = numCorrectTrain*100/length(typetrn);
        Mn = sparse(1:length(trainOn),1+[DATA(trainOn).type],1,length(trainOn),Nout); % confusion matrix finder;
        MTest = sparse(1:length(testOn),1+[DATA(testOn).type],1,length(testOn),Nout); % confusion matrix finder;
        M  = blkdiag(Mn,MTest);
        Y = Y./repmat(max(Y),Nout,1); Y(Y~=1)=0;
        %check for correct classification only, remove no classification
        %and misclassification
        misClassifiedSamples = find(sum(Y,1)~=1);
        Y(:,misClassifiedSamples) = 0;
        CM = (Y*M); % confusion matrix
        
        accuracyTrain = 100*trace(CM(:,1:Nout))/length(trainOn);
        numCorrectTrain = trace(CM(:,1:Nout));
        accuracyTest = 100*trace(CM(:,Nout+(1:Nout)))/length(testOn);
        numCorrectTest = trace(CM(:,Nout+(1:Nout)));
        % TEST
        %numCorrectTest = length(find((typetest-recognizedTest)==0));
        %accuracyTest = numCorrectTest(end)*100/length(typetest);
        
        % **** LOG results *****
        RESULT(kfold).accTest(epoch) = accuracyTest;
        RESULT(kfold).accTrain(epoch) = accuracyTrain;
        RESULT(kfold).numWchanges(epoch) = numWchanges;
        RESULT(kfold).readoutSpikesTraining(epoch) = readoutSpikesTraining;
        RESULT(kfold).readoutSpikesTestingAll(epoch) = readoutSpikesTestingAll;
        RESULT(kfold).W(:,:,epoch)= W;
        RESULT(kfold).CM(:,:,epoch) = CM;
        RESULT(kfold).spikeSCT(epoch).S=spikeSampleCountTrain;
        RESULT(kfold).spikeSCT(epoch).Y=Y;
        RESULT(kfold).spikeSCT(epoch).YS = spikeSampleCountTrain*M;
        s=sprintf('\r\n Accuracy : kFold(%i) Epoch(%i) Test %2.2f (%i/%i) Train:%2.2f (%i/%i) #dW : (%i) \t',kfold,epoch,accuracyTest,numCorrectTest,length(testOn),accuracyTrain,numCorrectTrain,length(trainOn),numWchanges);
        fprintf(s);
        if(LOGTXT);fprintf(LogFile,s);end
        
        %subplot(4,2,[1 3]);plot([0:length(accuracyTrain)-1],accuracyTrain); hold on; plot([0:length(accuracyTrain)-1],accuracyTest); hold off;title('Accuracy'); legend({'Train','Test'},'Location','southeast');
        %xlabel('epochs');ylabel('% accuracy'); ylim([0 100]);
        %subplot(4,2,[2 4]);plot(robustnessMetric);title('RobustnessMetric'); xlabel('Epochs'); ylabel('arb');
        %subplot(4,2,5); imagesc(spikeSampleCountTrain(:,[sortIdTrain sortIdTest]));    xlabel('sample');ylabel('class');title('SpikeDensity Training');
        %subplot(4,2,7); imagesc(CM'); xlabel('Input'); ylabel('Class'); title('CM');
        %subplot(4,2,[6 8]);plot(wmean'); xlabel('end of sample');title('Mean Input Wts');
        %subplot(1,4,kfold);plot([0:length(accuracyTrain)-1],accuracyTrain); hold on; plot([0:length(accuracyTrain)-1],accuracyTest); hold off;title('Accuracy'); legend({'Train','Test'},'Location','southeast');
        %drawnow;
        
        if(SAVE_VIDEO)
            frame = getframe(gcf);
            writeVideo(writerObj,frame);
        end
    end
    % Save Nfold Results for each
    
    RESULT(kfold).M=M;
    RESULT(kfold).CM = CM;
    RESULT(kfold).trainOn =  trainOn;
    RESULT(kfold).testOn= testOn;
end

%% Save params
save([savefilename '.mat'],'PARAM1','PARAM2','RESULT');
fprintf('DATA file saved\r\n');
%% Show Results
accTest = reshape([RESULT(:).accTest],Nepochs,Nfold);
accTrain = reshape([RESULT(:).accTrain],Nepochs,Nfold);
figure('name','Results: Accuracy');
%subplot(121); % Accuracy
plot(mean(accTest,2),'LineWidth',3);hold on;
plot(mean(accTrain,2),'LineWidth',3);
plot(accTest,'--');
plot(accTrain,'-');
set(gcf,'Color','w');
legend({'Test','Train'},'Location','southeast');
xlabel('Epochs');
ylabel('Accuracy(%)')
title([num2str(Nfold) '-fold Testing']);
ylim([0 100]);
%subplot(122); % Error
figure('name','Results: Error');
plot(100-mean(accTest,2),'LineWidth',3);hold on;
plot(100-mean(accTrain,2),'LineWidth',3);
plot(100-accTest,'--');
plot(100-accTrain,'-');
set(gcf,'Color','w');
legend({'Test','Train'},'Location','northeast');
xlabel('Epochs');
ylabel('Error(%)')
title([num2str(Nfold) '-fold Testing']);
ylim([0 100]); xlim([0 Nepochs-1]);
set(gca,'YScale','log');



