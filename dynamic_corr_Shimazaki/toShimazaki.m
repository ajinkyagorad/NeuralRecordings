% convert speech data to shimazaki format
clear all;

load 'speechDATA200.mat'
%%
% Get structure raw from the data
% Complete structure of spike timing data is as follows:
%data.raw.xs: {{1x300 cell}  {1x300 cell}  {1x300 cell}};

%data.raw.N = 3;            %Number of neurons.
%data.raw.n = 300;          %Number of repeated trials.
%data.raw.D = 0.001;        %Sampling resolution [s] of spike timing data.
%data.raw.period = [0 0.5]; %Period of observation.
%data.raw.miss = [];        %Optional
%data.raw.ext = [];         %Optional
%data.raw.ext = [];         %Optional
% 
% N: 3
% n: 300
% D: 1.0000e-03
% xs: {{1×300 cell}  {1×300 cell}  {1×300 cell}}
% miss: []
% ext: []
% period: [0 0.5000]

raw = struct([]);
Neurons =[1:10];% length(DATA(1).RES(:,1)); 
dt =1E-3;
for i =1:10
    for j =1:20 % same digit
        for n = 1:length(Neurons)
            % Complete structure of spike timing data is as follows:
            tspike = find(DATA(i,j).RES(Neurons(n),:))*dt;
            raw(i).xs{n}{j} = tspike;
            raw(i).N = length(Neurons);            %Number of neurons.
            raw(i).n = 20;          %Number of repeated trials.
            raw(i).D = dt;        %Sampling resolution [s] of spike timing data.
            raw(i).period = [0 dt*length(DATA(i,j).RES(1,:))]; %Period of observation.
            raw(i).miss = [];        %Optional
            raw(i).ext = [];         %Optional
            raw(i).ext = [];         %Optional
            
        end
    end
end
save('speechReservoirShimazaki.mat','raw');
