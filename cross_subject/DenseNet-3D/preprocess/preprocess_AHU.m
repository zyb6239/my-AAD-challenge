% c3;
format long
% add your eeglab address,or you can add the path to dir
% addpath(genpath('D:\eeglab_current\eeglab2022.0'));
% produce 2*2*2=8 data
data_types = {'1D','2D'};
paralen = 128;
sbnum = 8;
trnum = 16;

dataset = 'AHU';

data1D_name = [dataset '_1D.mat'];
data2D_name = [dataset '_2D.mat'];

EEG = zeros(sbnum,trnum,136*paralen,32);
ENV = zeros(sbnum,trnum,136*paralen,1);

rawdir=['E:/first/ISCSLP_Challenge/raw_data/preprocess/audio_only'];

fs = 128; % sampling rate
Wn = [14 31]/(fs/2);
order = 8;
[b,a] = butter(order,Wn,'bandpass');

sbname = ["SA","SB","SC","SD","SE","SF","SG","SH"];

for sb = 1:sbnum
    path = fullfile(rawdir,sbname(sb)+".mat");
    load(path);
    
    for tr = 1:trnum
        disp(['preprocess_data      subject:' num2str(sb) '   trial:' num2str(tr)]);
        trial = trials{tr};%read the trialnum's trial

        tmp = double(trial.EEG);

        eegtrain = tmp(1:136*paralen,:)';
        eegtrain_new = zeros(size(eegtrain));
        
        % We use 8-order IIR filter this time, and all the later result is
        % same
        for ch = 1:32
            x = eegtrain(ch,:);
            y = filter(b,a,x);
            eegtrain_new(ch,:) = y;
        end
        fs = 128;
        EEG_trial = pop_importdata('dataformat','array','nbchan',0,'data','eegtrain_new','srate',fs,'pnts',0,'xmin',0);

%         [EEG_trial,com,b] = pop_eegfiltnew(EEG_trial, 14,31,512,0,[],0);

        % verify the filter
        % [Pxx, F] = spectopo(EEG_trial.data, 0, 128, 'freqrange', [] );
        eegtrain = EEG_trial.data';

        % mean and std
        % 1e-12: avoid dividing zero
        eegtrain = (eegtrain-mean(eegtrain,2))./(std(eegtrain,0,2)+1e-12);

        % give label
        if trial.attended_ear=='L'
            labeltrain = ones(136*paralen,1);
        else
            labeltrain = zeros(136*paralen,1);
        end

        EEG(sb,tr,:,:) = eegtrain;
        ENV(sb,tr,:,:) = labeltrain;
    end

end

save(['E:/first/ISCSLP_Challenge/raw_data/preprocess/' data1D_name],'EEG','ENV');


% expand 1d to 2d

load(['E:/first/ISCSLP_Challenge/raw_data/preprocess/' data1D_name]);
eeglen = size(ENV,2);
EEG_2D = zeros(sbnum,trnum,136*paralen,7,7);


[~,map,~] = xlsread('/EEG_2D.xlsx'); % the channel position
load('E:/first/AVED/audio_only_eeg/processed_data/channel_list.mat') % the channel order
axis = zeros(32,2);
for cha = 1:32
    for w = 1:7
        for h = 1:7
            if strcmp(EEGMAP{cha},map{w,h})==1
                axis(cha,1) = w;
                axis(cha,2) = h;
            end
        end
    end
end

for cha = 1:32
    EEG_2D(:,:,:,axis(cha,1),axis(cha,2)) = EEG(:,:,:,cha);
end
EEG = EEG_2D;
save(['E:/first/ISCSLP_Challenge/raw_data/preprocess/' data2D_name],'EEG','ENV');







