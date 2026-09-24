
%% ===== Load dataset =====
S = load("/home/ubuntu/data/sess12_subj03_EEG_ERP.mat");
S = S.EEG_ERP_test;
smt  = S.smt;
x    = S.x;
y    = S.y_dec;
fs   = S.fs;
chan = S.chan;
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');
%% =========================================================
%% ===== STEP 1: CREATE CONTINUOUS DATASET (FOR ICA) =======
%% =========================================================
EEG_cont = eeg_emptyset();
EEG_cont.data = x';   % channels × time
EEG_cont.srate = fs;
EEG_cont.nbchan = size(EEG_cont.data,1);
EEG_cont.pnts = size(EEG_cont.data,2);
EEG_cont.trials = 1;
EEG_cont.xmin = 0;
EEG_cont.chanlocs = struct('labels', chan);
EEG_cont = pop_chanedit(EEG_cont, 'lookup', 'standard-10-5-cap385.elp');
EEG_cont = eeg_checkset(EEG_cont);
EEG_cont = pop_rmbase(EEG_cont, []);   % remove DC offset
EEG_cont = eeg_checkset(EEG_cont);
%%=== Resample the dataset====
EEG_cont = pop_resample(EEG_cont, 512);
pop_saveset(EEG_cont, 'filename', 'converted_cont.set');
%% ===== Filtering =====
EEG_cont  = pop_basicfilter( EEG_cont,  1:62 , 'Boundary', 'boundary', 'Cutoff', [ 0.1 30], 'Design', 'fir', 'Filter', 'bandpass', 'Order',2048 );
%% ===== Automated cleaning (continuous) =====
orig_chanlocs = EEG_cont.chanlocs;
orig_labels   = {EEG_cont.chanlocs.labels};
EEG_cont = pop_clean_rawdata(EEG_cont, ...
   'FlatlineCriterion', 5, ...
   'ChannelCriterion', 0.85, ...
   'LineNoiseCriterion', 3, ...
   'Highpass', 'off', ... % [0.25 0.75]
   'BurstCriterion', 10, ...
   'WindowCriterion', 'off', ...
   'BurstRejection', 'off', ...
   'Distance', 'Euclidian');
clean_labels = {EEG_cont.chanlocs.labels};
removed_labels = setdiff(orig_labels, clean_labels);
fprintf('Removed channels: ');
disp(removed_labels);
channels_removed = length(removed_labels);
%% ===== ICA =====
data_rank = EEG_cont.nbchan;
EEG_ica = pop_eegfiltnew(EEG_cont, 1, []);  % stabilize ICA
EEG_ica = pop_runica(EEG_ica, ...
   'extended', 1, ...
   'pca', data_rank);
EEG_cont.icaweights = EEG_ica.icaweights;
EEG_cont.icasphere  = EEG_ica.icasphere;
EEG_cont.icawinv    = EEG_ica.icawinv;
EEG_cont.icachansind = EEG_ica.icachansind;
clear EEG_ica;
%% ===== IC classification =====
EEG_cont = pop_iclabel(EEG_cont, 'default');
thresholds = [
   0 0;        % Brain
   0.8 1;      % Muscle
   0.8 1;      % Eye
   0.8 1;      % Heart
   0.8 1;      % Line Noise
   0.8 1;      % Channel Noise
   0 0;      % Other
   ];
EEG_cont = pop_icflag(EEG_cont, thresholds);
bad_ic = find(EEG_cont.reject.gcompreject);
%% =========================================================
%% ===== STEP 2: CREATE EPOCHED DATASET ====================
%% =========================================================
EEG = eeg_emptyset();
EEG.data = permute(smt, [3 1 2]);   % channels × time × trials
EEG.nbchan = size(EEG.data,1);
EEG.pnts   = size(EEG.data,2);
EEG.trials = size(EEG.data,3);
EEG.srate  = fs;
EEG.xmin = 0;
EEG.xmax = (EEG.pnts-1)/fs;
EEG.chanlocs = struct('labels', chan);
EEG = pop_chanedit(EEG, 'lookup', 'standard-10-5-cap385.elp');
EEG = pop_resample(EEG, 512);
EEG  = pop_basicfilter( EEG,  1:62 , 'Boundary', 'boundary', 'Cutoff', [ 0.1 30], 'Design', 'fir', 'Filter', 'bandpass', 'Order',100);
%EEG = pop_eeglindetrend( EEG, 'Baseline', 'all', 'ChanArray',  1:62 );
%% Event Addition
EEG.event = struct([]);
for i = 1:EEG.trials
   EEG.event(i).type = num2str(y(i));  % safer for EEGLAB
   EEG.event(i).latency = (i-1)*EEG.pnts + 1;
   EEG.event(i).epoch = i;
end
EEG = eeg_checkset(EEG, 'eventconsistency');
%% ===== Apply same bad channel removal =====
if ~isempty(removed_labels)
   EEG = pop_select(EEG, 'nochannel', removed_labels);
end
EEG = pop_eeglindetrend( EEG, 'Baseline', 'all', 'ChanArray',  1:62 );
%% ===== Apply ICA weights =====
EEG.icaweights  = EEG_cont.icaweights;
EEG.icasphere   = EEG_cont.icasphere;
EEG.icawinv     = EEG_cont.icawinv;
EEG.icachansind = EEG_cont.icachansind;
EEG = eeg_checkset(EEG);
if ~isempty(bad_ic)
   EEG = pop_subcomp(EEG, bad_ic, 0);
end
%% ===== Interpolate removed channels =====
EEG = pop_interp(EEG, orig_chanlocs, 'spherical');
%% ===== Final re-reference =====
EEG = pop_reref(EEG, []);
%% ===== Artifact rejection (epoch level) =====
%EEG = pop_eegthresh(EEG, ...
   %1, ...
   %1:EEG.nbchan, ...
   %-100, 100, ...
   %0, 0.8, ...
   %0, 1);
%% ===== Save cleaned dataset =====
pop_saveset(EEG, 'filename', 'clean_epoched_final_1.set', ...
   'filepath', '/home/ubuntu/data/processed/');
disp('Pipeline complete.');

