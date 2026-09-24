%% =========================
% 0. INIT
% =========================
clear; clc;

addpath(genpath('/home/ubuntu/eeglab'));
addpath('/home/ubuntu/eeglab/plugins/iclabel/matconvnet/matlab');
vl_setupnn();
eeglab nogui;

home_path         = '~/data/adaptive/';
processed_path    = fullfile(home_path, 'processed');
epoched_data_path = fullfile(home_path, 'epoched_data');

if ~exist(processed_path,    'dir'), mkdir(processed_path);    end
if ~exist(epoched_data_path, 'dir'), mkdir(epoched_data_path); end

[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');

%% Write BDF file once
bdf_path = fullfile(home_path, 'bin_lister_demo.txt');
fid = fopen(bdf_path, 'w');
fprintf(fid, 'bin 1\nTargets\n.{1}\n\nbin 2\nNon-Targets\n.{2}\n');
fclose(fid);

files = {
    'calibration-signals.mat'
    'eval-raw.mat'
    'training-run-1-raw.mat'
    'training-run-2-raw.mat'
    'training-run-3-raw.mat'
    'training-run-4-raw.mat'
    'training-run-5-raw.mat'
    'post-training-raw.mat'
};

for subj = 1:47

    subj_folder = sprintf('Subject-%02d', subj);
    subj_path   = fullfile(home_path, subj_folder, subj_folder);

    for f = 1:length(files)

        file_name = files{f};
        fpath     = fullfile(subj_path, file_name);

        if ~isfile(fpath)
            warning('Missing: %s — skipping.', fpath);
            continue;
        end

        [~, fname_noext, ~] = fileparts(file_name);
        file_base = sprintf('Subject-%02d_%s', subj, fname_noext);

        fprintf('\n=== Processing: %s ===\n', file_base);

        x = load(fpath);

        % Validate required fields
        required = {'samples', 'samplingFreq', 'stims', 'channelNames'};
        missing  = required(~cellfun(@(f) isfield(x, f), required));
        if ~isempty(missing)
            warning('Missing fields %s in %s — skipping.', strjoin(missing, ', '), file_name);
            continue;
        end

        samples  = x.samples;
        fs       = x.samplingFreq;
        stims    = x.stims;
        chNames  = x.channelNames;

        %% =========================================================
        %% STEP 1: CONTINUOUS PIPELINE
        %% =========================================================

        EEG_cont         = eeg_emptyset();
        EEG_cont.data    = samples';
        EEG_cont.srate   = fs;
        EEG_cont.nbchan  = size(samples, 2);
        EEG_cont.pnts    = size(samples, 1);
        EEG_cont.trials  = 1;
        EEG_cont.xmin    = 0;

        for i = 1:length(chNames)
            EEG_cont.chanlocs(i).labels = upper(strtrim(chNames{i}));
        end
        EEG_cont = pop_chanedit(EEG_cont, 'lookup', 'standard-10-5-cap385.elp');
        EEG_cont = eeg_checkset(EEG_cont);

        %% 1a. DC offset removal
        EEG_cont = pop_rmbase(EEG_cont, []);
        EEG_cont = eeg_checkset(EEG_cont);

        %% 1b. Bandpass
        EEG_cont = pop_eegfiltnew(EEG_cont, 0.1, 30);

        %% 1c. Bad channel detection via ASR
        orig_chanlocs    = EEG_cont.chanlocs;
        orig_labels      = {EEG_cont.chanlocs.labels};

        EEG_cont = pop_clean_rawdata(EEG_cont, ...
            'FlatlineCriterion',  5,    ...
            'ChannelCriterion',   0.85, ...
            'LineNoiseCriterion', 3,    ...
            'Highpass',           'off', ...
            'BurstCriterion',     20,   ...
            'WindowCriterion',    'off', ...
            'BurstRejection',     'off', ...
            'Distance',           'Euclidian');

        clean_labels     = {EEG_cont.chanlocs.labels};
        removed_channels = setdiff(orig_labels, clean_labels);
        fprintf('Subj %02d %s — Removed channels: ', subj, fname_noext);
        disp(removed_channels);

        %% 1d. ICA on 1 Hz HP copy
        data_rank = min(rank(double(EEG_cont.data')), EEG_cont.nbchan - 1);

        EEG_ica = pop_eegfiltnew(EEG_cont, 1, []);
        EEG_ica = pop_runica(EEG_ica, 'icatype', 'picard', 'maxiter', 500, 'pca', data_rank);

        EEG_cont.icaweights  = EEG_ica.icaweights;
        EEG_cont.icasphere   = EEG_ica.icasphere;
        EEG_cont.icawinv     = EEG_ica.icawinv;
        EEG_cont.icachansind = EEG_ica.icachansind;
        clear EEG_ica;

        %% 1e. ICLabel classification and flagging
        EEG_cont = pop_iclabel(EEG_cont, 'default');

        thresholds = [
            0   0;      % Brain
            0.8 1;      % Muscle
            0.8 1;      % Eye
            0.8 1;      % Heart
            0.8 1;      % Line Noise
            0.8 1;      % Channel Noise
            0   0;      % Other
        ];

        EEG_cont = pop_icflag(EEG_cont, thresholds);
        bad_ic   = find(EEG_cont.reject.gcompreject);

        %% =========================================================
        %% STEP 2: EPOCHED PIPELINE
        %% =========================================================

        EEG = EEG_cont;

        %% 2a. Build events via look-ahead
        stim_times    = stims(:, 1);
        stim_codes    = stims(:, 2);
        event_samples = round(stim_times * fs);

        flash_idx = find(stim_codes == 32779);
        EEG.event = struct('type', {}, 'latency', {});
        e_idx = 1;

        for i = 1:length(flash_idx)
            idx   = flash_idx(i);
            label = '';
            for j = idx+1 : min(idx+5, length(stim_codes))
                code = stim_codes(j);
                if code == 33285
                    label = 'target';
                    break;
                elseif code == 33286
                    label = 'nontarget';
                    break;
                end
            end
            if ~isempty(label)
                EEG.event(e_idx).type    = label;
                EEG.event(e_idx).latency = event_samples(idx);
                e_idx = e_idx + 1;
            end
        end

        if e_idx == 1
            warning('No labeled events found in %s — skipping.', file_base);
            clearvars -except home_path processed_path epoched_data_path ...
                              bdf_path files ALLEEG ALLCOM subj f;
            [ALLEEG, EEG, CURRENTSET] = deal([], eeg_emptyset(), 0);
            continue;
        end

        EEG = eeg_checkset(EEG, 'eventconsistency');

        %% 2a cont. Convert to numeric for ERPLAB
        for i = 1:length(EEG.event)
            if strcmp(EEG.event(i).type, 'target')
                EEG.event(i).type = 1;
            elseif strcmp(EEG.event(i).type, 'nontarget')
                EEG.event(i).type = 2;
            end
        end
        EEG = eeg_checkset(EEG, 'eventconsistency');

        %% 2b. Remove bad channels
        if ~isempty(removed_channels)
            EEG = pop_select(EEG, 'nochannel', removed_channels);
        end

        %% 2c. Subtract artifact ICs
        assert(isequal({EEG.chanlocs.labels}, ...
                       {EEG_cont.chanlocs(EEG_cont.icachansind).labels}), ...
               'Channel mismatch before ICA subtraction: %s', file_base);

        if ~isempty(bad_ic)
            EEG = pop_subcomp(EEG, bad_ic, 0);
        end

        %% 2d. Interpolate removed channels
        EEG = pop_interp(EEG, orig_chanlocs, 'spherical');

        %% 2e. CAR
        EEG = pop_reref(EEG, []);

        %% 2f. ERPLAB eventlist and bin assignment
        EEG = pop_creabasiceventlist(EEG, ...
            'AlphanumericCleaning', 'on', ...
            'BoundaryNumeric',      {-99}, ...
            'BoundaryString',       {'boundary'});

        EEG = pop_binlister(EEG, ...
            'BDF',     bdf_path, ...
            'IndexEL', 1, ...
            'SendEL2', 'EEG', ...
            'Voutput', 'EEG');

        %% 2g. Epoch and baseline correction
        EEG = pop_epochbin(EEG, [-200 800], 'pre');

        %% 2h. Artifact rejection
        EEG = pop_artextval(EEG, ...
            'Channel',   1:EEG.nbchan, ...
            'Flag',      1, ...
            'Threshold', [-100 100], ...
            'Twindow',   [-200 800]);

        EEG.setname = [file_base '_clean'];

        %% 2i. Save epoched .set
        pop_saveset(EEG, 'filename', [EEG.setname '.set'], 'filepath', processed_path);

        %% 2j. Compute and save ERP
        ERP = pop_averager(EEG, ...
            'Criterion',       'good', ...
            'ExcludeBoundary', 'on', ...
            'SEM',             'on');
        ERP.erpname = [file_base '_erp'];

        pop_savemyerp(ERP, ...
            'erpname',  ERP.erpname, ...
            'filename', [file_base '.erp'], ...
            'filepath', processed_path);

        %% 2k. Extract time-zero labels
        num_epochs  = EEG.trials;
        true_labels = zeros(1, num_epochs);

        for e = 1:num_epochs
            latencies = [EEG.epoch(e).eventlatency{:}];
            zero_idx  = find(latencies == 0, 1);
            evt_type  = EEG.epoch(e).eventtype{zero_idx};
            if strcmp(evt_type, '1') || (isnumeric(evt_type) && evt_type == 1)
                true_labels(e) = 1;
            elseif strcmp(evt_type, '2') || (isnumeric(evt_type) && evt_type == 2)
                true_labels(e) = 2;
            end
        end

        %% 2l. Save mat + txt outputs
        save(fullfile(epoched_data_path, [file_base '_epoch_data.mat']), 'EEG');

        fid = fopen(fullfile(epoched_data_path, [file_base '_true_labels.txt']), 'w');
        fprintf(fid, '%d\n', true_labels);
        fclose(fid);

        fid = fopen(fullfile(epoched_data_path, [file_base '_removed_channels.txt']), 'w');
        if isempty(removed_channels)
            fprintf(fid, 'None\n');
        else
            fprintf(fid, '%s\n', removed_channels{:});
        end
        fclose(fid);

        fprintf('Done: %s\n', file_base);

        clearvars -except home_path processed_path epoched_data_path ...
                          bdf_path files ALLEEG ALLCOM subj f;
        [ALLEEG, EEG, CURRENTSET] = deal([], eeg_emptyset(), 0);

    end  % file loop

    %% S3 sync after each subject completes
    s3_cmd = sprintf('aws s3 sync "%s" "s3://amzn-eeg-bucket/adaptiveP300/" --exclude "*" --include "Subject-%02d_*"', ...
                     processed_path, subj);
    status = system(s3_cmd);
    if status ~= 0
        warning('S3 sync failed for Subject-%02d', subj);
    else
        fprintf('S3 sync complete for Subject-%02d\n', subj);
    end

    s3_cmd = sprintf('aws s3 sync "%s" "s3://amzn-eeg-bucket/adaptiveP300/epoched/" --exclude "*" --include "Subject-%02d_*"', ...
                     epoched_data_path, subj);
    status = system(s3_cmd);
    if status ~= 0
        warning('S3 epoched sync failed for Subject-%02d', subj);
    else
        fprintf('S3 epoched sync complete for Subject-%02d\n', subj);
    end

end  % subject loop

disp('All done.');

