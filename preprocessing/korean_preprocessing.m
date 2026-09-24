home_path = 'D:\eeg_datasets\p300' %update the path where you saved your subjects
raw_data_path = fullfile(home_path, 'S01'); % Update this to loop over your 55 subject folders
processed_path = fullfile(home_path, 'Cluster_1_p');
epoched_data_path = fullfile(home_path, 'Cluster_1_epochs_data');
if ~exist(processed_path, 'dir'), mkdir(processed_path); end
if ~exist(epoched_data_path, 'dir'), mkdir(epoched_data_path); end
% Initialize EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');
% List all .mat files
mat_files = dir(fullfile(raw_data_path, '*.mat'));
for k = 1:length(mat_files)
   file_name = mat_files(k).name;
   file_base = erase(file_name, '.mat');
  
   % Load .mat data
   x = load(fullfile(raw_data_path, file_name));
  
   %% 1. Import raw data into EEGLAB
   EEG = pop_importdata( ...
       'dataformat', 'array', ...
       'data', x.data, ...
       'srate', x.srate, ... % Retaining the original 512 Hz
       'nbchan', size(x.data, 1), ...
       'pnts', size(x.data, 2), ...
       'xmin', 0, ...
       'chanlocs', x.chanlocs ...
   );
   EEG.setname = file_base;
   EEG = eeg_checkset(EEG);
  
   %% 2. Create events
   % Latencies map directly to the original 512 Hz sample indices
   EEG.event = struct('type', {}, 'latency', {});
   e_idx = 1;
   for i = 1:length(x.markers_target)
       if x.markers_target(i) == 1 || x.markers_target(i) == 2
           EEG.event(e_idx).type = num2str(x.markers_target(i));
           EEG.event(e_idx).latency = i;
           e_idx = e_idx + 1;
       end
   end
   EEG = eeg_checkset(EEG);
   assert(length(x.markers_target) == EEG.pnts, ...
   'Mismatch: markers_target is not aligned sample-by-sample with EEG data');
  
   %% 3. Filtering
   % Resampling removed; keeping the 0.5 to 30 Hz bandpass filter
   EEG = pop_eegfiltnew(EEG, 0.1, 30); 
  
   %% 4. Automated Bad Channel Cleaning (clean_rawdata)
   original_chanlocs = EEG.chanlocs;
  
   %EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion', 5, ...
       %'ChannelCriterion', 0.85, 'LineNoiseCriterion', 3, ...
       %'Highpass', 'off', 'BurstCriterion', 20);
  
   channels_removed = length(original_chanlocs) - EEG.nbchan;
   EEG = pop_interp(EEG, original_chanlocs, 'spherical');
  
   %% 5. Re-reference
   EEG = pop_reref(EEG, []);            % Common Average Reference
   % No mastoids present
  
   %% 6. Continuous Artifact Removal (ICA & ICLabel)

   data_rank = EEG.nbchan - channels_removed - 1; % account for CAR
   EEG_ica = pop_eegfiltnew(EEG, 1, []);
   EEG_ica = pop_runica(EEG_ica, 'icatype', 'runica', 'extended', 1, 'pca', data_rank);
   EEG.icaweights = EEG_ica.icaweights;
   EEG.icasphere = EEG_ica.icasphere;
   EEG.icawinv = EEG_ica.icawinv;
   EEG.icachansind = EEG_ica.icachansind;
   clear EEG_ica; % Free up memory
  
   EEG = pop_iclabel(EEG, 'default');
  
   % Flag components: [Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other]
   thresholds = [
   0 0;        % Brain
   0.8 1;      % Muscle
   0.8 1;      % Eye
   0.8 1;      % Heart
   0.8 1;      % Line Noise
   0.8 1;      % Channel Noise
   0.8 1       % Other
   ];
   EEG = pop_icflag(EEG, thresholds);
  
   bad_components = find(EEG.reject.gcompreject);
   if ~isempty(bad_components)
       EEG = pop_subcomp(EEG, bad_components, 0);
   end
  
   %% 7. Create basic event list
   EEG = pop_creabasiceventlist(EEG, 'AlphanumericCleaning', 'on', ...
       'BoundaryNumeric', {-99}, 'BoundaryString', {'boundary'});
  
   %% 8. Bin assignment
   EEG = pop_binlister(EEG, ...
       'BDF', fullfile(home_path, 'bin_lister_demo.txt'), ...
       'IndexEL', 1, ...
       'SendEL2', 'EEG', ...
       'Voutput', 'EEG');
  
   %% 9. Epoching and baseline correction
   EEG = pop_epochbin(EEG, [-200 800], 'pre');
  
   %% 10. Epoch Artifact Rejection (artextval)
   %EEG = pop_artextval(EEG, ...
       %'Channel', 1:EEG.nbchan, ...
       %'Flag', 1, ...
       %'Threshold', [-100 100], ...
       %'Twindow', [-200 800]);
  
   EEG.setname = [file_base '_only_ica'];
  
   %% 11. Save epoched .set files
   pop_saveset(EEG, 'filename', [EEG.setname '.set'], 'filepath', processed_path);
  
   %% 12. Compute ERP
   ERP = pop_averager(EEG, ...
       'Criterion', 'good', ...
       'ExcludeBoundary', 'on', ...
       'SEM', 'on');
   ERP.erpname = [file_base '_erp'];
  
   %% 13. Save ERP file
   pop_savemyerp(ERP, ...
       'erpname', ERP.erpname, ...
       'filename', [file_base '.erp'], ...
       'filepath', processed_path);
      
   %% 14. Extract EXACT time-zero labels for Python
   num_epochs = EEG.trials; % Only the epochs that survived artifact rejection
   true_labels = zeros(1, num_epochs);
   for e = 1:num_epochs
       % Get all latencies for this specific epoch
       % Using cell2mat because eventlatency is stored as a cell array
       latencies = [EEG.epoch(e).eventlatency{:}];
      
       % Find the index of the event that happened exactly at 0 ms
       zero_idx = find(latencies == 0, 1);
      
       % Get the corresponding event type
       evt_type = EEG.epoch(e).eventtype{zero_idx};
      
       % Extract the number (this safely handles '1', '2', or ERPLAB's 'B1(1)', 'B2(2)')
       if contains(evt_type, '1')
           true_labels(e) = 1;
       elseif contains(evt_type, '2')
           true_labels(e) = 2;
       end
   end
   %% 15. Save epoched EEG data and perfectly aligned event list
   save(fullfile(epoched_data_path, [file_base '_epoch_data.mat']), 'EEG');
  
   % Save the perfectly aligned labels to a text file
   output_file = fullfile(epoched_data_path, [file_base '_true_labels.txt']);
   fid = fopen(output_file, 'w');
   fprintf(fid, '%d ', true_labels);
   fclose(fid);
end % End of the main subject loop