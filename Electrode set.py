

#eeg extraction
p_id='chb01_02.edf'

data_dir='data'

raw_file=data_dir + '/' +p_id + '.edf'
raw=mne.io.read_raw_edf(fname)

# set electrode locations
montage='standard_1005'
raw.set_montage(montage)