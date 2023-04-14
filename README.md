# AlzheimersOnGNN

Note: All scripts run from the repo root directory

## Init environment
```
conda create -n alz python=3.9
pip install -r requirements.txt
```

### Install Clinica Dependencies

Download dcm2niix
```
conda install -c conda-forge dcm2niix
```

Download SPM12:
```
mkdir thirdparty && cd thirdparty
wget --no-check-certificate https://www.fil.ion.ucl.ac.uk/spm/download/restricted/eldorado/spm12.zip
wget --no-check-certificate https://www.fil.ion.ucl.ac.uk/spm/download/spm12_updates/spm12_updates_r7771.zip
unzip spm12.zip
unzip -o spm12_updates_r7771.zip -d spm12
```

Download Matlab, and don't forget to call inside Matlab once (modify `PREFIX_PATH`): 
```
addpath('PREFIX_PATH/thirdparty/spm12')
spm
```

<!-- conda install -y -c aramislab ants -->
### Download Necessary Data

1. Download ADNI MRI Data: (unzipped directory named currently as `ADNI_MRI_Source_rs-fMRI_BIDS` including folders of subjects)
2. Download ADNI Clinical Data (unzipped directory named currently as `clinical_data_merged` including number of CSV files)
3. Ensure
3. Modify paths if different in `convert_adni_to_bids_clinica.sh` and `preprocess_t1volume_adni_clinica.sh





## Credits and Acknowledgements:
- https://github.com/jamiesonor/imagin
- https://github.com/egyptdj/stagin
- https://github.com/NYUMedML/CNN_design_for_AD
- https://aramislab.paris.inria.fr/clinica/docs/public/latest/Third-party/