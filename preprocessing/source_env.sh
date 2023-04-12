# System language setting:
# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8

# # Miniconda
# source /path/to/your/Miniconda/etc/profile.d/conda.sh

# ANTs
# export ANTSPATH="/path/to/your/ANTs/"
# export PATH=${ANTSPATH}:${PATH}

# FreeSurfer
# export FREESURFER_HOME="/Applications/freesurfer"
# source ${FREESURFER_HOME}/SetUpFreeSurfer.sh &> /dev/null

# FSL
# Uncomment the line below if you are on Mac:
#export FSLDIR="/usr/local/fsl"
# Uncomment the line below if you are on Linux:
#export FSLDIR="/usr/share/fsl/6.0"
# export PATH="${FSLDIR}/bin":${PATH}
# source ${FSLDIR}/etc/fslconf/fsl.sh

# Matlab
export MATLAB_HOME="/usr/local/MATLAB/R2023a"
export PATH=${MATLAB_HOME}:${PATH}
export MATLABCMD=${MATLAB_HOME}/matlab

# SPM
export SPM_HOME="thirdparty/spm12"


# Only with windows
# export PATH="thirdparty\dcm2niix_win:$PATH"