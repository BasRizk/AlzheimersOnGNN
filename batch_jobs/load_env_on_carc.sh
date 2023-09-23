ENV_NAME=alz

module load gcc/11.3.0 
module load git/2.36.1 
module load git-lfs/3.2.0 


module load conda/23.3.1 


# CUDA
module load intel/19.0.4
module load cuda/10.2.89

# module load gcc/8.3.0 
# module load cuda/11.3.0


module load git vim


# MATLAB
module load matlab/2022a


if $(conda env list | grep -q $ENV_NAME); then
    echo "It's there."
else
    echo "It's not there. Create env $ENV_NAME before calling this script."
    exit 1
    # conda env create -f $ENV_NAME.yml
fi

source activate $ENV_NAME
echo "Done init Env with $ENV_NAME"