ENV_NAME=alz

module load conda/4.12.0
module load gcc/11.3.0 
module load git/2.36.1 
module load git-lfs/3.2.0 

# CUDA
module load intel/19.0.4
module load cuda/10.2.89

# MATLAB
module load matlab/2022a

source activate $ENV_NAME
echo "Done init Env with $ENV_NAME"
