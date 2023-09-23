n_gpus=1
gpu_name=v100
[ ! -z "$1" ] && n_gpus=$(( $1 > 4 ? 4 : $1))
[ ! -z "$2" ] && gpu_name=$2 
echo "Allocating $n_gpus $gpu_name"

ACCOUNT=ajiteshs_1045

salloc --time=2:00:00 --cpus-per-task=12 --mem=16GB --account=$ACCOUNT --gres=gpu:$gpu_name:$n_gpus --partition=gpu
