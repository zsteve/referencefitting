#!/usr/bin/bash
#SBATCH --job-name=bicycle-curated
#SBATCH --time=23:00:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=16GB
#SBATCH --array=1-30

source ~/.bashrc
conda activate perturb

paths=`sed -n "${SLURM_ARRAY_TASK_ID} p" paths_curated`
pathsArray=($paths)
DATA_PATH=${pathsArray[0]}
echo $DATA_PATH

CURATED="--curated"
if [ -z ${CURATED+x} ]; then
    backbone=$(echo $(basename $DATA_PATH) | cut -d'-' -f 2)
    rep=$(echo $(basename $DATA_PATH) | cut -d'-' -f 4)
else
    backbone=$(echo $(basename $DATA_PATH) | cut -d'-' -f 1)
    rep=$(echo $(basename $DATA_PATH) | cut -d'-' -f 3)
fi
echo $backbone $rep

python run_bicycle.py $backbone $rep /scratch/users/zys/Curated
