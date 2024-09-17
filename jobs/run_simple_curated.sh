#!/usr/bin/bash
#SBATCH --job-name=simple_curated
#SBATCH --time=1:00:00
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH --array=1-10

source ~/.bashrc
conda activate perturb
ml load matlab

SCRIPT_PATH=/home/users/zys/zys/referencefitting/jobs
JOB_PATH=/home/groups/xiaojie/zys/referencefitting/jobs

paths=`sed -n "${SLURM_ARRAY_TASK_ID} p" paths_curated`
pathsArray=($paths)
DATA_PATH=${pathsArray[0]}
echo $DATA_PATH

CURATED="--curated"
if [ -z ${CURATED+x} ]; then
    backbone=$(echo $(basename $DATA_PATH) | cut -d'-' -f 2)
else
    backbone=$(echo $(basename $DATA_PATH) | cut -d'-' -f 1)
fi
echo $backbone

N_GENES=$(awk 'END {print NR-1}' $JOB_PATH/$backbone"_centralities.csv")

for T in 5; do
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python $SCRIPT_PATH/run_simple.py $DATA_PATH --T $T $CURATED
done
