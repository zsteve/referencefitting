#!/usr/bin/bash
#SBATCH --job-name=RENGE
#SBATCH --time=8:00:00
#SBATCH -p normal
#SBATCH -c 4
#SBATCH --mem=8GB
#SBATCH --array=1-20

source ~/.bashrc
conda activate renge_jax

SCRIPT_PATH=/home/users/zys/zys/referencefitting/jobs
JOB_PATH=/home/groups/xiaojie/zys/referencefitting/jobs

paths=`sed -n "${SLURM_ARRAY_TASK_ID} p" paths`
pathsArray=($paths)
DATA_PATH=${pathsArray[0]}
echo $DATA_PATH

# CURATED="--curated"
if [ -z ${CURATED+x} ]; then
    backbone=$(echo $(basename $DATA_PATH) | cut -d'-' -f 2)
else
    backbone=$(echo $(basename $DATA_PATH) | cut -d'-' -f 1)
fi
echo $backbone

N_GENES=$(awk 'END {print NR-1}' $JOB_PATH/$backbone"_centralities.csv")

# for T in 5 8 10; do
for T in 8; do
# for T in 3 4; do
    NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python $SCRIPT_PATH/run_renge.py $DATA_PATH --T $T $CURATED
    for i in $(seq 1 $N_GENES); do 
        echo "Knockout top $i genes by centrality"
        NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python $SCRIPT_PATH/run_renge.py $DATA_PATH --T $T --centralities $JOB_PATH/$backbone"_centralities.csv" --numko $i $CURATED
    done
done
