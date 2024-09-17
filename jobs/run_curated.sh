#!/usr/bin/bash
#SBATCH --job-name=RF
#SBATCH --time=23:00:00
#SBATCH -p normal
#SBATCH -c 2
#SBATCH --mem=4GB
#SBATCH --array=1-30

source ~/.bashrc
conda activate perturb

SCRIPT_PATH=/home/groups/xiaojie/zys/temporal_perturb/scripts
JOB_PATH=/home/groups/xiaojie/zys/temporal_perturb/jobs

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

# T=5
# T=8
# T=10
# T=15
N_GENES=$(awk 'END {print NR-1}' $JOB_PATH/$backbone"_centralities.csv")
# N_GENES=5

# for T in 5 8 10; do
for T in 3 4; do
for reg in 0.0 0.00001 0.00005 0.0001 0.0005 0.001 0.005; do 
    NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python $SCRIPT_PATH/infer_perturb.py $DATA_PATH --reg $reg  --outfile "A_T_"$T"_reg_"$reg".csv" --T $T $CURATED
    for i in $(seq 1 $N_GENES); do 
        echo "Knockout top $i genes by centrality"
        NUMEXPR_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python $SCRIPT_PATH/infer_perturb.py $DATA_PATH --reg $reg  --outfile "A_T_"$T"_ko"$i"_reg_"$reg".csv" --T $T  --centralities $JOB_PATH/$backbone"_centralities.csv" --numko $i $CURATED
    done
done
done
