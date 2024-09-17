#!/bin/bash

for i in $(ls -d $SCRATCH/Synthetic/dyn-* | grep -v ko); do 
    for j in $(ls $i | grep -E "1000-([0-9]|10)$"); do
        DIR="$i/$j"
        echo "Processing $DIR"
        cp run.sh $DIR/ # copy all run scripts 
        # cp scripts/params* $DIR/ # copy param sets
        sed -i "s~__DATAPATH__~$DIR~g" $DIR/run.sh
        # sed -i "s~__DATAPATH__~$DIR~g" $DIR/run_cespgrn.sh
        # sed -i "s~__DATAPATH__~$DIR~g" $DIR/run_locCSN.sh
        # sed -i "s~__DATAPATH__~$DIR~g" $DIR/run_spliceJAC.sh
        # # sed -i "s~__DATAPATH__~$DIR~g" $DIR/run_undir.sh
        # echo "Submitting batch job"
        # # sbatch $DIR/run_preprocessing.sh
        # # sbatch $DIR/run.sh
        # sbatch $DIR/run_cespgrn.sh
        # sbatch $DIR/run_locCSN.sh
        # sbatch $DIR/run_spliceJAC.sh
        # # sbatch $DIR/run_undir.sh
    done
done
