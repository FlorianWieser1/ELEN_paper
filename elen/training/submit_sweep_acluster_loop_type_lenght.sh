#!/bin/bash
# Script to achieve sweep paralellization with multiple gpus
# For each gpu an independend sweep agent will be started, but merged in wandb
# Usage: ./sweep_submitter.sh $NUM_GPUS $SWEEPS_PER_GPU $TAG $YAML
source activate elen_test

if [ "$#" == 0 ] ; then
	NUM_GPUS=1
	SWEEPS_PER_GPU=1
	TAG="sweep-test"
	YAML="sweep_dummy.yaml"
else
	NUM_GPUS=$1 ; SWEEPS_PER_GPU=$2 ; TAG=$3 ; YAML=$4
fi

for PATH_FOLDER in $DATA_PATH/out_LP_500k_split_HH* ; do
	FOLDER=$(basename $PATH_FOLDER)
	echo $FOLDER
	TAG=$FOLDER

	rm -rf $TAG.* slurm_sweep_logs/test* # delete old log-files
	
	# set job name from command-line
	sed -i "1s/.*/name: $TAG/" $YAML
	#sed -i "s/LP_25k/$FOLDER/"	$YAML
	sed -i "/--train_dir/c\  - --train_dir=/home/florian_wieser/software/ARES/geometricDL/ARES_PL_sweep/elen_datasets/$FOLDER/pdbs/train" $YAML
	sed -i "/--val_dir/c\  - --val_dir=/home/florian_wieser/software/ARES/geometricDL/ARES_PL_sweep/elen_datasets/$FOLDER/pdbs/val" $YAML
	sed -i "/--test_dir/c\  - --test_dir=/home/florian_wieser/software/ARES/geometricDL/ARES_PL_sweep/elen_datasets/$FOLDER/pdbs/test" $YAML

	#wandb init # not each time neccessary
	wandb sweep $YAML &> sweep_id.tmp # generate sweep id 
	# parse sweep id for later use of sweep agent
	SWEEP_ID_STRING=$(grep "Run sweep agent" sweep_id.tmp)
	#rm sweep_id.tmp
	SWEEP_ID="${SWEEP_ID_STRING##*/}"
	
	# spawn NUM_GPUS sweep agent jobs with each --count n sweep trajectories
	echo "Submitting $NUM_GPUS agents with $SWEEPS_PER_GPU runs, for $TAG with sweep id $SWEEP_ID."
	sbatch --array=1-$NUM_GPUS -J $TAG \
							   -o slurm_sweep_logs/$TAG.%A_%a.log \
							   -e slurm_sweep_logs/$TAG.%A_%a.err \
							   --gres gpu:1 \
							   --ntasks-per-node 1 <<EOF
#!/bin/bash
source activate elen_test
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
wandb agent --count $SWEEPS_PER_GPU $SWEEP_ID
EOF
done
