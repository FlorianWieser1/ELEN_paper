#!/bin/bash
# Script to achieve sweep paralellization with multiple gpus
# For each gpu an independend sweep agent will be started, but merged in wandb
# Usage: ./sweep_submitter.sh $NUM_GPUS $SWEEPS_PER_GPU $TAG $YAML
source activate edn
if [ "$#" == 0 ] ; then
	NUM_GPUS=1
	SWEEPS_PER_GPU=1
	TAG="sweep-test"
	YAML="sweep_dummy.yaml"
else
	NUM_GPUS=$1 ; SWEEPS_PER_GPU=$2 ; TAG=$3 ; YAML=$4
fi

rm -rf $TAG.* slurm_sweep_logs/test* # delete old log-files

# set job name from command-line
sed -i "1s/.*/name: $TAG/" $YAML

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
						   --ntasks-per-node 1 \
						   --mem 5G <<EOF
#!/bin/bash
source activate ares
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
wandb agent --count $SWEEPS_PER_GPU $SWEEP_ID
EOF
