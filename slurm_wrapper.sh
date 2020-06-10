#! /bin/bash
#SBATCH --output=/checkpoint/dpathak/sim2real2sim/slurm_logs/%x.out
#SBATCH --error=/checkpoint/dpathak/sim2real2sim/slurm_logs/%x.err
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --signal=B:USR1@60
#SBATCH --open-mode=append
#SBATCH --time=3000
#SBATCH --mem=38G
#SBATCH --comment="CoRL deadline 07/28"

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}

# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

# Debug output
echo $SLURM_JOB_ID $SLURM_JOB_NAME $SLURMD_NODENAME $CUDA_VISIBLE_DEVICES
echo ${args}

# Load modules
source /etc/profile.d/modules.sh
source /private/home/dpathak/.bashrc
source /etc/profile

# path
export MUJOCO_PY_MJKEY_PATH=/private/home/dpathak/.mujoco/mjkey.txt
export MUJOCO_PY_MJPRO_PATH=/private/home/dpathak/.mujoco/mujoco200
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/private/home/dpathak/.mujoco/mujoco200/bin

# module add doesn't include CUPTI path and there is some bug in cudnn lib64 path, so have to be added manually for TF2
export LD_LIBRARY_PATH=/public/apps/cudnn/v7.6.5.32-cuda.10.1/lib64/:/public/apps/cuda/10.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# setup for modules and environments
source deactivate
module purge
module load cuda/10.1
module load cudnn/v7.6.5.32-cuda.10.1
module load anaconda3
source activate sim2real2simVenv

cd /private/home/dpathak/projects/sim2real2sim/
python dreamer.py ${args} &
wait $!
