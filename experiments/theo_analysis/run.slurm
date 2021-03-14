#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J winograd           # The job name
#SBATCH -o ret-%j.err        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ret-%j.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Needed resources

# (TODO)
# Please modify your requirements

#SBATCH -p nv-gpu           # Submit to 'nvidia-gpu' Partitiion or queue
#SBATCH -t 0-00:10:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                    # Request N nodes
#SBATCH --gres=gpu:2                 # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinityd
#SBATCH --constraint=Volta           # Request GPU Type

###
### The system will alloc 8 cores per gpu by default.
### If you need more or less, use following:
###SBATCH --cpus-per-task=8        # Request K cores
###

#SBATCH --qos=gpu-short               # Request QOS Type

#- Operstions
echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#- Load environments
source ~/.bashrc
source /tools/module_env.sh
module list                       # list modules loaded by default

##- tools
module load cmake/3.15.7
module load git/2.17.1
module load vim/8.1.2424
module load gcc/7.5.0

##- language
module load python3/3.6.8

##- cuda
#module load cuda-cudnn/11.0-8.0.4

##- virtualenv
##- source activate baseline
conda activate winograd 

#- Log information

module list                      # list modules loaded by default
echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)
nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit

#- Job step
#python example.py
python speed.py
#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
