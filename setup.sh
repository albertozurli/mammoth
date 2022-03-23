#!/bin/bash
module load python/3.8.2
module load profile/deeplrn
module load autoload pytorch
source venv/bin/activate
srun -A FF4_Axyon --pty --gres=gpu:1 --time 04:00:00 -p m100_usr_prod /bin/bash