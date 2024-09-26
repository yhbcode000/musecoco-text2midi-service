#! /usr/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate MuseCoco
cd /workspace/Music_X_Lab_Env/ChatPiano/modules/musecoco-text2midi-service
python flask_server.py