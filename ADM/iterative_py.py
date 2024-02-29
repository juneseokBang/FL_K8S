import os
import time

seed = [1,3,4]

for s in seed:
    os.system("CUDA_VISIBLE_DEVICES=1 python run.py --num_clients 20 --rounds 100 --IID 1 --dataset mnist")

    time.sleep(3)
