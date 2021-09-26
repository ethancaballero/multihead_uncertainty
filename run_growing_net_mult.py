import os
import time

#runs = 3
runs = 1
run_counter = 0
#for h_size in [10,100,1000,10000,100000,1000000,10000000]:
for h_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
    for run in range(runs):
        cwd = os.getcwd()
        #command = "python /network/home/ostapeno/dev/multi_head/run_growing_net.py \
        command = "python run_growing_net.py \
        --h_size %(h_size)s \
                  " % locals()
        print(command)

        os.system(command)
        #break
        time.sleep(2)

