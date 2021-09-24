import os
import time

runs = 5
run_counter = 0
for h_size in [10,100,1000,10000,100000]:
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

