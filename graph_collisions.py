import pickle
import numpy as np
import csv

to_csv = True
read_coll = True

coll_dir = 'benchmark_files/'
exp_name = 'cross_1_shield'

if read_coll:
    coll_file = coll_dir + exp_name + '_ag.pkl'
    with open(coll_file, 'rb') as f:
        c = pickle.load(f, encoding='bytes')

    sum_arr = np.zeros([len(c)])

    for i in range(len(c)):
        sum_arr[i] = np.sum(c[i])/2

    print(f' File : {coll_file}, Collisions sum {np.sum(sum_arr)}')

    if to_csv:
        new_file = coll_dir+exp_name+'_collision_graph.csv'
        with open(new_file, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\n')
            writer.writerow(sum_arr)




