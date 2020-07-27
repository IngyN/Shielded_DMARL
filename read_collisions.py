import pickle
import numpy as np
import csv

to_csv = True
read_coll = False

coll_dir = 'benchmark_files/'
rew_dir= 'learning_curves/'
exp_name = 'test_baseline'

if read_coll:
    coll_file = coll_dir + exp_name + '.pkl'
    with open(coll_file, 'rb') as f:
        c = pickle.load(f, encoding='bytes')

    sum = 0

    for i in c:
        for j in i:
            sum+= j

    print(f' File : {coll_file}, Collisions sum {sum/2}')

if to_csv:
    rew_file = rew_dir + exp_name +'_rewards.pkl'
    new_file = rew_dir + exp_name + '_rewards.csv'

    with open(rew_file, 'rb') as f:
        rew = pickle.load(f, encoding='bytes')

    print(f'Rewards : {rew}')

    print(f' New file :{new_file}')

    with open(new_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\n')
        writer.writerow(rew)