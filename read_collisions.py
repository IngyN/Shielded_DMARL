import pickle
import numpy as np

file = 'benchmark_files/debug_collisions.pkl'

with open(file, 'rb') as f:
    c = pickle.load(f, encoding='bytes')

sum = 0

for i in c:
    for j in i:
        sum+= j

print(f'Collisions sum {sum}')