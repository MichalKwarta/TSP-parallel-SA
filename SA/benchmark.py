from asyncio import subprocess
from subprocess import Popen, PIPE
import os
from typing import Tuple
import json
import sys

sequential_version = f"{os.getcwd()}/main"
openmp_version = f"{os.getcwd()}/mainMP"

def run_SEQ(filename:str,iters = 10)->list[Tuple[float,float]]:
    return [tuple(map(float,Popen([sequential_version,filename], stdout=PIPE).stdout.readline().strip().decode('utf-8').split())) for _ in range(iters)]
        


def run_Parallel(filename:str,workers:int,iters= 10)->list[Tuple[float,float]]:
    return [tuple(map(float,Popen([openmp_version,filename,str(workers)], stdout=PIPE).stdout.readline().strip().decode('utf-8').split())) for _ in range(iters)]


WORKERS_CONFIG = [2**x for x in range(8)]
iteration_limit = 20
data = {}
for file in os.listdir(os.getcwd()+'/data'):
    print(file)
    current_file_data = {'seq': run_SEQ("data/"+file, iteration_limit)}
    for workers_count in WORKERS_CONFIG:    
        print(workers_count)
        key = f'parallel_{workers_count}'
        current_file_data[key] = run_Parallel("data/"+file, workers_count, iteration_limit)
    data[file] = current_file_data


with open('results.json','w+') as f:
    json.dump(data,f,indent=6)
