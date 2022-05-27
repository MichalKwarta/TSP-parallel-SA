from asyncio import subprocess
from subprocess import Popen, PIPE
import os
from typing import Tuple
import json
import cuda
import seq

iteration_limit = 20
data = {}
for file in os.listdir(os.getcwd()+"/data"):
    print(file)
    data[file] = {'seq':seq.benchmark(f"data/{file}",iteration_limit),'cuda':cuda.benchmark(f"data/{file}",iteration_limit)}





with open('results.json','w+') as f:
    json.dump(data,f,indent=6)
