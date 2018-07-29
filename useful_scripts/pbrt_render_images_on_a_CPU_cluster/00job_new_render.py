# import commands
import os
import subprocess
import sys
from subprocess import Popen, PIPE
import math
import numpy as np
import random

#rootDir = os.path.dirname(os.path.abspath(__file__))

task_size = 5
#begin_index = 64000
#end_index = 65000

begin_index = 185000
end_index = 190000

num_of_view = end_index - begin_index
num_of_node = int(num_of_view / task_size)

for node_id in range(0, num_of_node):

	index0 = begin_index + task_size * node_id
	index1 = index0 + task_size
	
	file_name =  "job_wu_auto_" + str(index0) + "_" + str(index1) + ".sh"
	f = open(file_name,"w")
	
	line1 = "#!/bin/bash" + '\n'
	line2 = "#SBATCH --mail-user=wu@iam.unibe.ch" + '\n'
	line3 = "#SBATCH --mail-type=end" + '\n'
	line4 = "#SBATCH --job-name=\"GG-" + str(index0) + "_" + str(index1) + "\"" + '\n'
	line5 = "#SBATCH --nodes=1" + '\n'
	line6 = "#SBATCH --time=8:50:00" + '\n'
	line7 = "#SBATCH --mem-per-cpu=4G" + '\n'
	line8 = "#SBATCH --cpus-per-task=12" + '\n'
	line9 = "#SBATCH --ntasks=1" + '\n'
	line10 = "module add gcc/5.3.0" + '\n'
	line11 = "srun python 00random_pbrt_script_uni.py " + str(index0) + " " + str(index1) + '\n'

	f.write(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + line10 + line11)
	f.close()

	command = ["sbatch", file_name]
	#process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	process = subprocess.Popen(command)
