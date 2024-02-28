'''
Script for plotting the scaling graphs (figure 9) for the "Vast TVB parameter space exploration: A Modular Framework
for Accelerating the Multi-Scale Simulation of Human Brain Dynamics" manuscript
'''


import matplotlib.pyplot as plt
import numpy as np


num_processes = [1,2,4,8,16]

# Execution time for each number of processes
execution_time_weak = [541, 559, 569, 580, 576]

# Weak scaling parameters
execution_time_strong = [541, 537, 533, 528, 497]

# Plotting strong and weak scaling on the same graph
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
# plt.plot(num_processes, execution_time[0] / np.array(execution_time), marker='o', linestyle='-', label='Strong Scaling')
plt.plot(num_processes, execution_time_strong, marker='o', linestyle='-', label='Strong Scaling')
plt.plot(num_processes, execution_time_weak, marker='o', linestyle='-', label='Weak Scaling')
plt.xticks(num_processes)
plt.xlabel('Number of GPUs * 4')
plt.ylabel('Execution Time (s)')
plt.title('Simulation')
plt.legend()

# second plot for analysis

# Execution time for each number of processes
execution_time_strong = [10000, 4785, 2250, 1084, 540]

# Weak scaling parameters
execution_time_weak = [10000, 10440, 10500, 10501, 10200]

# Plotting strong and weak scaling on the same graph
plt.subplot(1, 2, 2)
# plt.plot(num_processes, execution_time[0] / np.array(execution_time), marker='o', linestyle='-', label='Strong Scaling')
plt.plot(num_processes, execution_time_strong, marker='o', linestyle='-', label='Strong Scaling')
plt.plot(num_processes, execution_time_weak, marker='o', linestyle='-', label='Weak Scaling')
plt.xticks(num_processes)
plt.yticks(('10,000', '8,000', '6,000','4,000','2,000'))
plt.xlabel('Number of CPUs * 4')
plt.title('Analysis')
plt.legend()



plt.show()
