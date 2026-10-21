import matplotlib.pyplot as plt
import numpy as np

# Performance metrics under different constraints
def analyze_resource_constraints():
    # Hardware constraints
    cpu_cores = [1, 2, 4, 8, 16]
    memory_gb = [4, 8, 16, 32, 64]
    gpu_acceleration = [False, True]
    
    # Software constraints
    spark_executors = [1, 2, 4, 8]
    docker_quota = [0.25, 0.5, 1.0, 2.0]
    os_threads = [10, 50, 100, 200]
    
    # Simulated performance data
    latency = {
        'cpu': [120, 80, 45, 30, 25],
        'memory': [90, 75, 60, 55, 50],
        'gpu': [100, 40]
    }
    
    throughput = {
        'spark': [50, 90, 150, 180],
        'docker': [30, 60, 100, 120],
        'threads': [40, 80, 110, 130]
    }
    
    # Plot hardware constraints impact
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cpu_cores, latency['cpu'], 'b-o', label='CPU Cores')
    plt.plot(memory_gb, latency['memory'], 'g-s', label='Memory (GB)')
    plt.xlabel('Resource Quantity')
    plt.ylabel('Processing Latency (ms)')
    plt.title('Hardware Constraints Impact')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(['No GPU', 'With GPU'], latency['gpu'], color=['r', 'b'])
    plt.ylabel('Processing Latency (ms)')
    plt.title('GPU Acceleration Impact')
    
    plt.tight_layout()
    plt.savefig('hardware_constraints.png')
    plt.savefig('hardware_constraints.pdf')
    plt.close()
    
    # Plot software constraints impact
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(spark_executors, throughput['spark'], 'b-o')
    plt.xlabel('Spark Executors')
    plt.ylabel('Throughput (events/sec)')
    plt.title('Spark Configuration')
    
    plt.subplot(1, 3, 2)
    plt.plot(docker_quota, throughput['docker'], 'g-s')
    plt.xlabel('Docker CPU Quota')
    plt.ylabel('Throughput (events/sec)')
    plt.title('Docker Resource Limits')
    
    plt.subplot(1, 3, 3)
    plt.plot(os_threads, throughput['threads'], 'r-^')
    plt.xlabel('OS Threads Limit')
    plt.ylabel('Throughput (events/sec)')
    plt.title('Threading Constraints')
    
    plt.tight_layout()
    plt.savefig('software_constraints.png')
    plt.savefig('software_constraints.pdf')
    plt.close()

if __name__ == "__main__":
    analyze_resource_constraints()