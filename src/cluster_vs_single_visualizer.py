import matplotlib.pyplot as plt
import numpy as np

# Data for comparison
metrics = ['Throughput', 'Scalability', 'Fault Tolerance']
single_node = [20, 0, 0]  # 20MB/s, 0 for no scalability, 0 for no fault tolerance
distributed = [120, 1, 99.2]  # 120MB/s, 1 for scalable, 99.2% fault tolerance

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Bar positions
x = np.arange(len(metrics))
width = 0.35

# Plot bars
rects1 = ax.bar(x - width/2, single_node, width, label='Single Node Processing')
rects2 = ax.bar(x + width/2, distributed, width, label='4-Node Distributed Processing')

# Add labels and title
ax.set_ylabel('Performance')
ax.set_title('Comparison Between Single Node and Distributed Processing')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Custom y-axis for different metrics
ax.set_yticks([0, 20, 50, 80, 100, 120])
ax.set_yticklabels(['No', '20MB/s', '50MB/s', '80MB/s', '100%', '120MB/s'])

# Save figures
plt.savefig('cluster_vs_single_comparison.png')
plt.savefig('cluster_vs_single_comparison.pdf')
plt.close()