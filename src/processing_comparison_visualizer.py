import matplotlib.pyplot as plt
import numpy as np

# Data for comparison
metrics = ['Processing Latency', 'Data Continuity', 'Use Case']
traditional = [5, 0, 1]  # 5s latency, 0 for discontinuous, 1 for offline
realtime = [0.08, 1, 2]   # 80ms latency, 1 for continuous, 2 for real-time

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Bar positions
x = np.arange(len(metrics))
width = 0.35

# Plot bars
rects1 = ax.bar(x - width/2, traditional, width, label='Traditional Batch Processing')
rects2 = ax.bar(x + width/2, realtime, width, label='Real-time Streaming (Our Solution)')

# Add labels and title
ax.set_ylabel('Performance')
ax.set_title('Comparison Between Traditional and Real-time EEG Processing')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Custom y-axis for different metrics
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(['Discontinuous\n(Offline)', 'Continuous\n(Real-time)', 'Epilepsy Warning', 
                   'BCI', 'Seconds', 'Milliseconds'])

# Save figures
plt.savefig('processing_comparison.png')
plt.savefig('processing_comparison.pdf')
plt.close()