import matplotlib.pyplot as plt
import numpy as np

# Sample data for different chip types
chips = ['CPU', 'GPU', 'TPU', 'FPGA']
processor_performance = [0.6, 0.9, 0.8, 0.7]  # Normalized performance for processor module
visualization_performance = [0.5, 0.95, 0.7, 0.6]  # Normalized performance for visualization module

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Performance Comparison Across Different Chip Types', fontsize=14)

# Processor module performance
ax1.bar(chips, processor_performance, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])
ax1.set_title('Processor Module Performance')
ax1.set_ylabel('Normalized Performance')
ax1.set_ylim(0, 1)

# Visualization module performance
ax2.bar(chips, visualization_performance, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])
ax2.set_title('Visualization Module Performance')
ax2.set_ylabel('Normalized Performance')
ax2.set_ylim(0, 1)

# Save figures
plt.tight_layout()
plt.savefig('processor_vs_visualization_chip_performance.png', dpi=300, bbox_inches='tight')
plt.savefig('processor_vs_visualization_chip_performance.pdf', format='pdf', bbox_inches='tight')
plt.close()

print("Visualization saved as 'processor_vs_visualization_chip_performance.png' and '.pdf'")