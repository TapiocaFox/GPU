#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for remote systems
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read benchmark results
df = pd.read_csv('benchmark_results.csv')

# Format N values for display (convert to readable format)
def format_n(n):
    if n >= 1000000000:
        return f'{n//1000000000}B'
    elif n >= 1000000:
        return f'{n//1000000}M'
    elif n >= 1000:
        return f'{n//1000}K'
    else:
        return str(n)

n_labels = [format_n(n) for n in df['N']]

# Create the bar graph
fig, ax = plt.subplots(figsize=(12, 7))

# Create bars
bars = ax.bar(n_labels, df['Speedup'], color='steelblue', edgecolor='navy', linewidth=1.5)

# Customize the plot
ax.set_xlabel('N (Input Size)', fontsize=14, fontweight='bold')
ax.set_ylabel('Speedup (CPU Time / GPU Time)', fontsize=14, fontweight='bold')
ax.set_title('GPU Speedup vs CPU for Prime Number Generation\n(Sieve of Eratosthenes)', 
             fontsize=16, fontweight='bold', pad=20)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add speedup values on top of bars
for i, (bar, speedup) in enumerate(zip(bars, df['Speedup'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{speedup:.2f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add a horizontal line at y=1 to show break-even point
ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break-even (1x)')

# Customize tick labels
ax.tick_params(axis='both', labelsize=11)

# Add legend
ax.legend(fontsize=11)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_file = 'speedup_graph.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n{'='*60}")
print(f"Speedup graph saved to: {output_file}")
print(f"{'='*60}")

# Create a detailed table
print("\n" + "="*60)
print("BENCHMARK RESULTS SUMMARY")
print("="*60)
print(f"{'N':<15} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<10}")
print("-"*60)
for _, row in df.iterrows():
    print(f"{row['N']:<15} {row['CPU_Time']:<15.4f} {row['GPU_Time']:<15.4f} {row['Speedup']:<10.2f}x")
print("="*60)
print(f"\nGraph file: {output_file}")
print("Download the PNG file to view the speedup visualization.")
print("="*60 + "\n")

# Close the figure to free memory
plt.close()