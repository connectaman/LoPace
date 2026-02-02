"""
Generate comprehensive visualizations for LoPace compression metrics.
Creates publication-quality SVG plots suitable for research papers.
"""

import os
import sys
import time
import tracemalloc
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use Agg backend which supports both SVG and PNG
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd

# Add parent directory to path to import lopace
sys.path.insert(0, str(Path(__file__).parent.parent))

from lopace import PromptCompressor, CompressionMethod


def save_both_formats(output_dir: Path, filename_base: str):
    """Save figure in both SVG and high-quality PNG formats."""
    # Save SVG
    plt.savefig(output_dir / f'{filename_base}.svg', format='svg', bbox_inches='tight', dpi=300)
    # Save PNG with high quality
    plt.savefig(output_dir / f'{filename_base}.png', format='png', bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  Saved: {filename_base}.svg and {filename_base}.png")


# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',  # Default format, but we'll save both
    'svg.fonttype': 'none',  # Editable text in SVG
    'mathtext.default': 'regular',
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.0,
    'patch.linewidth': 1.5,
})


def load_prompts_from_jsonl(jsonl_path: Path) -> List[Tuple[str, str]]:
    """Load prompts from JSONL file. Returns list of (title, prompt_text) tuples."""
    prompts = []
    
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    print(f"Loading prompts from: {jsonl_path}")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                # Extract markdown content (the actual prompt text)
                markdown = data.get('markdown', '')
                title = data.get('title', f'Prompt {line_num}')
                
                # Use markdown as the prompt text
                if markdown and len(markdown.strip()) > 0:
                    prompts.append((title, markdown))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(prompts)} prompts from JSONL file")
    return prompts


def measure_compression(
    compressor: PromptCompressor,
    prompt: str,
    method: CompressionMethod
) -> Dict:
    """Measure compression metrics for a given prompt and method."""
    # Memory tracking
    tracemalloc.start()
    
    # Compression
    start_time = time.perf_counter()
    compressed = compressor.compress(prompt, method)
    compression_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    compression_memory = peak / (1024 * 1024)  # MB
    
    tracemalloc.stop()
    
    # Decompression
    tracemalloc.start()
    start_time = time.perf_counter()
    decompressed = compressor.decompress(compressed, method)
    decompression_time = time.perf_counter() - start_time
    current, peak = tracemalloc.get_traced_memory()
    decompression_memory = peak / (1024 * 1024)  # MB
    tracemalloc.stop()
    
    # Verify losslessness
    is_lossless = prompt == decompressed
    
    # Calculate metrics
    original_size = len(prompt.encode('utf-8'))
    compressed_size = len(compressed)
    
    metrics = {
        'method': method.value,
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
        'space_savings_percent': (1 - compressed_size / original_size) * 100 if original_size > 0 else 0,
        'bytes_saved': original_size - compressed_size,
        'compression_time_ms': compression_time * 1000,
        'decompression_time_ms': decompression_time * 1000,
        'compression_throughput_mbps': (original_size / (1024 * 1024)) / compression_time if compression_time > 0 else 0,
        'decompression_throughput_mbps': (compressed_size / (1024 * 1024)) / decompression_time if decompression_time > 0 else 0,
        'compression_memory_mb': compression_memory,
        'decompression_memory_mb': decompression_memory,
        'is_lossless': is_lossless,
        'num_characters': len(prompt),
    }
    
    return metrics


def run_benchmarks(jsonl_path: Path) -> pd.DataFrame:
    """Run compression benchmarks on all prompts and methods."""
    compressor = PromptCompressor(model="cl100k_base", zstd_level=15)
    prompts = load_prompts_from_jsonl(jsonl_path)
    
    all_results = []
    
    print("Running benchmarks...")
    total_prompts = len(prompts)
    for idx, (title, prompt) in enumerate(prompts, 1):
        print(f"  Processing prompt {idx}/{total_prompts} ({len(prompt)} chars)...")
        
        for method in [CompressionMethod.ZSTD, CompressionMethod.TOKEN, CompressionMethod.HYBRID]:
            metrics = measure_compression(compressor, prompt, method)
            metrics['prompt_id'] = idx
            metrics['prompt_title'] = title
            metrics['prompt_length'] = len(prompt)
            all_results.append(metrics)
    
    df = pd.DataFrame(all_results)
    return df


def plot_compression_ratio(df: pd.DataFrame, output_dir: Path):
    """Plot compression ratios by method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Compression ratio by method (boxplot)
    ax1 = axes[0]
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token (BPE)', 'Hybrid']
    
    data_by_method = [df[df['method'] == m]['compression_ratio'].values for m in method_order]
    
    bp = ax1.boxplot(data_by_method, labels=method_labels, patch_artist=True,
                     widths=0.6, showmeans=True, meanline=True)
    
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Compression Ratio', fontweight='bold')
    ax1.set_xlabel('Compression Method', fontweight='bold')
    ax1.set_title('(a) Compression Ratio Distribution by Method', fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    
    # Right: Compression ratio vs prompt length (scatter/line plot)
    ax2 = axes[1]
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token (BPE)', 'Hybrid']
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax2.scatter(method_df['prompt_length'], method_df['compression_ratio'], 
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['compression_ratio'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax2.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax2.set_ylabel('Compression Ratio', fontweight='bold')
    ax2.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax2.set_title('(b) Compression Ratio vs Prompt Length', fontweight='bold', pad=15)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    save_both_formats(output_dir, 'compression_ratio')
    plt.close()


def plot_space_savings(df: pd.DataFrame, output_dir: Path):
    """Plot space savings percentages."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token (BPE)', 'Hybrid']
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    # Create boxplot for space savings by method
    data_by_method = [df[df['method'] == m]['space_savings_percent'].values for m in method_order]
    
    bp = ax.boxplot(data_by_method, labels=method_labels, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Space Savings (%)', fontweight='bold')
    ax.set_xlabel('Compression Method', fontweight='bold')
    ax.set_title('Space Savings by Compression Method', fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim(0, 100)
    
    # Add mean value annotations
    for i, (method, label) in enumerate(zip(method_order, method_labels)):
        mean_val = df[df['method'] == method]['space_savings_percent'].mean()
        ax.text(i + 1, mean_val + 3, f'Mean: {mean_val:.1f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold', color=colors[i])
    
    plt.tight_layout()
    save_both_formats(output_dir, 'space_savings')
    plt.close()


def plot_disk_size_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot original vs compressed disk sizes."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token (BPE)', 'Hybrid']
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    # Top: Scatter plot showing original vs compressed sizes
    ax1 = axes[0]
    
    # Get unique prompts (by prompt_id)
    unique_prompts = df.groupby('prompt_id').first()
    original_sizes = unique_prompts['original_size_bytes'].values / 1024  # KB
    
    x_pos = np.arange(len(unique_prompts))
    width = 0.25
    
    # Plot original size
    ax1.bar(x_pos - width, original_sizes, width, label='Original Size', 
           color='#e74c3c', alpha=0.7)
    
    # Plot compressed sizes for each method
    for i, (method, label, color) in enumerate(zip(method_order, method_labels, colors)):
        method_df = df[df['method'] == method].sort_values('prompt_id')
        compressed_sizes = method_df['compressed_size_bytes'].values / 1024  # KB
        ax1.bar(x_pos + i * width, compressed_sizes, width, label=label, 
               color=color, alpha=0.8)
    
    ax1.set_ylabel('Size (KB)', fontweight='bold')
    ax1.set_xlabel('Prompt ID', fontweight='bold')
    ax1.set_title('Disk Size: Original vs Compressed', fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'P{i+1}' for i in range(len(unique_prompts))], rotation=45, ha='right')
    ax1.legend(loc='upper left', framealpha=0.9, ncol=4)
    ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax1.set_yscale('log')
    
    # Bottom: Space savings distribution
    ax2 = axes[1]
    
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax2.scatter(method_df['prompt_length'], method_df['space_savings_percent'],
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['space_savings_percent'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax2.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax2.set_ylabel('Space Savings (%)', fontweight='bold')
    ax2.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax2.set_title('Space Savings vs Prompt Length', fontweight='bold', pad=15)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 100)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    save_both_formats(output_dir, 'disk_size_comparison')
    plt.close()


def plot_speed_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot compression and decompression speed metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token (BPE)', 'Hybrid']
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    # Top-left: Compression time vs prompt length
    ax1 = axes[0, 0]
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax1.scatter(method_df['prompt_length'], method_df['compression_time_ms'],
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['compression_time_ms'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax1.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax1.set_ylabel('Compression Time (ms)', fontweight='bold')
    ax1.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax1.set_title('(a) Compression Time vs Prompt Length', fontweight='bold', pad=15)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    
    # Top-right: Decompression time vs prompt length
    ax2 = axes[0, 1]
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax2.scatter(method_df['prompt_length'], method_df['decompression_time_ms'],
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['decompression_time_ms'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax2.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax2.set_ylabel('Decompression Time (ms)', fontweight='bold')
    ax2.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax2.set_title('(b) Decompression Time vs Prompt Length', fontweight='bold', pad=15)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    # Bottom-left: Compression throughput vs prompt length
    ax3 = axes[1, 0]
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax3.scatter(method_df['prompt_length'], method_df['compression_throughput_mbps'],
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['compression_throughput_mbps'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax3.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax3.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax3.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax3.set_title('(c) Compression Throughput vs Prompt Length', fontweight='bold', pad=15)
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim(bottom=0)
    ax3.set_xscale('log')
    
    # Bottom-right: Decompression throughput vs prompt length
    ax4 = axes[1, 1]
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax4.scatter(method_df['prompt_length'], method_df['decompression_throughput_mbps'],
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['decompression_throughput_mbps'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax4.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax4.set_ylabel('Throughput (MB/s)', fontweight='bold')
    ax4.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax4.set_title('(d) Decompression Throughput vs Prompt Length', fontweight='bold', pad=15)
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(bottom=0)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    save_both_formats(output_dir, 'speed_metrics')
    plt.close()


def plot_memory_usage(df: pd.DataFrame, output_dir: Path):
    """Plot memory usage during compression and decompression."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token (BPE)', 'Hybrid']
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    # Left: Compression memory vs prompt length
    ax1 = axes[0]
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax1.scatter(method_df['prompt_length'], method_df['compression_memory_mb'],
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['compression_memory_mb'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax1.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax1.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax1.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax1.set_title('(a) Compression Memory Usage', fontweight='bold', pad=15)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)
    ax1.set_xscale('log')
    
    # Right: Decompression memory vs prompt length
    ax2 = axes[1]
    for method, label, color in zip(method_order, method_labels, colors):
        method_df = df[df['method'] == method]
        ax2.scatter(method_df['prompt_length'], method_df['decompression_memory_mb'],
                   label=label, color=color, alpha=0.6, s=50)
        
        # Add trend line
        if len(method_df) > 1:
            z = np.polyfit(method_df['prompt_length'], method_df['decompression_memory_mb'], 1)
            p = np.poly1d(z)
            sorted_lengths = sorted(method_df['prompt_length'].unique())
            ax2.plot(sorted_lengths, p(sorted_lengths), color=color, linestyle='--', 
                    linewidth=2, alpha=0.8)
    
    ax2.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax2.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax2.set_title('(b) Decompression Memory Usage', fontweight='bold', pad=15)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    save_both_formats(output_dir, 'memory_usage')
    plt.close()


def plot_comprehensive_comparison(df: pd.DataFrame, output_dir: Path):
    """Create a comprehensive comparison heatmap."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token\n(BPE)', 'Hybrid']
    
    # Top-left: Compression ratio by method (boxplot data as heatmap)
    ax1 = axes[0, 0]
    compression_ratio_data = []
    for method in method_order:
        method_df = df[df['method'] == method]
        compression_ratio_data.append([method_df['compression_ratio'].mean()])
    
    # Create a single column heatmap
    compression_ratio_matrix = np.array(compression_ratio_data)
    im1 = ax1.imshow(compression_ratio_matrix, cmap='YlOrRd', aspect='auto', vmin=0)
    ax1.set_xticks([0])
    ax1.set_yticks(np.arange(len(method_labels)))
    ax1.set_xticklabels(['All Prompts'])
    ax1.set_yticklabels(method_labels)
    ax1.set_ylabel('Compression Method', fontweight='bold')
    ax1.set_xlabel('', fontweight='bold')
    ax1.set_title('(a) Mean Compression Ratio', fontweight='bold', pad=15)
    
    # Add text annotations
    for i in range(len(method_labels)):
        text = ax1.text(0, i, f'{compression_ratio_matrix[i][0]:.2f}x',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=12)
    
    plt.colorbar(im1, ax=ax1, label='Compression Ratio')
    
    # Top-right: Space savings by method
    ax2 = axes[0, 1]
    space_savings_data = []
    for method in method_order:
        method_df = df[df['method'] == method]
        space_savings_data.append([method_df['space_savings_percent'].mean()])
    
    space_savings_matrix = np.array(space_savings_data)
    im2 = ax2.imshow(space_savings_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax2.set_xticks([0])
    ax2.set_yticks(np.arange(len(method_labels)))
    ax2.set_xticklabels(['All Prompts'])
    ax2.set_yticklabels(method_labels)
    ax2.set_ylabel('Compression Method', fontweight='bold')
    ax2.set_xlabel('', fontweight='bold')
    ax2.set_title('(b) Mean Space Savings (%)', fontweight='bold', pad=15)
    
    for i in range(len(method_labels)):
        text = ax2.text(0, i, f'{space_savings_matrix[i][0]:.1f}%',
                       ha="center", va="center", color="black", fontweight='bold', fontsize=12)
    
    plt.colorbar(im2, ax=ax2, label='Space Savings (%)')
    
    # Bottom-left: Compression throughput by method
    ax3 = axes[1, 0]
    speed_data = []
    for method in method_order:
        method_df = df[df['method'] == method]
        speed_data.append([method_df['compression_throughput_mbps'].mean()])
    
    speed_matrix = np.array(speed_data)
    im3 = ax3.imshow(speed_matrix, cmap='viridis', aspect='auto')
    ax3.set_xticks([0])
    ax3.set_yticks(np.arange(len(method_labels)))
    ax3.set_xticklabels(['All Prompts'])
    ax3.set_yticklabels(method_labels)
    ax3.set_ylabel('Compression Method', fontweight='bold')
    ax3.set_xlabel('', fontweight='bold')
    ax3.set_title('(c) Mean Compression Throughput (MB/s)', fontweight='bold', pad=15)
    
    for i in range(len(method_labels)):
        text = ax3.text(0, i, f'{speed_matrix[i][0]:.2f}',
                       ha="center", va="center", color="white", fontweight='bold', fontsize=12)
    
    plt.colorbar(im3, ax=ax3, label='Throughput (MB/s)')
    
    # Bottom-right: Memory usage by method
    ax4 = axes[1, 1]
    memory_data = []
    for method in method_order:
        method_df = df[df['method'] == method]
        memory_data.append([method_df['compression_memory_mb'].mean()])
    
    memory_matrix = np.array(memory_data)
    im4 = ax4.imshow(memory_matrix, cmap='plasma', aspect='auto')
    ax4.set_xticks([0])
    ax4.set_yticks(np.arange(len(method_labels)))
    ax4.set_xticklabels(['All Prompts'])
    ax4.set_yticklabels(method_labels)
    ax4.set_ylabel('Compression Method', fontweight='bold')
    ax4.set_xlabel('', fontweight='bold')
    ax4.set_title('(d) Mean Compression Memory Usage (MB)', fontweight='bold', pad=15)
    
    for i in range(len(method_labels)):
        text = ax4.text(0, i, f'{memory_matrix[i][0]:.2f}',
                       ha="center", va="center", color="white", fontweight='bold', fontsize=12)
    
    plt.colorbar(im4, ax=ax4, label='Memory (MB)')
    
    plt.tight_layout()
    save_both_formats(output_dir, 'comprehensive_comparison')
    plt.close()


def plot_scalability(df: pd.DataFrame, output_dir: Path):
    """Plot how metrics scale with prompt size."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    method_order = ['zstd', 'token', 'hybrid']
    method_labels = ['Zstd', 'Token (BPE)', 'Hybrid']
    colors = ['#3498db', '#2ecc71', '#9b59b6']
    
    # Get unique prompt sizes
    prompt_sizes = sorted(df['prompt_length'].unique())
    
    # Top-left: Compression ratio vs prompt size
    ax1 = axes[0, 0]
    for method, label, color in zip(method_order, method_labels, colors):
        means = []
        sizes = []
        for size in prompt_sizes:
            subset = df[(df['method'] == method) & (df['prompt_length'] == size)]
            if len(subset) > 0:
                means.append(subset['compression_ratio'].mean())
                sizes.append(size)
        
        ax1.plot(sizes, means, marker='o', linewidth=2.5, markersize=8,
                label=label, color=color, markerfacecolor=color, markeredgewidth=2)
    
    ax1.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax1.set_ylabel('Compression Ratio', fontweight='bold')
    ax1.set_title('(a) Compression Ratio vs Prompt Size', fontweight='bold', pad=15)
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')
    
    # Top-right: Space savings vs prompt size
    ax2 = axes[0, 1]
    for method, label, color in zip(method_order, method_labels, colors):
        means = []
        sizes = []
        for size in prompt_sizes:
            subset = df[(df['method'] == method) & (df['prompt_length'] == size)]
            if len(subset) > 0:
                means.append(subset['space_savings_percent'].mean())
                sizes.append(size)
        
        ax2.plot(sizes, means, marker='s', linewidth=2.5, markersize=8,
                label=label, color=color, markerfacecolor=color, markeredgewidth=2)
    
    ax2.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax2.set_ylabel('Space Savings (%)', fontweight='bold')
    ax2.set_title('(b) Space Savings vs Prompt Size', fontweight='bold', pad=15)
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    
    # Bottom-left: Compression time vs prompt size
    ax3 = axes[1, 0]
    for method, label, color in zip(method_order, method_labels, colors):
        means = []
        sizes = []
        for size in prompt_sizes:
            subset = df[(df['method'] == method) & (df['prompt_length'] == size)]
            if len(subset) > 0:
                means.append(subset['compression_time_ms'].mean())
                sizes.append(size)
        
        ax3.plot(sizes, means, marker='^', linewidth=2.5, markersize=8,
                label=label, color=color, markerfacecolor=color, markeredgewidth=2)
    
    ax3.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax3.set_ylabel('Compression Time (ms)', fontweight='bold')
    ax3.set_title('(c) Compression Time vs Prompt Size', fontweight='bold', pad=15)
    ax3.legend(framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    
    # Bottom-right: Memory vs prompt size
    ax4 = axes[1, 1]
    for method, label, color in zip(method_order, method_labels, colors):
        means = []
        sizes = []
        for size in prompt_sizes:
            subset = df[(df['method'] == method) & (df['prompt_length'] == size)]
            if len(subset) > 0:
                means.append(subset['compression_memory_mb'].mean())
                sizes.append(size)
        
        ax4.plot(sizes, means, marker='d', linewidth=2.5, markersize=8,
                label=label, color=color, markerfacecolor=color, markeredgewidth=2)
    
    ax4.set_xlabel('Prompt Length (characters)', fontweight='bold')
    ax4.set_ylabel('Memory Usage (MB)', fontweight='bold')
    ax4.set_title('(d) Memory Usage vs Prompt Size', fontweight='bold', pad=15)
    ax4.legend(framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xscale('log')
    
    plt.tight_layout()
    save_both_formats(output_dir, 'scalability_analysis')
    plt.close()


def plot_original_vs_decompressed(jsonl_path: Path, output_dir: Path):
    """Plot original vs decompressed data comparison across multiple prompts."""
    compressor = PromptCompressor(model="cl100k_base", zstd_level=15)
    prompts = load_prompts_from_jsonl(jsonl_path)
    
    # Select a diverse sample of prompts for visualization (up to 5)
    num_to_show = min(5, len(prompts))
    selected_prompts = prompts[:num_to_show]
    
    # Use Hybrid method (best compression)
    method = CompressionMethod.HYBRID
    
    fig, axes = plt.subplots(len(selected_prompts), 1, figsize=(16, 14))
    if len(selected_prompts) == 1:
        axes = [axes]
    
    fig.suptitle('Original vs Decompressed: Lossless Compression Verification', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    for idx, (title, prompt) in enumerate(selected_prompts):
        ax = axes[idx]
        
        # Compress and decompress
        compressed = compressor.compress(prompt, method)
        decompressed = compressor.decompress(compressed, method)
        
        # Verify losslessness
        is_lossless = prompt == decompressed
        
        # Create representation: show byte-by-byte or character-by-character
        original_bytes = prompt.encode('utf-8')
        decompressed_bytes = decompressed.encode('utf-8')
        
        # Sample points for visualization (every Nth byte/char for performance)
        sample_rate = max(1, len(original_bytes) // 200)  # ~200 points max
        sample_indices = np.arange(0, len(original_bytes), sample_rate)
        
        # Get byte values (0-255) for visualization
        original_byte_values = np.array([original_bytes[i] for i in sample_indices])
        decompressed_byte_values = np.array([decompressed_bytes[i] for i in sample_indices])
        
        # Normalize to 0-100 range for better visualization
        original_normalized = (original_byte_values / 255.0) * 100
        decompressed_normalized = (decompressed_byte_values / 255.0) * 100
        
        # Plot original (blue line)
        ax.plot(sample_indices, original_normalized, 'b-', linewidth=2.0, 
               label='Original', alpha=0.7)
        
        # Plot decompressed (red line) - should overlap perfectly for lossless
        ax.plot(sample_indices, decompressed_normalized, 'r-', linewidth=2.0, 
               label='Decompressed', alpha=0.7, linestyle='--')
        
        # Mark key compression points (sample every Nth point)
        step = max(1, len(sample_indices) // 20)
        key_indices = sample_indices[::step]
        key_original = original_normalized[::step]
        ax.scatter(key_indices, key_original, 
                  color='red', s=40, alpha=0.8, zorder=5, 
                  label='Sample Points', marker='o', edgecolors='darkred', linewidths=1)
        
        # Add text info
        original_size = len(original_bytes)
        compressed_size = len(compressed)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        space_saved = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
        
        info_text = (f"Size: {original_size} → {compressed_size} bytes "
                    f"({space_saved:.1f}% saved, {compression_ratio:.2f}x) | "
                    f"Lossless: {'✓' if is_lossless else '✗'}")
        
        ax.text(0.02, 0.95, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               fontweight='bold')
        
        ax.set_ylabel(f'{title}\n(Normalized Byte Values)', fontweight='bold')
        ax.set_xlabel('Byte Position' if idx == len(selected_prompts) - 1 else '', fontweight='bold')
        ax.set_title(f'{title} - {len(original_bytes)} bytes', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
        ax.set_ylim(-5, 105)
        
        # Highlight that they overlap perfectly (lossless)
        if is_lossless:
            ax.axhspan(-5, 105, alpha=0.05, color='green', zorder=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_both_formats(output_dir, 'original_vs_decompressed')
    plt.close()


def main():
    """Main function to generate all visualizations."""
    # Create output directory
    output_dir = Path(__file__).parent.parent / 'screenshots'
    output_dir.mkdir(exist_ok=True)
    
    # JSONL file path
    jsonl_path = Path(__file__).parent / 'transformers-4-34-0.jsonl'
    
    print("=" * 70)
    print("LoPace Visualization Generator")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"JSONL file: {jsonl_path}")
    print()
    
    # Run benchmarks
    print("Step 1: Running compression benchmarks...")
    df = run_benchmarks(jsonl_path)
    
    # Save raw data
    csv_path = output_dir / 'benchmark_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved benchmark data to: {csv_path}")
    
    print("\nStep 2: Generating visualizations...")
    
    # Generate all plots
    plot_compression_ratio(df, output_dir)
    plot_space_savings(df, output_dir)
    plot_disk_size_comparison(df, output_dir)
    plot_speed_metrics(df, output_dir)
    plot_memory_usage(df, output_dir)
    plot_comprehensive_comparison(df, output_dir)
    plot_scalability(df, output_dir)
    plot_original_vs_decompressed(jsonl_path, output_dir)
    
    print("\n" + "=" * 70)
    print("Visualization generation complete!")
    print(f"All plots saved to: {output_dir}")
    print("=" * 70)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 70)
    for method in ['zstd', 'token', 'hybrid']:
        method_df = df[df['method'] == method]
        print(f"\n{method.upper()}:")
        print(f"  Mean Compression Ratio: {method_df['compression_ratio'].mean():.2f}x")
        print(f"  Mean Space Savings: {method_df['space_savings_percent'].mean():.2f}%")
        print(f"  Mean Compression Time: {method_df['compression_time_ms'].mean():.2f} ms")
        print(f"  Mean Throughput: {method_df['compression_throughput_mbps'].mean():.2f} MB/s")


if __name__ == "__main__":
    main()