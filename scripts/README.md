# Visualization Scripts

This directory contains scripts for generating publication-quality visualizations of LoPace compression metrics.

## generate_visualizations.py

Generates comprehensive SVG visualizations suitable for research papers.

### Features

- Tests all compression methods (Zstd, Token, Hybrid) on 10 prompts
- Prompts include small, medium, and very large sizes
- Measures comprehensive metrics:
  - Compression ratio
  - Space savings
  - Disk size (original vs compressed)
  - Compression/decompression speed
  - Memory usage
  - Throughput

### Generated Visualizations

1. **compression_ratio.svg** - Compression ratio distribution and by prompt size
2. **space_savings.svg** - Space savings percentages with error bars
3. **disk_size_comparison.svg** - Original vs compressed disk sizes
4. **speed_metrics.svg** - Compression/decompression time and throughput
5. **memory_usage.svg** - Memory usage during compression/decompression
6. **comprehensive_comparison.svg** - Heatmaps comparing all metrics
7. **scalability_analysis.svg** - How metrics scale with prompt size

### Usage

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run the script
python scripts/generate_visualizations.py
```

### Output

All visualizations are saved as high-resolution SVG files in the `screenshots/` directory:

```
screenshots/
├── benchmark_data.csv          # Raw benchmark data
├── compression_ratio.svg
├── space_savings.svg
├── disk_size_comparison.svg
├── speed_metrics.svg
├── memory_usage.svg
├── comprehensive_comparison.svg
└── scalability_analysis.svg
```

### Plot Specifications

- **Format**: SVG (vector graphics, scalable)
- **Resolution**: 300 DPI equivalent
- **Fonts**: Serif fonts (Times New Roman) for publication quality
- **Style**: Clean, professional design with legends and labels
- **Size**: Optimized for paper inclusion (adjustable)

### Customization

Edit `scripts/generate_visualizations.py` to customize:
- Number of test prompts
- Prompt sizes and content
- Plot colors and styles
- Output directory
- Additional metrics

### Requirements

- Python 3.8+
- pandas
- matplotlib
- seaborn
- numpy
- lopace (installed)


# Validation Scripts

This directory also contains scripts for validating the correctness of LoPace
compression methods.

## validate_compression_methods_hf.py

Validates round-trip integrity (compress → decompress) for all supported
compression methods using a public Hugging Face dataset.

### What it does

- Tests all compression methods:
  - Zstd
  - Token-based
  - Hybrid
- Uses the `argilla/prompt-collective` dataset for realistic prompts
- Verifies lossless compression using SHA-256 hashes
- Reports:
  - Total row count
  - Success count
  - Failure count
- Generates:
  - Console summary table

### Usage

```bash
pip install -e .
pip install -r requirements-dev.txt
python scripts/validate_compression_methods_hf.py
```