# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure
- GitHub Actions CI/CD workflow
- Contributing guidelines
- Code of Conduct

## [0.1.0] - 2024-12-XX

### Added
- **Core Features**:
  - Zstd compression method
  - Token-based (BPE) compression method
  - Hybrid compression method (Token + Zstd)
  - Lossless compression/decompression

- **Compression Methods**:
  - `compress_zstd()` / `decompress_zstd()` - Zstandard compression
  - `compress_token()` / `decompress_token()` - BPE tokenization with binary packing
  - `compress_hybrid()` / `decompress_hybrid()` - Combined tokenization and Zstd
  - Generic `compress()` / `decompress()` methods

- **Evaluation Metrics**:
  - Compression Ratio (CR)
  - Space Savings (SS)
  - Bits Per Character (BPC)
  - Throughput (MB/s)
  - SHA-256 hash verification
  - Exact match verification
  - Reconstruction error calculation
  - Shannon Entropy calculation
  - Theoretical compression limits

- **API**:
  - `PromptCompressor` class with configurable tokenizer and Zstd level
  - `CompressionMethod` enum for method selection
  - `compress_and_return_both()` method
  - `get_compression_stats()` method
  - `calculate_shannon_entropy()` method
  - `get_theoretical_compression_limit()` method

- **Streamlit Web App**:
  - Interactive compression interface
  - Real-time metrics calculation
  - Side-by-side method comparison
  - Comprehensive evaluation dashboard

- **Documentation**:
  - Comprehensive README with usage examples
  - API reference documentation
  - Mathematical background explanation
  - Installation instructions

- **Testing**:
  - Complete test suite with pytest
  - Test coverage for all compression methods
  - Edge case testing
  - Lossless verification tests

- **DevOps**:
  - GitHub Actions CI/CD pipeline
  - Automated testing on multiple Python versions
  - Automated PyPI publishing

### Technical Details

- **Supported Tokenizers**: cl100k_base, p50k_base, r50k_base, gpt2
- **Zstd Levels**: 1-22 (default: 15)
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Dependencies**: zstandard>=0.22.0, tiktoken>=0.5.0
- **Smart Format Detection**: Automatically uses uint16 or uint32 based on token ID ranges

### Fixed
- Token ID overflow handling for tokenizers with vocab > 65535
- Format byte handling for backward compatibility
- Error handling for edge cases

[Unreleased]: https://github.com/amanulla/lopace/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/amanulla/lopace/releases/tag/v0.1.0