"""
LoPace - Lossless Optimized Prompt Accurate Compression Engine

A professional Python package for compressing and decompressing prompts
using multiple techniques: Zstd, Token-based (BPE), and Hybrid methods.
"""

from .compressor import PromptCompressor, CompressionMethod

__version__ = "0.1.0"
__all__ = ["PromptCompressor", "CompressionMethod"]