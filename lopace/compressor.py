"""
Main compression module implementing Zstd, Token-based, and Hybrid compression methods.

The compression algorithms used:
- Zstd: Uses LZ77 (sliding window) and FSE (Finite State Entropy, a variant of Huffman coding)
  internally via the zstandard library
- Token-based: Uses BPE (Byte-Pair Encoding) via tiktoken
- Hybrid: Combines tokenization + Zstd compression
"""

import struct
import math
from collections import Counter
from enum import Enum
from typing import Union, Tuple, Optional, Dict

try:
    import zstandard as zstd
except ImportError:
    zstd = None

try:
    import tiktoken
except ImportError:
    tiktoken = None


class CompressionMethod(Enum):
    """Compression methods available."""
    ZSTD = "zstd"
    TOKEN = "token"
    HYBRID = "hybrid"


class PromptCompressor:
    """
    Professional prompt compressor supporting multiple compression techniques.
    
    Methods:
        - Zstd: Dictionary-based compression using Zstandard
        - Token: Byte-Pair Encoding (BPE) tokenization with binary packing
        - Hybrid: Combination of tokenization and Zstd compression
    
    Args:
        model: Tokenizer model name (default: "cl100k_base")
            Options: "cl100k_base", "p50k_base", "r50k_base", "gpt2", etc.
        zstd_level: Zstd compression level (1-22, default: 15)
            Higher levels provide better compression but are slower.
    """
    
    def __init__(
        self, 
        model: str = "cl100k_base",
        zstd_level: int = 15
    ):
        if zstd is None:
            raise ImportError(
                "zstandard is required. Install it with: pip install zstandard"
            )
        
        if tiktoken is None:
            raise ImportError(
                "tiktoken is required. Install it with: pip install tiktoken"
            )
        
        self.tokenizer = tiktoken.get_encoding(model)
        self.zstd_level = zstd_level
        self.model = model
        
        # Validate zstd_level
        if not (1 <= zstd_level <= 22):
            raise ValueError("zstd_level must be between 1 and 22")
    
    def compress_zstd(self, text: str) -> bytes:
        """
        Compress prompt using Zstandard algorithm.
        
        Args:
            text: Original prompt string
            
        Returns:
            Compressed bytes
            
        Example:
            >>> compressor = PromptCompressor()
            >>> compressed = compressor.compress_zstd("Your prompt here")
            >>> original = compressor.decompress_zstd(compressed)
        """
        data_bytes = text.encode('utf-8')
        compressed_blob = zstd.compress(data_bytes, level=self.zstd_level)
        return compressed_blob
    
    def decompress_zstd(self, compressed_blob: bytes) -> str:
        """
        Decompress Zstandard-compressed prompt.
        
        Args:
            compressed_blob: Compressed bytes from compress_zstd()
            
        Returns:
            Original prompt string
        """
        raw_bytes = zstd.decompress(compressed_blob)
        return raw_bytes.decode('utf-8')
    
    def compress_token(self, text: str) -> bytes:
        """
        Compress prompt using BPE tokenization and binary packing.
        
        This method:
        1. Converts text to token IDs using the tokenizer
        2. Packs token IDs as unsigned integers (uint16 or uint32)
           - Uses uint16 (2 bytes) if all token IDs <= 65535
           - Uses uint32 (4 bytes) if any token ID > 65535
        
        Args:
            text: Original prompt string
            
        Returns:
            Compressed bytes (format byte + binary-packed token IDs)
            Format: [1 byte format flag: 0=uint16, 1=uint32][packed token IDs]
            
        Example:
            >>> compressor = PromptCompressor()
            >>> compressed = compressor.compress_token("Your prompt here")
            >>> original = compressor.decompress_token(compressed)
        """
        # Step 1: Convert text to list of token IDs
        token_ids = list(self.tokenizer.encode(text))  # Ensure it's a list
        
        if not token_ids:
            # Empty token list - return just format byte
            return struct.pack('B', 0)  # uint16 format
        
        # Step 2: Determine if we can use uint16 or need uint32
        # Check if ANY token ID exceeds uint16 range (0-65535)
        max_token_id = max(token_ids)
        min_token_id = min(token_ids)
        use_uint32 = (max_token_id > 65535) or (min_token_id < 0)
        
        # Step 3: Pack token IDs (format byte + token data)
        format_byte = 1 if use_uint32 else 0  # 0 = uint16, 1 = uint32
        
        try:
            if use_uint32:
                # Use uint32 (4 bytes per token) - format 'I'
                token_data = struct.pack(f'{len(token_ids)}I', *token_ids)
            else:
                # Use uint16 (2 bytes per token) - format 'H'
                # Double-check all IDs fit in uint16 range
                if max(token_ids) > 65535:
                    # Fallback to uint32 if somehow we got here
                    format_byte = 1
                    token_data = struct.pack(f'{len(token_ids)}I', *token_ids)
                else:
                    token_data = struct.pack(f'{len(token_ids)}H', *token_ids)
        except (struct.error, OverflowError) as e:
            # If packing fails, fallback to uint32
            format_byte = 1
            token_data = struct.pack(f'{len(token_ids)}I', *token_ids)
        
        # Combine format byte with token data
        binary_payload = struct.pack('B', format_byte) + token_data
        
        return binary_payload
    
    def decompress_token(self, binary_payload: bytes) -> str:
        """
        Decompress token-based compressed prompt.
        
        Args:
            binary_payload: Compressed bytes from compress_token()
            Format: [1 byte format flag: 0=uint16, 1=uint32][packed token IDs]
            
        Returns:
            Original prompt string
        """
        if len(binary_payload) < 1:
            raise ValueError("Invalid compressed data: missing format byte")
        
        # Step 1: Read format byte
        format_byte = struct.unpack('B', binary_payload[0:1])[0]
        token_data = binary_payload[1:]
        
        if format_byte == 1:
            # uint32 format (4 bytes per token)
            if len(token_data) % 4 != 0:
                raise ValueError("Invalid compressed data: uint32 data length not divisible by 4")
            num_tokens = len(token_data) // 4
            token_ids = struct.unpack(f'{num_tokens}I', token_data)
        else:
            # uint16 format (2 bytes per token)
            if len(token_data) % 2 != 0:
                raise ValueError("Invalid compressed data: uint16 data length not divisible by 2")
            num_tokens = len(token_data) // 2
            token_ids = struct.unpack(f'{num_tokens}H', token_data)
        
        # Step 2: Decode token IDs back to string
        return self.tokenizer.decode(list(token_ids))
    
    def compress_hybrid(self, text: str) -> bytes:
        """
        Compress prompt using hybrid approach (Token + Zstd).
        
        This is the most efficient method:
        1. Tokenizes text to reduce redundancy
        2. Packs tokens as binary
        3. Applies Zstd compression on the binary data
        
        Provides the best compression ratio for database storage.
        
        Args:
            text: Original prompt string
            
        Returns:
            Compressed bytes
            
        Example:
            >>> compressor = PromptCompressor()
            >>> compressed = compressor.compress_hybrid("Your prompt here")
            >>> original = compressor.decompress_hybrid(compressed)
        """
        # Step 1: Tokenize
        tokens = list(self.tokenizer.encode(text))  # Ensure it's a list
        
        if not tokens:
            # Empty token list - return compressed empty data
            empty_data = struct.pack('B', 0)  # uint16 format
            return zstd.compress(empty_data, level=self.zstd_level)
        
        # Step 2: Convert to binary (determine uint16 or uint32)
        max_token_id = max(tokens)
        min_token_id = min(tokens)
        use_uint32 = (max_token_id > 65535) or (min_token_id < 0)
        
        format_byte = 1 if use_uint32 else 0  # 0 = uint16, 1 = uint32
        
        try:
            if use_uint32:
                # Use uint32 (4 bytes per token)
                token_data = struct.pack('B', format_byte) + struct.pack(f'{len(tokens)}I', *tokens)
            else:
                # Use uint16 (2 bytes per token)
                # Double-check all IDs fit in uint16 range
                if max(tokens) > 65535:
                    # Fallback to uint32 if somehow we got here
                    format_byte = 1
                    token_data = struct.pack('B', format_byte) + struct.pack(f'{len(tokens)}I', *tokens)
                else:
                    token_data = struct.pack('B', format_byte) + struct.pack(f'{len(tokens)}H', *tokens)
        except (struct.error, OverflowError) as e:
            # If packing fails, fallback to uint32
            format_byte = 1
            token_data = struct.pack('B', format_byte) + struct.pack(f'{len(tokens)}I', *tokens)
        
        # Step 3: Final Zstd compression
        compressed_blob = zstd.compress(token_data, level=self.zstd_level)
        
        return compressed_blob
    
    def decompress_hybrid(self, blob: bytes) -> str:
        """
        Decompress hybrid-compressed prompt.
        
        Args:
            blob: Compressed bytes from compress_hybrid()
            
        Returns:
            Original prompt string
        """
        # Step 1: Decompress Zstd
        token_data = zstd.decompress(blob)
        
        if len(token_data) < 1:
            raise ValueError("Invalid compressed data: missing format byte")
        
        # Step 2: Read format byte and unpack token IDs
        format_byte = struct.unpack('B', token_data[0:1])[0]
        packed_data = token_data[1:]
        
        if format_byte == 1:
            # uint32 format (4 bytes per token)
            if len(packed_data) % 4 != 0:
                raise ValueError("Invalid compressed data: uint32 data length not divisible by 4")
            num_tokens = len(packed_data) // 4
            tokens = struct.unpack(f'{num_tokens}I', packed_data)
        else:
            # uint16 format (2 bytes per token)
            if len(packed_data) % 2 != 0:
                raise ValueError("Invalid compressed data: uint16 data length not divisible by 2")
            num_tokens = len(packed_data) // 2
            tokens = struct.unpack(f'{num_tokens}H', packed_data)
        
        # Step 3: Decode to text
        return self.tokenizer.decode(list(tokens))
    
    def compress(
        self, 
        text: str, 
        method: CompressionMethod = CompressionMethod.HYBRID
    ) -> bytes:
        """
        Compress prompt using the specified method.
        
        Args:
            text: Original prompt string
            method: Compression method to use (default: HYBRID)
            
        Returns:
            Compressed bytes
            
        Example:
            >>> compressor = PromptCompressor()
            >>> compressed = compressor.compress("Your prompt", CompressionMethod.HYBRID)
            >>> original = compressor.decompress(compressed, CompressionMethod.HYBRID)
        """
        if method == CompressionMethod.ZSTD:
            return self.compress_zstd(text)
        elif method == CompressionMethod.TOKEN:
            return self.compress_token(text)
        elif method == CompressionMethod.HYBRID:
            return self.compress_hybrid(text)
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def decompress(
        self, 
        compressed_data: bytes, 
        method: CompressionMethod = CompressionMethod.HYBRID
    ) -> str:
        """
        Decompress prompt using the specified method.
        
        Args:
            compressed_data: Compressed bytes
            method: Compression method used for compression
            
        Returns:
            Original prompt string
        """
        if method == CompressionMethod.ZSTD:
            return self.decompress_zstd(compressed_data)
        elif method == CompressionMethod.TOKEN:
            return self.decompress_token(compressed_data)
        elif method == CompressionMethod.HYBRID:
            return self.decompress_hybrid(compressed_data)
        else:
            raise ValueError(f"Unknown compression method: {method}")
    
    def compress_and_return_both(
        self, 
        text: str, 
        method: CompressionMethod = CompressionMethod.HYBRID
    ) -> Tuple[str, bytes]:
        """
        Compress prompt and return both original and compressed versions.
        
        Args:
            text: Original prompt string
            method: Compression method to use (default: HYBRID)
            
        Returns:
            Tuple of (original_prompt, compressed_bytes)
            
        Example:
            >>> compressor = PromptCompressor()
            >>> original, compressed = compressor.compress_and_return_both("Your prompt")
        """
        compressed = self.compress(text, method)
        return (text, compressed)
    
    def get_compression_stats(
        self, 
        text: str, 
        method: Optional[CompressionMethod] = None
    ) -> dict:
        """
        Get compression statistics for a given prompt.
        
        Args:
            text: Original prompt string
            method: Compression method to analyze (None = all methods)
            
        Returns:
            Dictionary with compression statistics
        """
        methods = [method] if method else [
            CompressionMethod.ZSTD,
            CompressionMethod.TOKEN,
            CompressionMethod.HYBRID
        ]
        
        original_size = len(text.encode('utf-8'))
        stats = {
            'original_size_bytes': original_size,
            'original_size_tokens': len(self.tokenizer.encode(text)),
            'methods': {}
        }
        
        for m in methods:
            compressed = self.compress(text, m)
            compressed_size = len(compressed)
            compression_ratio = compressed_size / original_size if original_size > 0 else 0
            space_saved = 1 - compression_ratio
            
            stats['methods'][m.value] = {
                'compressed_size_bytes': compressed_size,
                'compression_ratio': compression_ratio,
                'space_saved_percent': space_saved * 100,
                'bytes_saved': original_size - compressed_size
            }
        
        return stats
    
    def calculate_shannon_entropy(self, text: str, unit: str = 'character') -> float:
        """
        Calculate Shannon Entropy of the input text.
        
        Shannon Entropy formula: H(X) = -∑ P(x_i) * log₂(P(x_i))
        
        This determines the theoretical lower limit of compression based on
        character/byte frequency distribution.
        
        Args:
            text: Input text to analyze
            unit: Unit to analyze ('character' or 'byte')
                - 'character': Analyze individual characters
                - 'byte': Analyze bytes (for binary data)
            
        Returns:
            Shannon entropy in bits
            
        Example:
            >>> compressor = PromptCompressor()
            >>> entropy = compressor.calculate_shannon_entropy("Hello world")
            >>> print(f"Theoretical compression limit: {entropy:.2f} bits per character")
        """
        if not text:
            return 0.0
        
        if unit == 'byte':
            # Analyze bytes
            data = text.encode('utf-8')
            frequencies = Counter(data)
        else:  # unit == 'character'
            # Analyze characters
            frequencies = Counter(text)
        
        # Calculate probabilities
        length = len(text) if unit == 'character' else len(data)
        probabilities = [count / length for count in frequencies.values()]
        
        # Calculate Shannon Entropy: H(X) = -∑ P(x_i) * log₂(P(x_i))
        entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
        
        return entropy
    
    def get_theoretical_compression_limit(self, text: str, unit: str = 'character') -> Dict[str, float]:
        """
        Calculate theoretical compression limit using Shannon Entropy.
        
        This provides the theoretical minimum size achievable through entropy coding.
        
        Args:
            text: Input text to analyze
            unit: Unit to analyze ('character' or 'byte')
            
        Returns:
            Dictionary with theoretical limits:
            - entropy_bits_per_unit: Shannon entropy in bits
            - theoretical_min_bits: Minimum total bits needed
            - theoretical_min_bytes: Minimum bytes needed (theoretical limit)
            - original_size_bytes: Original size in bytes
            - theoretical_compression_ratio: Theoretical best compression ratio
            
        Example:
            >>> compressor = PromptCompressor()
            >>> limits = compressor.get_theoretical_compression_limit("Your prompt")
            >>> print(f"Theoretical minimum: {limits['theoretical_min_bytes']:.2f} bytes")
        """
        if not text:
            return {
                'entropy_bits_per_unit': 0.0,
                'theoretical_min_bits': 0.0,
                'theoretical_min_bytes': 0.0,
                'original_size_bytes': 0.0,
                'theoretical_compression_ratio': 0.0
            }
        
        # Calculate Shannon Entropy
        entropy = self.calculate_shannon_entropy(text, unit)
        
        # Calculate theoretical minimums
        if unit == 'byte':
            num_units = len(text.encode('utf-8'))
            original_size_bytes = len(text.encode('utf-8'))
        else:  # character
            num_units = len(text)
            original_size_bytes = len(text.encode('utf-8'))
        
        theoretical_min_bits = entropy * num_units
        theoretical_min_bytes = theoretical_min_bits / 8.0
        
        # Theoretical compression ratio
        theoretical_compression_ratio = (
            theoretical_min_bytes / original_size_bytes 
            if original_size_bytes > 0 else 0.0
        )
        
        return {
            'entropy_bits_per_unit': entropy,
            'theoretical_min_bits': theoretical_min_bits,
            'theoretical_min_bytes': theoretical_min_bytes,
            'original_size_bytes': original_size_bytes,
            'theoretical_compression_ratio': theoretical_compression_ratio,
            'theoretical_space_savings_percent': (1 - theoretical_compression_ratio) * 100
        }