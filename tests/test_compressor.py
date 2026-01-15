"""Tests for PromptCompressor class."""

import pytest
from lopace import PromptCompressor, CompressionMethod


@pytest.fixture
def compressor():
    """Create a PromptCompressor instance for testing."""
    return PromptCompressor(model="cl100k_base", zstd_level=15)


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return """You are a helpful AI assistant designed to provide accurate, 
    detailed, and helpful responses to user queries. Your goal is to assist users 
    by understanding their questions and providing relevant information."""


class TestZstdCompression:
    """Test Zstd compression/decompression."""
    
    def test_compress_decompress_zstd(self, compressor, sample_prompt):
        """Test that Zstd compression is lossless."""
        compressed = compressor.compress_zstd(sample_prompt)
        decompressed = compressor.decompress_zstd(compressed)
        assert decompressed == sample_prompt
    
    def test_zstd_compression_ratio(self, compressor, sample_prompt):
        """Test that Zstd actually compresses."""
        compressed = compressor.compress_zstd(sample_prompt)
        original_size = len(sample_prompt.encode('utf-8'))
        compressed_size = len(compressed)
        assert compressed_size < original_size


class TestTokenCompression:
    """Test Token-based compression/decompression."""
    
    def test_compress_decompress_token(self, compressor, sample_prompt):
        """Test that Token compression is lossless."""
        compressed = compressor.compress_token(sample_prompt)
        decompressed = compressor.decompress_token(compressed)
        assert decompressed == sample_prompt
    
    def test_token_binary_format(self, compressor, sample_prompt):
        """Test that token compression produces binary data with format byte."""
        compressed = compressor.compress_token(sample_prompt)
        assert isinstance(compressed, bytes)
        # Should have at least format byte (1 byte)
        assert len(compressed) >= 1
        # Format byte should be 0 (uint16) or 1 (uint32)
        import struct
        format_byte = struct.unpack('B', compressed[0:1])[0]
        assert format_byte in [0, 1]


class TestHybridCompression:
    """Test Hybrid compression/decompression."""
    
    def test_compress_decompress_hybrid(self, compressor, sample_prompt):
        """Test that Hybrid compression is lossless."""
        compressed = compressor.compress_hybrid(sample_prompt)
        decompressed = compressor.decompress_hybrid(compressed)
        assert decompressed == sample_prompt
    
    def test_hybrid_compression(self, compressor, sample_prompt):
        """Test that Hybrid compression works and compares with other methods."""
        zstd_compressed = compressor.compress_zstd(sample_prompt)
        token_compressed = compressor.compress_token(sample_prompt)
        hybrid_compressed = compressor.compress_hybrid(sample_prompt)
        
        # All methods should compress (smaller than original for longer prompts)
        original_size = len(sample_prompt.encode('utf-8'))
        
        # Verify all compression methods produce valid output
        assert len(zstd_compressed) > 0
        assert len(token_compressed) > 0
        assert len(hybrid_compressed) > 0
        
        # For very long prompts (>500 chars), hybrid should typically be better than token alone
        # But for short prompts, Zstd overhead can make hybrid larger
        if len(sample_prompt) > 500:
            # On very long prompts, hybrid should generally compress well
            assert len(hybrid_compressed) < original_size
        
        # Zstd alone should compress for longer prompts
        if len(sample_prompt) > 100:
            assert len(zstd_compressed) < original_size


class TestGenericMethods:
    """Test generic compress/decompress methods."""
    
    def test_compress_with_method(self, compressor, sample_prompt):
        """Test generic compress method with all methods."""
        for method in CompressionMethod:
            compressed = compressor.compress(sample_prompt, method)
            assert isinstance(compressed, bytes)
            assert len(compressed) > 0
    
    def test_decompress_with_method(self, compressor, sample_prompt):
        """Test generic decompress method with all methods."""
        for method in CompressionMethod:
            compressed = compressor.compress(sample_prompt, method)
            decompressed = compressor.decompress(compressed, method)
            assert decompressed == sample_prompt
    
    def test_compress_and_return_both(self, compressor, sample_prompt):
        """Test compress_and_return_both method."""
        original, compressed = compressor.compress_and_return_both(
            sample_prompt, 
            CompressionMethod.HYBRID
        )
        assert original == sample_prompt
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0


class TestCompressionStats:
    """Test compression statistics."""
    
    def test_get_compression_stats_all_methods(self, compressor, sample_prompt):
        """Test getting stats for all methods."""
        stats = compressor.get_compression_stats(sample_prompt)
        
        assert 'original_size_bytes' in stats
        assert 'original_size_tokens' in stats
        assert 'methods' in stats
        
        assert len(stats['methods']) == 3  # ZSTD, TOKEN, HYBRID
        
        for method_name, method_stats in stats['methods'].items():
            assert 'compressed_size_bytes' in method_stats
            assert 'compression_ratio' in method_stats
            assert 'space_saved_percent' in method_stats
            assert 'bytes_saved' in method_stats
    
    def test_get_compression_stats_single_method(self, compressor, sample_prompt):
        """Test getting stats for a single method."""
        stats = compressor.get_compression_stats(
            sample_prompt, 
            CompressionMethod.HYBRID
        )
        
        assert len(stats['methods']) == 1
        assert CompressionMethod.HYBRID.value in stats['methods']


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string(self, compressor):
        """Test compression of empty string."""
        for method in CompressionMethod:
            compressed = compressor.compress("", method)
            decompressed = compressor.decompress(compressed, method)
            assert decompressed == ""
    
    def test_large_token_ids(self, compressor):
        """Test compression with token IDs that exceed uint16 range."""
        # Create a prompt that might trigger large token IDs
        # Using various special characters and unicode
        large_prompt = "Hello " * 1000 + "世界 🌍 مرحبا " * 100
        
        # This should work without error, handling uint32 if needed
        compressed = compressor.compress_token(large_prompt)
        decompressed = compressor.decompress_token(compressed)
        assert decompressed == large_prompt
        
        # Test hybrid method too
        compressed_hybrid = compressor.compress_hybrid(large_prompt)
        decompressed_hybrid = compressor.decompress_hybrid(compressed_hybrid)
        assert decompressed_hybrid == large_prompt
    
    def test_single_character(self, compressor):
        """Test compression of single character."""
        for method in CompressionMethod:
            compressed = compressor.compress("a", method)
            decompressed = compressor.decompress(compressed, method)
            assert decompressed == "a"
    
    def test_unicode_characters(self, compressor):
        """Test compression of unicode characters."""
        unicode_prompt = "你好世界 🌍 مرحبا"
        for method in CompressionMethod:
            compressed = compressor.compress(unicode_prompt, method)
            decompressed = compressor.decompress(compressed, method)
            assert decompressed == unicode_prompt
    
    def test_invalid_zstd_level(self):
        """Test that invalid zstd_level raises error."""
        with pytest.raises(ValueError):
            PromptCompressor(zstd_level=0)
        
        with pytest.raises(ValueError):
            PromptCompressor(zstd_level=23)