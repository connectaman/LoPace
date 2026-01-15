"""
Streamlit app for LoPace - Interactive Prompt Compression with Evaluation Metrics
"""

import streamlit as st
import hashlib
import time
from typing import Dict, Any, List, Tuple
from lopace import PromptCompressor, CompressionMethod


def calculate_metrics(
    original_text: str,
    compressed_data: bytes,
    compression_time: float,
    decompression_time: float,
    decompressed_text: str,
    compressor: PromptCompressor = None
) -> Dict[str, Any]:
    """
    Calculate all evaluation metrics for compression.
    
    Args:
        compressor: PromptCompressor instance for Shannon Entropy calculation
    
    Returns:
        Dictionary with all metrics
    """
    original_size_bytes = len(original_text.encode('utf-8'))
    compressed_size_bytes = len(compressed_data)
    original_size_bits = original_size_bytes * 8
    compressed_size_bits = compressed_size_bytes * 8
    num_characters = len(original_text)
    
    # Compression Ratio (CR)
    compression_ratio = original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0
    
    # Space Savings (SS)
    space_savings = (1 - (compressed_size_bytes / original_size_bytes)) * 100 if original_size_bytes > 0 else 0
    
    # Bits Per Character (BPC)
    bits_per_character = compressed_size_bits / num_characters if num_characters > 0 else 0
    
    # Throughput (MB/s)
    compression_throughput = (original_size_bytes / (1024 * 1024)) / compression_time if compression_time > 0 else 0
    decompression_throughput = (compressed_size_bytes / (1024 * 1024)) / decompression_time if decompression_time > 0 else 0
    
    # SHA-256 Hash
    original_hash = hashlib.sha256(original_text.encode('utf-8')).hexdigest()
    decompressed_hash = hashlib.sha256(decompressed_text.encode('utf-8')).hexdigest()
    hash_match = original_hash == decompressed_hash
    
    # Exact Match (Fidelity)
    exact_match = original_text == decompressed_text
    
    # Reconstruction Error
    reconstruction_error = 0.0 if exact_match else 1.0
    
    # Shannon Entropy (if compressor provided)
    shannon_entropy = None
    theoretical_min_bytes = None
    theoretical_compression_ratio = None
    if compressor:
        try:
            shannon_entropy = compressor.calculate_shannon_entropy(original_text)
            limits = compressor.get_theoretical_compression_limit(original_text)
            theoretical_min_bytes = limits['theoretical_min_bytes']
            theoretical_compression_ratio = limits['theoretical_compression_ratio']
        except Exception:
            pass
    
    return {
        'original_size_bytes': original_size_bytes,
        'compressed_size_bytes': compressed_size_bytes,
        'original_size_bits': original_size_bits,
        'compressed_size_bits': compressed_size_bits,
        'num_characters': num_characters,
        'compression_ratio': compression_ratio,
        'space_savings': space_savings,
        'bits_per_character': bits_per_character,
        'compression_throughput': compression_throughput,
        'decompression_throughput': decompression_throughput,
        'compression_time': compression_time,
        'decompression_time': decompression_time,
        'original_hash': original_hash,
        'decompressed_hash': decompressed_hash,
        'hash_match': hash_match,
        'exact_match': exact_match,
        'reconstruction_error': reconstruction_error,
        'shannon_entropy': shannon_entropy,
        'theoretical_min_bytes': theoretical_min_bytes,
        'theoretical_compression_ratio': theoretical_compression_ratio,
    }


def format_hash(hash_str: str) -> str:
    """Format hash for display."""
    return f"{hash_str[:16]}...{hash_str[-16:]}"


def format_bytes(data: bytes, max_display: int = 500) -> str:
    """Format bytes for display with hex representation."""
    if len(data) <= max_display:
        hex_str = data.hex()
        # Add space every 2 characters for readability
        return ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
    else:
        preview_data = data[:max_display]
        hex_str = preview_data.hex()
        preview_formatted = ' '.join(hex_str[i:i+2] for i in range(0, len(hex_str), 2))
        return f"{preview_formatted} ... (truncated, {len(data)} total bytes)"


def main():
    st.set_page_config(
        page_title="LoPace - Prompt Compression",
        page_icon="🗜️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .data-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🗜️ LoPace</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Lossless Optimized Prompt Accurate Compression Engine</div>', unsafe_allow_html=True)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        tokenizer_model = st.selectbox(
            "Tokenizer Model",
            options=["cl100k_base", "p50k_base", "r50k_base", "gpt2"],
            index=0,
            help="BPE tokenizer model for token-based compression"
        )
        
        zstd_level = st.slider(
            "Zstd Compression Level",
            min_value=1,
            max_value=22,
            value=15,
            help="Higher values = better compression but slower (1-22)"
        )
        
        st.markdown("---")
        st.markdown("### 📊 About Metrics")
        st.info("""
        **Compression Ratio (CR)**: How many times smaller (e.g., 4.5x)
        
        **Space Savings (SS)**: Percentage of space reduced (e.g., 75%)
        
        **Bits Per Character (BPC)**: Average bits to store one character
        
        **Throughput**: Speed in MB/s for compression/decompression
        
        **Hash Match**: SHA-256 verification of losslessness
        
        **Exact Match**: Character-by-character comparison
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Compression Methods")
        st.caption("""
        - **Zstd**: Dictionary-based compression
        - **Token**: BPE tokenization with binary packing
        - **Hybrid**: Token + Zstd (recommended)
        """)
    
    # Main content area - Two column layout
    col_left, col_right = st.columns([1, 1], gap="large")
    
    with col_left:
        st.markdown("### 📝 Input Prompt")
        default_prompt = """You are a helpful AI assistant designed to provide accurate, 
detailed, and helpful responses to user queries. Your goal is to assist users 
by understanding their questions and providing relevant information, explanations, 
or guidance. Always be respectful, clear, and concise in your communications. 
If you are uncertain about something, it's better to acknowledge that uncertainty 
rather than provide potentially incorrect information."""
        
        input_prompt = st.text_area(
            "Enter your prompt:",
            value=default_prompt,
            height=400,
            help="Enter the system prompt or any text you want to compress",
            label_visibility="collapsed",
            key="input_prompt_textarea"
        )
        
        # Character and byte count
        char_count = len(input_prompt)
        byte_count = len(input_prompt.encode('utf-8'))
        st.caption(f"📏 {char_count:,} characters | {byte_count:,} bytes")
        
        compress_button = st.button("🗜️ Compress & Analyze", type="primary", use_container_width=True)
    
    with col_right:
        st.markdown("### 📦 Compressed & Decompressed Data")
        
        if not compress_button:
            st.info("👈 Enter a prompt on the left and click **'Compress & Analyze'** to see compression results")
        elif not input_prompt.strip():
            st.warning("⚠️ Please enter a prompt to compress")
        else:
            try:
                # Initialize compressor
                compressor = PromptCompressor(model=tokenizer_model, zstd_level=zstd_level)
                
                # Process all methods
                methods = [
                    CompressionMethod.ZSTD,
                    CompressionMethod.TOKEN,
                    CompressionMethod.HYBRID
                ]
                
                method_names = {
                    CompressionMethod.ZSTD: "Zstd",
                    CompressionMethod.TOKEN: "Token (BPE)",
                    CompressionMethod.HYBRID: "Hybrid (Recommended)"
                }
                
                method_icons = {
                    CompressionMethod.ZSTD: "🔵",
                    CompressionMethod.TOKEN: "🟢",
                    CompressionMethod.HYBRID: "🟣"
                }
                
                # Store results for metrics section
                all_results: Dict[str, Dict[str, Any]] = {}
                all_metrics: Dict[str, Dict[str, Any]] = {}
                
                # Create tabs for each method
                tabs = st.tabs([f"{method_icons[m]} {method_names[m]}" for m in methods])
                
                for tab, method in zip(tabs, methods):
                    with tab:
                        # Compress and measure time
                        start_compress = time.perf_counter()
                        compressed = compressor.compress(input_prompt, method)
                        compression_time = time.perf_counter() - start_compress
                        
                        # Decompress and measure time
                        start_decompress = time.perf_counter()
                        decompressed = compressor.decompress(compressed, method)
                        decompression_time = time.perf_counter() - start_decompress
                        
                        # Calculate metrics
                        metrics = calculate_metrics(
                            input_prompt,
                            compressed,
                            compression_time,
                            decompression_time,
                            decompressed,
                            compressor=compressor
                        )
                        
                        all_results[method.value] = {
                            'compressed': compressed,
                            'decompressed': decompressed,
                            'method_name': method_names[method]
                        }
                        all_metrics[method.value] = metrics
                        
                        # Display compressed data
                        st.markdown("#### 🔐 Compressed Data (Hex)")
                        with st.container():
                            st.markdown('<div class="data-box">', unsafe_allow_html=True)
                            st.code(format_bytes(compressed, max_display=1000), language="text")
                            st.caption(f"Size: {len(compressed):,} bytes | Showing first 1000 bytes")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display decompressed data
                        st.markdown("#### 🔓 Decompressed Data (Original Text)")
                        with st.container():
                            st.markdown('<div class="data-box">', unsafe_allow_html=True)
                            st.text_area(
                                "Decompressed text:",
                                value=decompressed,
                                height=300,
                                disabled=True,
                                label_visibility="collapsed",
                                key=f"decompressed_text_{method.value}"
                            )
                            st.caption(f"✅ Lossless: {'Verified' if metrics['exact_match'] else 'FAILED'}")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Quick verification status
                        if metrics['exact_match'] and metrics['hash_match']:
                            st.success("✅ **Lossless Verification**: All checks passed!")
                        else:
                            st.error("❌ **Lossless Verification**: Failed!")
                
                # Store results in session state for metrics section
                st.session_state['all_results'] = all_results
                st.session_state['all_metrics'] = all_metrics
                st.session_state['input_prompt'] = input_prompt
                st.session_state['compressor'] = compressor
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)
    
    # Metrics Section - Below the two columns
    if compress_button and 'all_metrics' in st.session_state:
        st.markdown("---")
        st.markdown("## 📊 Comprehensive Evaluation Metrics")
        
        all_metrics = st.session_state['all_metrics']
        all_results = st.session_state['all_results']
        methods = [
            CompressionMethod.ZSTD,
            CompressionMethod.TOKEN,
            CompressionMethod.HYBRID
        ]
        
        method_names = {
            CompressionMethod.ZSTD: "Zstd",
            CompressionMethod.TOKEN: "Token (BPE)",
            CompressionMethod.HYBRID: "Hybrid (Recommended)"
        }
        
        # Primary Evaluation Metrics
        st.markdown("### 📈 Primary Evaluation Metrics")
        
        for method in methods:
            metrics = all_metrics[method.value]
            method_name = method_names[method]
            
            with st.expander(f"📊 {method_name} - Detailed Metrics", expanded=(method == CompressionMethod.HYBRID)):
                # Create metric columns
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Compression Ratio (CR)",
                        f"{metrics['compression_ratio']:.2f}x",
                        help="$CR = \\frac{S_{original}}{S_{compressed}}$"
                    )
                
                with col2:
                    st.metric(
                        "Space Savings (SS)",
                        f"{metrics['space_savings']:.2f}%",
                        help="$SS = 1 - \\frac{S_{compressed}}{S_{original}}$"
                    )
                
                with col3:
                    st.metric(
                        "Bits Per Character (BPC)",
                        f"{metrics['bits_per_character']:.2f}",
                        help="$BPC = \\frac{Total Bits}{Total Characters}$"
                    )
                
                with col4:
                    st.metric(
                        "Compression Time",
                        f"{metrics['compression_time']*1000:.2f} ms"
                    )
                
                # Throughput
                st.markdown("#### ⚡ Throughput")
                throughput_col1, throughput_col2 = st.columns(2)
                
                with throughput_col1:
                    st.metric(
                        "Compression Throughput",
                        f"{metrics['compression_throughput']:.2f} MB/s",
                        help="$T = \\frac{Data Size}{Time}$"
                    )
                
                with throughput_col2:
                    st.metric(
                        "Decompression Throughput",
                        f"{metrics['decompression_throughput']:.2f} MB/s"
                    )
                
                # Size Information
                st.markdown("#### 💾 Size Information")
                size_col1, size_col2, size_col3 = st.columns(3)
                
                with size_col1:
                    st.metric("Original Size", f"{metrics['original_size_bytes']:,} bytes")
                
                with size_col2:
                    st.metric("Compressed Size", f"{metrics['compressed_size_bytes']:,} bytes")
                
                with size_col3:
                    bytes_saved = metrics['original_size_bytes'] - metrics['compressed_size_bytes']
                    st.metric("Bytes Saved", f"{bytes_saved:,}", delta=f"{metrics['space_savings']:.1f}%")
                
                # Lossless Verification
                st.markdown("#### ✅ Lossless Verification")
                
                # SHA-256 Hash Verification
                hash_col1, hash_col2 = st.columns(2)
                
                with hash_col1:
                    st.markdown("**Original Hash (SHA-256)**")
                    st.code(format_hash(metrics['original_hash']), language="text")
                
                with hash_col2:
                    st.markdown("**Decompressed Hash (SHA-256)**")
                    st.code(format_hash(metrics['decompressed_hash']), language="text")
                
                # Verification Status
                verif_col1, verif_col2 = st.columns(2)
                
                with verif_col1:
                    if metrics['hash_match']:
                        st.success("✅ **Hash Match**: SHA-256 hashes are identical")
                    else:
                        st.error("❌ **Hash Mismatch**: Hashes do not match!")
                
                with verif_col2:
                    if metrics['exact_match']:
                        st.success("✅ **Exact Match**: Fidelity 100% - All characters match")
                    else:
                        st.error("❌ **Exact Match**: Fidelity 0% - Characters do not match")
                
                # Reconstruction Error
                st.markdown("#### Reconstruction Error")
                if metrics['reconstruction_error'] == 0.0:
                    st.success(f"✅ **Error Rate: 0.0** - Lossless compression verified")
                    st.latex(r"E = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(x_i \neq \hat{x}_i) = 0")
                else:
                    st.error(f"❌ **Error Rate: {metrics['reconstruction_error']:.4f}**")
                
                # Shannon Entropy & Theoretical Limits
                if metrics.get('shannon_entropy') is not None:
                    st.markdown("#### 📐 Shannon Entropy & Theoretical Limits")
                    st.markdown("""
                    **Shannon Entropy** determines the theoretical compression limit:
                    $H(X) = -\\sum_{i=1}^{n} P(x_i) \\log_2 P(x_i)$
                    """)
                    
                    entropy_col1, entropy_col2, entropy_col3 = st.columns(3)
                    
                    with entropy_col1:
                        st.metric(
                            "Shannon Entropy (bits/char)",
                            f"{metrics['shannon_entropy']:.4f}",
                            help="Theoretical bits needed per character"
                        )
                    
                    with entropy_col2:
                        st.metric(
                            "Theoretical Min (bytes)",
                            f"{metrics['theoretical_min_bytes']:.2f}",
                            help="Theoretical minimum size achievable"
                        )
                    
                    with entropy_col3:
                        if metrics['theoretical_compression_ratio']:
                            theoretical_savings = (1 - metrics['theoretical_compression_ratio']) * 100
                            st.metric(
                                "Theoretical Savings",
                                f"{theoretical_savings:.2f}%",
                                help="Best possible space savings"
                            )
                    
                    # Comparison: Actual vs Theoretical
                    actual_vs_theoretical = (
                        metrics['compressed_size_bytes'] / metrics['theoretical_min_bytes']
                        if metrics['theoretical_min_bytes'] and metrics['theoretical_min_bytes'] > 0 
                        else None
                    )
                    
                    if actual_vs_theoretical:
                        st.info(
                            f"📊 **Efficiency**: Actual compression is "
                            f"**{actual_vs_theoretical:.2f}x** the theoretical minimum. "
                            f"Lower is better (1.0x = optimal)."
                        )
        
        # Comparison Table
        st.markdown("### 📊 Method Comparison Table")
        
        comparison_data = {
            'Method': [method_names[m] for m in methods],
            'Compression Ratio (x)': [f"{all_metrics[m.value]['compression_ratio']:.2f}" for m in methods],
            'Space Savings (%)': [f"{all_metrics[m.value]['space_savings']:.2f}" for m in methods],
            'BPC': [f"{all_metrics[m.value]['bits_per_character']:.2f}" for m in methods],
            'Original (bytes)': [f"{all_metrics[m.value]['original_size_bytes']:,}" for m in methods],
            'Compressed (bytes)': [f"{all_metrics[m.value]['compressed_size_bytes']:,}" for m in methods],
            'Compress Speed (MB/s)': [f"{all_metrics[m.value]['compression_throughput']:.2f}" for m in methods],
            'Decompress Speed (MB/s)': [f"{all_metrics[m.value]['decompression_throughput']:.2f}" for m in methods],
            'Lossless': ['✅' if all_metrics[m.value]['hash_match'] and all_metrics[m.value]['exact_match'] else '❌' for m in methods],
        }
        
        st.dataframe(comparison_data, use_container_width=True, hide_index=True)
        
        # Best method recommendation
        best_method = max(methods, key=lambda m: all_metrics[m.value]['compression_ratio'])
        best_ratio = all_metrics[best_method.value]['compression_ratio']
        best_savings = all_metrics[best_method.value]['space_savings']
        
        st.success(
            f"🏆 **Best Compression Method**: **{method_names[best_method]}** "
            f"with **{best_ratio:.2f}x** compression ratio "
            f"({best_savings:.2f}% space savings)"
        )


if __name__ == "__main__":
    main()