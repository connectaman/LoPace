import hashlib
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from lopace import PromptCompressor, CompressionMethod


def sha256_hex(text: str) -> str:
    """
    Compute SHA-256 hash of a string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_method_validation(
    compressor: PromptCompressor,
    method: CompressionMethod,
    prompts: list[str],
) -> tuple[int, int, int]:
    """
    Compress + decompress prompts using a given method.
    Returns (row_count, success_count, failure_count).
    """
    success = 0
    failure = 0

    for prompt in tqdm(prompts, desc=f"Testing {method.name}", leave=False):
        original_hash = sha256_hex(prompt)

        compressed = compressor.compress(prompt, method)
        decompressed = compressor.decompress(compressed, method)

        decompressed_hash = sha256_hex(decompressed)

        if original_hash == decompressed_hash:
            success += 1
        else:
            failure += 1

    return len(prompts), success, failure


def main():
    # -----------------------------
    # Load Hugging Face dataset
    # -----------------------------
    dataset = load_dataset("argilla/prompt-collective", split="train")

    # Extract prompt column
    prompts = dataset["prompt"]

    # 🔹 OPTIONAL: limit rows for quicker testing
    # prompts = prompts[:1000]

    print(f"\nLoaded {len(prompts)} prompts from Hugging Face dataset\n")

    # Initialize compressor
    compressor = PromptCompressor()

    methods = [
        ("Zstd", CompressionMethod.ZSTD),
        ("Token", CompressionMethod.TOKEN),
        ("Hybrid", CompressionMethod.HYBRID),
    ]

    results = []

    for name, method in methods:
        row_count, success, failure = run_method_validation(
            compressor, method, prompts
        )

        results.append(
            {
                "Method": name,
                "Row Count": row_count,
                "Success Count": success,
                "Failure Count": failure,
            }
        )

    # -----------------------------
    # Display summary
    # -----------------------------
    df = pd.DataFrame(results)

    print("\nCompression Integrity Summary (HuggingFace Dataset):\n")
    print(df.to_string(index=False))



if __name__ == "__main__":
    main()
