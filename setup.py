"""Setup configuration for LoPace package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lopace",
    version="0.1.4",
    author="Aman Ulla",
    description="Lossless Optimized Prompt Accurate Compression Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/connectaman/LoPace",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    keywords="prompt compression, tokenization, zstd, bpe, nlp, optimization",
    project_urls={
        "Bug Reports": "https://github.com/connectaman/LoPace/issues",
        "Source": "https://github.com/connectaman/LoPace",
        "PyPI": "https://pypi.org/project/lopace/",
        "Hugging Face Spaces": "https://huggingface.co/spaces/codewithaman/LoPace",
    },
)