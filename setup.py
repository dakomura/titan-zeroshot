from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="titan-zeroshot",
    version="0.1.0",
    author="Daisuke Komura",
    author_email="kdais-prm@m.u-tokyo.ac.jp",
    description="Titan特徴量の分析ツール（UMAP可視化とk-NN分類）",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dakomura/titan-zeroshot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "titan-zeroshot=titan_zeroshot.main:main",
        ],
    },
) 