from setuptools import setup, find_packages

requirements = [
    "numpy==1.22.3",
    "matplotlib==3.5.0",
    "torch==1.10.0",
    "gpytorch==1.5.1",
    "scikit-learn==1.0.2",
    "pandas==1.3.4",
    "wbml==0.3.14",
    "wandb==0.12.14",
    "kmeans-pytorch==0.3",
    "tqdm==4.64.0",
]

setup(
    name="npcgp",
    author="Magnus Ross & Tom McDonald",
    packages=["npcgp"],
    description="Implementation of NP-CGP model.",
    long_description=open("README.md").read(),
    install_requires=requirements,
    python_requires=">=3.9",
)
