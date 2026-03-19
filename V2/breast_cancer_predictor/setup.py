from setuptools import setup, find_packages

setup(
    name="breast_cancer_predictor",
    version="1.0.0",
    description="A machine learning tool for predicting key mutations in breast cancer",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "requests>=2.28.0",
        "biopython>=1.80",
        "joblib>=1.2.0",
    ],
    extras_require={
        "gpu": ["tensorflow-gpu>=2.10.0"],
    },
)
