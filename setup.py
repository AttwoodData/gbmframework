from setuptools import setup, find_packages
import os
import re

# Read version without importing the package
with open('src/gbmframework/__init__.py', 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="gbmframework",
    version=version,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "shap"
    ],
    extras_require={
        'xgboost': ['xgboost'],
        'lightgbm': ['lightgbm'],
        'catboost': ['catboost'],
        'hyperopt': ['hyperopt'],
        'all': ['xgboost', 'lightgbm', 'catboost', 'hyperopt']
    },
    author="Mark Attwood + Claude 3.7",
    author_email="attwoodanalytics@gmail.com",
    description="A unified framework for Gradient Boosting Models with SHAP analysis",
    keywords="machine learning, gradient boosting, xgboost, lightgbm, catboost, shap",
    url="https://github.com/yourusername/gbmframework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)