from setuptools import setup, find_packages
import os
import re

def get_version():
    version_file = os.path.join('gbmframework', '_version.py')
    with open(version_file, 'r') as f:
        version_line = f.read()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    match = re.search(version_regex, version_line, re.M)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="gbmframework",
    version=get_version(),
    packages=find_packages(),
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
