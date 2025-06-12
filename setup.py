from setuptools import setup, find_packages

setup(
    name="games_analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "pymongo",
        "requests",
        "beautifulsoup4",
        "joblib"
    ],
) 