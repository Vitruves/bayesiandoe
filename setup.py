from setuptools import setup, find_packages

setup(
    name="bayesiandoe",
    version="1.0.5",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "optuna",
        "scipy",
        "PySide6",
        "rdkit",
    ],
    entry_points={
        "console_scripts": [
            "bayesiandoe=bayesiandoe.__main__:main",
        ],
    },
    author="Johan H.G. Natter",
    description="Bayesian Design of Experiments for Chemical Optimization",
)