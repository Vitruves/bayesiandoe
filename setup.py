from setuptools import setup, find_packages

setup(
    name="bayesiandoe",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "alembic",
        "colorlog",
        "contourpy",
        "cycler",
        "fonttools",
        "importlib-metadata",
        "importlib-resources",
        "joblib",
        "kiwisolver",
        "mako",
        "markupsafe",
        "matplotlib",
        "numpy",
        "optuna",
        "packaging",
        "pandas",
        "patsy",
        "pillow",
        "pyparsing",
        "pyside6",
        "python-dateutil",
        "pytz",
        "pyyaml",
        "rdkit",
        "scikit-learn",
        "scipy",
        "shiboken6",
        "six",
        "sqlalchemy",
        "statsmodels",
        "threadpoolctl",
        "tqdm",
        "typing-extensions",
        "tzdata",
        "botorch",
        "gpytorch",
        "torch",
        "pytest",
        "gpyopt",
        "zipp",
    ],
    entry_points={
        "console_scripts": [
            "bayesiandoe=bayesiandoe.__main__:main",
        ],
    },
    author="Johan H.G. Natter",
    description="Bayesian Design of Experiments for Chemical Optimization",
)