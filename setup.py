from setuptools import setup

setup(
    name="whisper_reproducibility",
    version="0.2",
    author="Vesal Kasmaeifar",
    description="Workflow reproducing the figures and tables of the WHISPER manuscript",
    packages=["src"],
    py_modules=["script"],
    package_data={"src": ["go-basic.obo", "goa_human.gaf"]},
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "openpyxl",
        "upsetplot",
        "gprofiler-official",
        "goatools",
        "requests",
        "snakemake",
    ],
    entry_points={"console_scripts": ["whisper-repro=script:main"]},
    python_requires=">=3.9",
)
