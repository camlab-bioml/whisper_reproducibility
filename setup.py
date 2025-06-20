from setuptools import setup, find_packages

setup(
    name="puppi_reproducibility",
    version="0.1",
    author="Vesal Kasmaeifar",
    author_email="vesal.kasmaeifar@mail.utoronto.com",
    description="PU Learning pipeline for proximity proteomics and AP-MS PPI scoring",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "goatools",
        "requests",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "puppi=script:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
