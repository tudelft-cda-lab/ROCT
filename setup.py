import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="roct",
    version="0.0.3",
    author="Daniel Vos",
    author_email="D.A.Vos@tudelft.nl",
    url="https://github.com/tudelft-cda-lab/ROCT",
    description="Robust Optimal Classification Trees",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    py_modules=["roct"],
    install_requires=[
        "numpy<=1.21",
        "groot-trees",
        "gurobipy",
        "python-sat",
        "tqdm",
        "seaborn",
    ]
)