import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meshtal", 
    version="0.0.1",
    author="Miguel Magan",
    author_email="mmagan@essbilbao.org",
    description="Module to operate with MCNP mesh tallies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MigMagan/meshtal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: General Public License V3", 
        "Operating System :: Linux",
    ],
    python_requires='>=3.5',
)
