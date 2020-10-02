import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="meshtal", 
    version="0.1.0-alpha",
    author="Miguel Magan",
    author_email="mmagan@essbilbao.org",
    description="Module read and post-process MCNP mesh tallies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MigMagan/meshtal",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: General Public License V3", 
        "Operating System :: Linux",
        "Development Status :: 3 - Alpha",
    ],
    python_requires='>=3.5',
)
