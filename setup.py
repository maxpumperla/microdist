import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="microdist",
    version="0.1.0",
    author="Max Pumperla",
    author_email="max.pumperla@gmail.com",
    install_requires=["ray"],
    description="The tiniest distributed neural network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maxpumperla/microdist",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
