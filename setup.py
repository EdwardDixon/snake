import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-snake", # Replace with your own username
    version="0.0.1",
    author="Edward Dixon",
    author_email="dixon.edward@gmail.com",
    description="Sample PyTorch implementation of the snake activation function",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EdwardDixon/snake",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch',
      ],
    python_requires='>=3.6',
)