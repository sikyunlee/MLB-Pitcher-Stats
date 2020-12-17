#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="MLB-Pitcher-Stats-sikyunlee", # Replace with your own username
    version="0.0.1",
    author="Sikyun (George) Lee",
    author_email="sikyunlee91@gmail.com",
    description="Package that provides basic MLB pitcher statistics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sikyunlee/MLB-Pitcher-Stats",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

