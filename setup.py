from setuptools import setup

setup(
    name="appleamx",
    version="0.1.0",
    description="A short description",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="James Taylor",
    python_requires=">=3.9",
    py_modules=["appleamx", "appleamx_ops", "appleamx_pools"],
    install_requires=[
        "exo-lang>=1.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)