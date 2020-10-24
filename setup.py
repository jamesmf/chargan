from setuptools import setup, find_packages

setup(
    name="chargan",
    version="0.0.1",
    author="jamesmf",
    author_email="",
    description=("character-level generative networks"),
    license="BSD",
    keywords="embeddings composable character-level",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=[
        "transformers==3.4.0",
        "tokenizers==0.9.2",
        "datasets==1.1.2",
        "tensorflow-gpu==2.3.1",
        "numpy==1.18.5",
        "tqdm==4.49.0",
        "pytest==6.1.1",
    ],
)
