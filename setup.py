import setuptools

setuptools.setup(
    name="gan_collection",
    version="0.0.1",
    author="Luca Simonetto",
    author_email="luca.simonetto.94@gmail.com",
    description="collection of various gans",
    long_description_content_type="text/markdown",
    url="...", #TODO: add
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "keras",
        "tensorflow",
        "pillow",
        "scikit-learn",
        "scikit-image",
        "h5py",
        "imageio",
        "pyyaml",
        "graphviz",
        "scipy==1.2.0",
        "pydot",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

)
