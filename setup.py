# chardet's setup.py
from distutils.core import setup
import setuptools
setup(
    name = "inphosemantics",
    version = "0.1",
    description = "Indiana Philosophy Ontology Project data mining tools",
    author = "The Indiana Philosophy Ontology (InPhO) Project",
    author_email = "inpho@indiana.edu",
    url = "http://inpho.cogs.indiana.edu/",
    download_url = "http://www.github.com/inpho/inphosemantics",
    keywords = [],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Text Processing :: Linguistic",
        ],
    install_requires=[
        "numpy>=1.4.0,<=1.4.99",
        "nltk>=2.0.0"
    ],
    packages=['inphosemantics', 'inphosemantics.corpus',
              'inphosemantics.model', 'inphosemantics.viewer',
              'inphosemantics.inpho'],
    # data_files=[('inphosemantics/corpus',
    #              ['inphosemantics/corpus/beagle-stopwords-jones.txt'])]

)
