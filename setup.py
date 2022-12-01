# -*- coding: utf-8 -*-
"""bheEASyMesh: a meshing tool for OpenGeoSys bhe-meshes"""

from setuptools import setup


README = open("README.md").read()


setup(name="bheEASyMesh",
      version='0.2.0',
      maintainer="Simon Richter",
      maintainer_email="simon.richter@htwk-leipzig.de",
      long_description=README,
      long_description_content_type="text/markdown",
      author="Simon Richter",
      author_email="simonr92@gmx.de",
      url="https://github.com/Eqrisi/BHE_EASyMesh.git",
      #   classifiers=["Intended Audience :: Science/Research",
      #       "Topic :: Scientific/Engineering :: Visualization",
      #       "Topic :: Scientific/Engineering :: Physics",
      #       "Topic :: Scientific/Engineering :: Mathematics",
      #       "License :: OSI Approved :: MIT License",
      #       "Programming Language :: Python :: 3",
      #       "Programming Language :: Python :: 3.8"],
      license="MIT -  see LICENSE.txt",
      platforms=["Windows"],#, "Linux", "Solaris", "Mac OS-X", "Unix"], not tested on other platforms
      include_package_data=True,
    #   python_requires='>=3.8',
      install_requires=[
        "numpy",
        "pandas",
        "gmsh",
        "vtk", 
        "scipy"],
      py_modules=["bheEASyMesh/bhemeshing"])

