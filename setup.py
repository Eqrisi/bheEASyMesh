# -*- coding: utf-8 -*-
"""bheEASyMesh: a meshing tool for OpenGeoSys bhe-meshes"""

from setuptools import setup


README = open("README.md").read()


setup(name="bheEASyMesh",
      version='v0.2.1',
      maintainer="Simon Richter",
      maintainer_email="simon.richter@htwk-leipzig.de",
      long_description=README,
      long_description_content_type="text/markdown",
      author="Simon Richter",
      author_email="simonr92@gmx.de",
      url="https://github.com/Eqrisi/bheEASyMesh.git",
      classifiers=[],
      download_url="https://github.com/Eqrisi/bheEASyMesh/archive/refs/tags/v0.2.1.tar.gz",
      license="MIT -  see LICENSE.txt",
      platforms=["Windows", "Linux", "Mac OS-X"],
      keywords = ['Meshing', 'BHE', 'GSHP', 'Finite-Element-Method', 'OpenGeoSys'],
      include_package_data=True,
      install_requires=[
        "numpy",
        "gmsh",
        "vtk", 
        "scipy"],
      py_modules=["bheEASyMesh/bhemeshing"])

