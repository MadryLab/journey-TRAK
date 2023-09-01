#!/usr/bin/env python
from setuptools import setup

setup(name="diffusion-trak",
      version="1.0.0",
      description="Diffusion TRAK: Data Attribution for Diffusion Models",
      long_description="",
      author="MadryLab",
      author_email='krisgrg@mit.edu',
      license_files=('LICENSE.txt', ),
      packages=['diffusion-trak'],
      install_requires=[
          "traker",
          "diffusers",
      ],
      extras_require={
          'notebooks':
              ["scikit_learn",
               "torchvision",
               "seaborn",
               ], },
      include_package_data=True,
      )
