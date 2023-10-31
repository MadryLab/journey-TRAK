#!/usr/bin/env python
from setuptools import setup

setup(name="diffusion_trak",
      version="1.0.0",
      description="Diffusion TRAK: Data Attribution for Diffusion Models",
      long_description="",
      author="MadryLab",
      author_email='krisgrg@mit.edu',
      license_files=('LICENSE.txt', ),
      packages=['diffusion_trak'],
      install_requires=[
          "traker>=0.2.2",
          "diffusers==0.15.1",
          "transformers",
          "datasets"
      ],
      extras_require={
          'notebooks':
              ["scikit_learn",
               "torchvision",
               "seaborn",
               ], },
      include_package_data=True,
      )
