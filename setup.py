"""
Setup script for 3D Tongue Tip Tracking
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='tongue-tracking-3d',
    version='1.0.0',
    author='Saumya Saksena',
    author_email='kumaar324@gmail.com',
    description='Optical Flow based Tongue Tip Tracking in 3D for medical applications',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/dronefreak/3D-tongue-tip-tracking',
    project_urls={
        'Bug Reports': 'https://github.com/dronefreak/3D-tongue-tip-tracking/issues',
        'Source': 'https://github.com/dronefreak/3D-tongue-tip-tracking',
    },
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    keywords='tongue tracking, optical flow, 3D reconstruction, medical imaging, computer vision',
    python_requires='>=3.6',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'black>=20.0',
            'flake8>=3.8',
        ],
    },
    py_modules=['facial_landmarks_video', 'calib-camera'],
    scripts=[
        'facial_landmarks_video.py',
        'calib-camera.py',
    ],
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', 'LICENSE'],
    },
)
