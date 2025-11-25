"""
Setup script for MechanicsDSL v0.5.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
# This assumes setup.py is in the root, next to README.md
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else "A Domain-Specific Language for Classical Mechanics"

setup(
    name="mechanics-dsl",
    version="0.5.0",
    description="A Domain-Specific Language for Classical Mechanics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Noah Parsons",
    author_email="nomapa223@gmail.com",
    url="https://github.com/MechanicsDSL/mechanicsdsl",
    license="MIT",
    # find_packages() will now automatically detect the 'mechanics_dsl' folder
    # because it contains an __init__.py
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "sympy>=1.9.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "web": [
            "streamlit>=1.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mechanics-dsl=mechanics_dsl.core:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="physics mechanics lagrangian hamiltonian simulation dsl",
)
