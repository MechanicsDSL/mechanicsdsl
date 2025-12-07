#!/bin/bash
# Compile script for Fortran pendulum demo

echo "Compiling Fortran pendulum simulation..."

# Check for gfortran
if ! command -v gfortran &> /dev/null; then
    echo "Error: gfortran not found. Please install a Fortran compiler."
    exit 1
fi

gfortran -O3 -o pendulum pendulum.f90

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./pendulum"
else
    echo "Build failed."
    exit 1
fi
