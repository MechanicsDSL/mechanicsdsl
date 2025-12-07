#!/bin/bash
# Compile script for C++ pendulum demo

echo "Compiling C++ pendulum simulation..."

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ not found. Please install a C++ compiler."
    exit 1
fi

g++ -O3 -std=c++17 -o pendulum pendulum.cpp

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./pendulum"
else
    echo "Build failed."
    exit 1
fi
