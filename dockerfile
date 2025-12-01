# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container if you have one
COPY requirements.txt .

# Install all dependencies (adjust this list as needed)
RUN pip install pytest pytest-cov matplotlib sympy scipy

# Copy the rest of your application code into the container
COPY . .

# Command to run the tests when the container starts (optional for CI)
# CMD ["pytest", "tests/"]
