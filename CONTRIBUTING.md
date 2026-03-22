# Contributing to MechanicsDSL

Thank you for your interest in contributing! MechanicsDSL is a pedagogical tool designed to make computational physics accessible. We welcome contributions from students, educators, and researchers.

## Getting Started

- Please review this document before contributing.
- MechanicsDSL fosters a supportive, inclusive community. All contributors must follow our Code of Conduct.

## Reporting Bugs

1. **Check the Issue Tracker:** Search for existing reports before opening a new issue.
2. **Open a New Issue:** Use the Bug Report template and provide:
   - The exact DSL code causing the error.
   - The full Python traceback.
   - Your operating system and Python version.

## Suggesting Features

Have an idea for a new physics system or compiler optimization? Open a Feature Request issue! We are especially interested in:
- New visualization types (e.g., phase portraits, Poincaré maps).
- Support for non-conservative forces.
- Educational examples and tutorials.

## Submitting Pull Requests

- **Fork** the repository and clone your fork locally.
- **Create a new branch** for your feature/bugfix:  
  `git checkout -b feature/my-new-feature`
- **Implement and test** your changes (run `pytest` for regression checks).
- **Commit** using clear, descriptive messages:  
  `git commit -m "Add feature: <description>"`
- **Push** to your fork and submit a Pull Request.

## Development Setup

Clone your fork  
git clone https://github.com/YOUR_USERNAME/mechanicsdsl.git  
cd mechanicsdsl  

Install in editable mode with dev dependencies  
pip install -e ".[dev]"  

Run tests  
pytest  

## Coding Standards

- **Python:** Follow PEP 8. Run `black .` before submitting.
- **Documentation:** All new functions require docstrings.
- **DSL Syntax:** Document new DSL commands in the README.

## Student Contributors

If you are a high school or undergraduate student, please mention this in your PR!  
MechanicsDSL is committed to mentoring the next generation of computational physicists.

---

Thank you for helping build a tool for accessible, modern physics!
