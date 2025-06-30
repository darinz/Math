# Comprehensive Calculus Guide for AI/ML and Data Science

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-orange.svg)](https://matplotlib.org/)
[![SymPy](https://img.shields.io/badge/SymPy-1.10+-purple.svg)](https://www.sympy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-red.svg)](https://scipy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This guide provides a comprehensive introduction to calculus concepts essential for artificial intelligence, machine learning, and data science applications. Each section includes theoretical explanations, practical Python code examples, and real-world applications.

## Table of Contents

### Interactive Jupyter Notebooks
1. [Limits and Continuity](01-limits-continuity.ipynb) - [📖 Markdown](01-limits-continuity.md)
2. [Derivatives and Differentiation](02-derivatives.ipynb) - [📖 Markdown](02-derivatives.md)
3. [Applications of Derivatives](03-derivative-applications.ipynb) - [📖 Markdown](03-derivative-applications.md)
4. [Integration](04-integration.ipynb) - [📖 Markdown](04-integration.md)
5. [Applications of Integration](05-integration-applications.ipynb) - [📖 Markdown](05-integration-applications.md)
6. [Multivariable Calculus](06-multivariable-calculus.ipynb) - [📖 Markdown](06-multivariable-calculus.md)
7. [Vector Calculus](07-vector-calculus.ipynb) - [📖 Markdown](07-vector-calculus.md)
8. [Optimization Techniques](08-optimization.ipynb) - [📖 Markdown](08-optimization.md)
9. [Calculus in Machine Learning](09-ml-applications.ipynb) - [📖 Markdown](09-ml-applications.md)
10. [Numerical Methods](10-numerical-methods.ipynb) - [📖 Markdown](10-numerical-methods.md)

### Additional Resources
- [📓 Comprehensive Examples Notebook](calculus_examples.ipynb) - Interactive examples covering all topics
- [📋 Summary](SUMMARY.md) - Quick reference guide
- [📦 Requirements](requirements.txt) - Python dependencies

## Prerequisites

- Basic Python programming knowledge
- Understanding of algebra and trigonometry
- Familiarity with mathematical notation

## Required Python Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import scipy.optimize as opt
import scipy.integrate as integrate
from scipy import linalg
import pandas as pd
import seaborn as sns
```

## Quick Start

Each section can be read independently, but we recommend following the order for a complete understanding. Code examples are designed to be run in Jupyter notebooks or Python scripts.

### Setup and Installation

1. **Install dependencies** using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```

2. **Interactive learning**: Use the individual chapter notebooks (`.ipynb` files) for hands-on practice with specific topics:
   - Each notebook contains executable code examples
   - Interactive visualizations and demonstrations
   - Step-by-step explanations with working code

3. **Comprehensive examples**: Use the `calculus_examples.ipynb` Jupyter notebook for hands-on practice with all concepts covered in this guide.

4. **Progressive learning**: Follow the chapter order for a complete understanding of calculus fundamentals and applications.

## Learning Formats

### 📓 Jupyter Notebooks (`.ipynb`)
- **Interactive code execution** - Run examples directly in the notebook
- **Rich visualizations** - Dynamic plots and graphs
- **Step-by-step demonstrations** - See calculus concepts in action
- **Immediate feedback** - Experiment with parameters and see results

### 📖 Markdown Files (`.md`)
- **Detailed theoretical explanations** - Comprehensive mathematical foundations
- **Reference material** - Quick lookup for formulas and concepts
- **GitHub-friendly** - Easy to read in any markdown viewer
- **Print-friendly** - Clean formatting for documentation

## Applications in AI/ML

- **Gradient Descent**: Understanding derivatives for optimization
- **Backpropagation**: Chain rule applications in neural networks
- **Loss Functions**: Derivatives for model training
- **Probability Distributions**: Integration for probability calculations
- **Feature Engineering**: Calculus for data transformations

## Contributing

Feel free to contribute improvements, additional examples, or corrections to this guide. Both markdown and notebook formats are welcome! 