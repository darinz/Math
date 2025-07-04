# Mathematics for AI/ML and Data Science

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-green.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-orange.svg)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive collection of mathematical concepts, methods, and Python implementations essential for artificial intelligence, machine learning, and data science applications. This repository provides structured learning materials with theoretical foundations, practical code examples, and real-world applications.

## Overview

This repository contains three major mathematical disciplines, each organized into progressive learning modules, plus a comprehensive resource collection:

### [Calculus](calculus/)
Comprehensive coverage from fundamental concepts to advanced applications in machine learning:
- Limits, continuity, and derivatives
- Integration and its applications
- Multivariable and vector calculus
- Optimization techniques
- Numerical methods
- Machine learning applications (gradient descent, backpropagation)

### [Linear Algebra](linear-algebra/)
Essential linear algebra concepts with practical implementations:
- Vector and matrix operations
- Linear transformations and eigenvalues
- Vector spaces and matrix decompositions
- Applications in dimensionality reduction and neural networks
- Numerical linear algebra methods

### [Statistics](statistics/)
Statistical foundations and methods for data science:
- Descriptive statistics and probability theory
- Statistical inference and hypothesis testing
- Regression analysis and time series
- Multivariate statistics and Bayesian methods
- Experimental design and statistical learning

### [Reference](reference/)
Additional reference materials and supplementary content:
- Mathematical reference guides and textbooks
- Probability and statistics resources
- Linear algebra materials for deep learning
- Numerical methods documentation
- MIT course materials and additional learning resources
- Chain rule and derivative references
- Matrix cookbook and calculation references
- Gradient review materials
- Stanford Math 51 textbook
- Probability for Computer Science materials

## Repository Structure

```
Math/
├── calculus/           # 10 comprehensive chapters
│   ├── 01-limits-continuity.md
│   ├── 02-derivatives.md
│   ├── 03-derivative-applications.md
│   ├── 04-integration.md
│   ├── 05-integration-applications.md
│   ├── 06-multivariable-calculus.md
│   ├── 07-vector-calculus.md
│   ├── 08-optimization.md
│   ├── 09-ml-applications.md
│   ├── 10-numerical-methods.md
│   ├── calculus_examples.ipynb    # Interactive Jupyter notebook
│   ├── requirements.txt           # Python dependencies
│   ├── README.md                  # Learning guide
│   └── SUMMARY.md                 # Complete summary
├── linear-algebra/     # 9 structured modules
│   ├── 01-vectors.md
│   ├── 02-matrices.md
│   ├── 03-linear-transformations.md
│   ├── 04-eigenvalues-eigenvectors.md
│   ├── 05-vector-spaces.md
│   ├── 06-linear-independence.md
│   ├── 07-matrix-decompositions.md
│   ├── 08-ml-applications.md
│   ├── 09-numerical-linear-algebra.md
│   ├── README.md                  # Learning guide
│   └── SUMMARY.md                 # Complete summary
├── statistics/         # 10 statistical topics
│   ├── 01-descriptive-statistics.md
│   ├── 02-probability-fundamentals.md
│   ├── 03-statistical-inference.md
│   ├── 04-regression-analysis.md
│   ├── 05-time-series-analysis.md
│   ├── 06-multivariate-statistics.md
│   ├── 07-bayesian-statistics.md
│   ├── 08-experimental-design.md
│   ├── 09-statistical-learning.md
│   ├── 10-advanced-topics.md
│   ├── README.md                  # Learning guide
│   └── SUMMARY.md                 # Complete summary
├── reference/                           # Reference materials and supplementary content
│   ├── calculation_ref.pdf              # Mathematical calculation reference
│   ├── derivatives_Section14_5.pdf      # Derivatives reference (Section 14.5)
│   ├── gradient_review.pdf              # Gradient review materials
│   ├── Greek_Alphabet.pdf               # Greek alphabet reference
│   ├── linear_algebra_for_dl.pdf        # Linear Algebra for Deep Learning
│   ├── math-of-ml-book.pdf              # Mathematics of Machine Learning
│   ├── math51book.pdf                   # Stanford Math 51 textbook
│   ├── matrixcookbook.pdf               # Matrix cookbook reference
│   ├── MIT_ProbStats.zip                # MIT Probability & Statistics materials
│   ├── ML-Math_18-657-fall-2015.zip     # MIT Math for ML course (Fall 2015)
│   ├── ML-Matrix_18-065-spring-2018.zip # MIT Matrix Methods course (Spring 2018)
│   ├── numerical_for_dl.pdf             # Numerical for Deep Learning
│   ├── prob_for_dl.pdf                  # Probability for Deep Learning
│   ├── prob_reference.pdf               # Probability reference guide
│   ├── ProbabilityForCS.pdf             # Probability for Computer Science
│   ├── the_chain_rule.pdf               # Chain rule reference
│   └── the-chain-rule_lecture9.pdf      # Chain rule lecture materials
└── README.md                            # This file
```

Each discipline folder contains:
- Detailed markdown chapters with theoretical explanations
- Python code examples and implementations
- Practical exercises and applications
- Comprehensive README with learning paths
- Summary files for navigation
- Additional resources (Jupyter notebooks, requirements files)

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Basic programming knowledge
- Familiarity with mathematical notation
- Understanding of algebra and trigonometry

### Installation
```bash
# Clone the repository
git clone https://github.com/darinz/Math
cd Math

# Install required dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn sympy statsmodels
```

### Learning Path
1. **Choose your discipline**: Start with calculus, linear algebra, or statistics based on your background
2. **Follow the structure**: Each discipline has a README.md with learning objectives and prerequisites
3. **Progressive learning**: Chapters are designed to build upon previous concepts
4. **Practice**: Run code examples and complete exercises in each chapter
5. **Apply**: Use concepts in your own AI/ML projects

## Key Features

- **Comprehensive Coverage**: From fundamentals to advanced applications
- **Practical Implementation**: Python code examples for all concepts
- **Real-world Applications**: Direct connections to AI/ML use cases
- **Progressive Learning**: Structured chapters that build upon each other
- **Interactive Examples**: Jupyter notebook compatibility
- **Professional Quality**: Rigorous mathematical foundations with clear explanations

## Applications in AI/ML

### Calculus Applications
- Gradient descent optimization algorithms
- Neural network backpropagation
- Loss function derivatives
- Probability distribution calculations
- Feature engineering transformations

### Linear Algebra Applications
- Principal Component Analysis (PCA)
- Neural network weight matrices
- Singular Value Decomposition (SVD)
- Dimensionality reduction techniques
- Optimization algorithms

### Statistics Applications
- Model evaluation and validation
- Hypothesis testing for A/B testing
- Regression analysis for predictive modeling
- Bayesian inference for uncertainty quantification
- Experimental design for ML experiments

## Contributing

Contributions are welcome to improve content, add examples, or correct errors. Please ensure all mathematical content is accurate and code examples are well-documented.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This repository serves as a comprehensive mathematical foundation for AI/ML practitioners, combining theoretical rigor with practical implementation.
