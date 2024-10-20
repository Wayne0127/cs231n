# CS231n Assignment 1

This repository contains my solutions for **CS231n: Convolutional Neural Networks for Visual Recognition** Assignment 1. The assignment focuses on image classification using techniques such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Softmax, and Neural Networks. We also explore using handcrafted features to improve classification performance.

## File Structure
```
cs231n-assignment1/
└── cs231n/
    ├── data_utils.py            # Utility functions for handling datasets
    ├── features.py              # Functions for extracting handcrafted features
    ├── gradient_check.py        # Gradient checking for backpropagation
    ├── layers.py                # Implementation of neural network layers
    ├── layer_utils.py           # Utility functions for layers
    ├── optim.py                 # Optimization algorithms
    ├── solver.py                # Solver for training models
    ├── vis_utils.py             # Visualization utilities
    ├── __init__.py              # Package initialization file
    └── classifiers/
        ├── fc_net.py            # Two-layer fully connected neural network implementation
        ├── k_nearest_neighbor.py# K-Nearest Neighbors classifier implementation
        ├── linear_classifier.py # Base class for linear classifiers
        ├── linear_svm.py        # SVM classifier implementation using a linear model
        ├── softmax.py           # Softmax classifier implementation
        └── __init__.py          # Package initialization file for classifiers
├── features.ipynb               # Handcrafted image features for classification
├── file_tree.txt                # File structure of the project
├── knn.ipynb                    # K-Nearest Neighbors implementation
├── softmax.ipynb                # Softmax classifier implementation
├── svm.ipynb                    # Support Vector Machine classifier implementation
├── two_layer_net.ipynb          # Two-layer fully connected neural network
└── README.md                    # This README file
```
## Instructions

1. **Run the Notebooks**: Each Jupyter notebook corresponds to a specific part of the assignment. Follow the order and execute all cells to get the results.
2. **Implementations**: The main implementation is inside the Python files in the `cs231n/` directory. Notebooks call the functions implemented in these files.
3. **Evaluate the Results**: After implementing the functions, evaluate the models using the provided datasets in the notebooks.

## Requirements

- **Python 3.x**
- **Jupyter Notebook**
- **NumPy**
- **Matplotlib**

Install the required packages using:

```bash
pip install -r requirements.txt
```
