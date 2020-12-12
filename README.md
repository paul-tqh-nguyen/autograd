# Automatic Differentiation Engine

This is a toy automatic differentiation engine using dynamic computation graphs (similar to PyTorch).

## Usage Tutorials

For a tutorial on how to use the higher-level interfaces (e.g. linear layers with gradients) offered by our automatic differentiation engine, see our [logistic regression example](https://github.com/paul-tqh-nguyen/autograd/blob/main/examples/logistic_regression.ipynb).

For a tutorial on how to use the lower-level interfaces (e.g. matrices or vectors with gradients or atomic numbers with gradients), see our [linear regression example](https://github.com/paul-tqh-nguyen/autograd/blob/main/examples/linear_regression.ipynb).

## Install Instructions

All the necessary dependencies can be installed via our [`environment.yml`](https://github.com/paul-tqh-nguyen/autograd/blob/main/environment.yml). See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for instructions on how to create an environment with the needed dependencies.

To install our automatic differentiation engine, simply run `python3 setup.py` from the root directory of the checkout. 