
'''

This file contains high-level interfaces for conventient layer generation.

Sections:
* Imports
* Layer Classes

'''

###########
# Imports #
###########

import numpy as np
from abc import ABC, abstractmethod
from typing_extensions import Literal

from .variable import Variable, VariableOperand
from .misc_utilities import *

#################
# Layer Classes #
#################

class Layer(ABC):
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Variable:
        raise NotImplementedError

class LinearLayer(Layer):
    
    def __init__(self, number_of_inputs: int, number_of_outputs: int, init_method: Literal['random', 'zero'] = 'random'):
        if init_method == 'random':
            initializer = np.random.rand
        elif init_method == 'zero':
            def initializer(*args, **kwargs) -> np.ndarray:
                return np.zeros(*args, dtype=float, **kwargs)
        else:
            raise TypeError('{repr(init_method)} is not a valid initialization method.')
        self.matrix = Variable(initializer(number_of_inputs*number_of_outputs).reshape([number_of_inputs, number_of_outputs]))
        self.biases = Variable(np.expand_dims(initializer(number_of_outputs), 0))
        return

    def __call__(self, operand: VariableOperand) -> Variable:
        if isinstance(operand, (int, float, bool, np.number)):
            operand = Variable(np.array(operand))
        if len(operand.shape) == 0:
            operand = np.expand_dims(operand, axis=0)
        if len(operand.shape) == 1:
            operand = np.expand_dims(operand, axis=0)
        matrix_product = np.matmul(operand, self.matrix)
        linear_layer_result = matrix_product + self.biases
        return linear_layer_result

    def l1_norm(self) -> Variable:
        norm = self.matrix.sum()
        return norm

    def l2_norm(self, include_squareroot: bool = False) -> Variable:
        norm = self.matrix.pow(2.0).sum()
        if include_squareroot:
            norm = norm.pow(0.5)
        return norm

class LogisticRegressionLayer(LinearLayer):
    
    def __init__(self, number_of_inputs: int, number_of_outputs: int):
        super().__init__(number_of_inputs, number_of_outputs, init_method='zero')
        return

    def __call__(self, operand: VariableOperand) -> Variable:
        logit = super().__call__(operand)
        result = logit.sigmoid()
        return result
