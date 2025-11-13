# -*- coding: utf-8 -*-
"""
Implementation of forard and backward propagation through the nodes
The ``forward`` method populates the output of operation nodes.
The ``backward`` method propagates the gradient back to the parent nodes (if existent).
"""

#Import libraries.
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import math
import string
from main import NODE_NAMES
from Operator import OPERATIONS

class Node:
    """
    A class representing a node in a computation graph,
    Each node can represent a constant, a variable, or an operation applied to its parents
    """

    def __init__(self, output=None, parent_left=None, parent_right=None, operation=None, depth=0):
        """
        Initializes a node with the given parameters.

        Args:
            output: The value of the node: a float representing a constant or a char representing a variable.
                    If "None" it has to be populated later by direct access (for variables) or using forward call for operations.
            parent_left: The left parent node (used for binary operations).
            parent_right: The right parent node (used for binary and UNARY operations).
            operation: The operation name (a key from OPERATIONS dictionary) if the node is an operator.
        """

        self.operation_name = operation # Used for plotting
        self.depth = depth              # Used by the executor to infer when to execute the node during forward and backward calls
        self.is_constant = False
        self.is_variable = False
        self.parent_left  = parent_left
        self.parent_right = parent_right
        self.output   = output
        self.gradient = 0


        # Validate initialization
        if (output and operation) or not (output or operation): 
            raise Exception("A node definition has to contain only one of: [operation, output]")

        # Initialize depending on the type of the node
        if operation:
            self.name      = NODE_NAMES.pop() # Used for plotting: gives a name to the node
            self.operation = OPERATIONS[operation]
        elif isinstance(output, str):
            self.name        = output # A variable node already has a name
            self.is_variable = True
        else:
            self.name        = str(output)
            self.is_constant = True

    def forward(self):
        """
        If the node is a constant or variable node, just return.
        If the node is an operator node, compute and set the output value of the node.
        Parent nodes are expected to be initialized and contain a valid value in the output attribute.
        The value for unary operations is contained in self.parent_right.
        """
        if (self.is_constant == True or self.is_variable == True):
            return
        
        else:
            
            if self.parent_left is None or self.parent_right is None:  #Unary operator
                  a           = self.parent_right.output
                  self.output = self.operation.f(a)
                  
            elif self.parent_left is not None and self.parent_right is not None: #Binary operator
                  a           = self.parent_left.output
                  b           = self.parent_right.output
                  self.output = self.operation.f(a,b)
                  
            else:             
                raise ValueError(
                f"Error in operator node '{self.operation}': "
                "binary op missing right parent or unary op missing right input.")



    def backward(self):
        """
        If the node is a constant or variable node, just return.
        If the node is an operator node,
        - compute the local gradient,
        - combine it with the upstream gradient (chain rule)
        - and accumulate the result into the gradient of the parents nodes.
        The gradient of the output node in the graph is expected to be 1
        """
        if (self.is_constant == True or self.is_variable == True):
            return
        
        else:
            if self.parent_left is None and self.parent_right is not None:
                a  = self.parent_right.output
                da = self.operation.df(a)[0]
                self.parent_right.gradient += self.gradient * da
            else:
                a = self.parent_left.output
                b = self.parent_right.output
                da, db = self.operation.df(a, b)
                self.parent_left.gradient  += self.gradient * da
                self.parent_right.gradient += self.gradient * db

            

    def __repr__(self):
        """
        Detailed string representation for debugging.
        """
        return str(self.output)


