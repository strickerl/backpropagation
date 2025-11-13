# -*- coding: utf-8 -*-
"""
Implement all the required arithmetic functions and non-linearities. 
Each class has a function "self.f()" implementing the actual function, and 
"self.df()" implementing the first partial derivative.

self.f()  = FLOAT
self.df() = LIST[FLOAT] or LIST[FLOAT,FLOAT] 

"""

#Import libraries.
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import math
import string




class Operator(ABC):

    @abstractmethod
    def f(self, a, b = None) -> float:
        raise NotImplementedError()
        return f_res

    @abstractmethod
    def df(self, a, b = None) -> list:
        raise NotImplementedError()
        return [df_res]

class Pow(Operator):

    def f(self, a, b):
        return a**b

    def df(self, a, b):
        if a <= 0: ## work-around: treat as unary operation if a negative
            return [b * (a ** (b - 1))]
        else:
            return [b * (a ** (b - 1)), (a ** b) * math.log(a)]

class Add(Operator):

    def f(self, a, b):
        return a+b
    
    def df(self, a, b):
        return [1.0, 1.0]

class Sub(Operator):

    def f(self, a, b):
        return a-b

    def df(self, a, b):
        return [1.0, -1.0]

class Mult(Operator):

    def f(self, a, b):
        return a*b

    def df(self, a, b):
        return [b, a]

class Div(Operator):

    def f(self, a, b):
        if b == 0.:
           raise ZeroDivisionError("Division by zero in f calculation in Div.f()")
        else:
           return a/b

    def df(self, a, b):
        if b == 0.:
           raise ZeroDivisionError("Division by zero in df calculation in Div.f()")
        else:
            return [1.0/b, -a/b**2]

class Exp(Operator):

    def f(self, a, b = None):
       return math.exp(a)

    def df(self, a, b = None):
       return [math.exp(a)]

class Log(Operator):
    ## natural logarithm

    def f(self, a, b = None):
        if a <= 0.:
           raise ValueError("Log is undefined for a <= 0 in Log.f()")
        else:
           return math.log(a)

    def df(self, a, b = None):
        if a <= 0.:
           raise ValueError("Log is undefined for a <= 0 in Log.f()")
        else:
           return [1.0/a]

class Sin(Operator):

    def f(self, a, b = None):
        return a.sin(a)

    def df(self, a, b = None):
        return [math.cos(a)]

class Cos(Operator):

    def f(self, a, b = None):
        return a.cos(a)

    def df(self, a, b = None):
        return [-math.sin(a)]


OPERATIONS = {"+": Add(), "-": Sub(), "exp": Exp(), "log": Log(), "^": Pow(), "sin": Sin(), "cos": Cos(), "*": Mult(), "/": Div()}