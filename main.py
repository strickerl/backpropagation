# -*- coding: utf-8 -*-
"""Stricker: Automatic differentiation: forward an backwards propagation algorithm 
    
"""

#Import libraries.
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import math
import string
import Node
import Executor
from Operator import Exp,Log,Add,Pow,Sub,Div,Cos,Sin

# Generate node names: ['A',...'Z','AA',...,'AZ',...,'ZZ']
NODE_NAMES = [c for c in string.ascii_uppercase]
NODE_NAMES.extend([f"{a}{b}" for a in string.ascii_uppercase for b in string.ascii_uppercase])
NODE_NAMES.reverse()




"""
Visually test how computation graph is built.


"""

# Plot and check values.
operations_example_0 = [["x","^",2],"+",["x","^",2]]
variables_values_example_0 = {"x":2}
executor = Executor(operations_example_0)
executor.initialize_variables(variables_values_example_0)
executor.plot_graph()

# Notice in this example how x² are merged into a single node.
len(executor.node_cache) # Should be 4

"""# <font color=purple><strong>Testing and Grading</strong></font>


This cell contains public test cases. **Make sure your whole notebook runs without throwing errors**. Your final grade will be determined by these test cases and **additional private ones**.
"""

###--------- OPERATIONS ---------####

tests = [
    lambda: Exp().f(3)     == math.exp(3),
    lambda: Exp().df(3)[0] == math.exp(3),
    lambda: Log().f(math.exp(2)) == 2,
    lambda: Log().df(5)[0]       == 0.2,
    lambda: Add().f(2, 3)        == 5,
    lambda: Add().df(2, 3)       == [1, 1]
]

passed = 0
failed = 0

for test in tests:
    try:
        assert test()
        passed += 1
    except AssertionError:
        failed += 1

score1 = passed/(passed+failed)*10
print(f"Operations Score: {score1} points")



###--------- NODE ---------####

n1 = Node(2)
n2 = Node(3)
n3 = Node(operation="*", parent_right=n1, parent_left=n2)
n3.forward()
n3.gradient = 1
n3.backward()

tests = [
    lambda: n1.output == 2,
    lambda: n2.output == 3,
    lambda: n3.output == 6,
    lambda: n2.gradient == 2,
]

passed = 0
failed = 0

for test in tests:
    try:
        assert test()
        passed += 1
    except AssertionError:
        failed += 1

score2 = passed/(passed+failed)*10
print(f"Node Score: {score2} points")



###--------- EXECUTOR: FORWARD & BACKWARD ---------####

operations1 = [[2,"+","x"],"+",[2,"+","x"]]
values1 = {"x":3}
output1 = 10
gradient1 = {"x":2}
example_1 = {"operations": operations1, "values": values1, "output": output1, "gradients": gradient1}

operations2 = [[[[2,"+","x"],"+",[2,"+","y"]],"*",3],"^","y"]
values2 = {"x":3, "y":2}
output2 = 729
gradient2 = {"x":162,"y":2564.665}
example_2 = {"operations": operations2, "values": values2, "output": output2, "gradients": gradient2}

operations3 = ["exp", [2,"*",[[["sin", "x"], "+", ["cos", "x"]],"*",["y","*",[[["sin", "x"], "+", ["cos", "x"]],"*","y"]]]]]
values3 = {"x":5, "y":3}
output3 = 3668.80
gradient3 = {"x":-110821.892,"y":20074.746}
example_3 = {"operations": operations3, "values": values3, "output": output3, "gradients": gradient3}

operations4 = ["exp",[["sin", [["log",["x","+","y"]],"^",2]],"+",["cos",[["log",["x","+","y"]],"^",2]]]]
values4 = {"x":1, "y":2}
output4 = 3.63
gradient4 = {"x":-1.54,"y":-1.54}
example_4 = {"operations": operations4, "values": values4, "output": output4, "gradients": gradient4}

examples = [example_1, example_2, example_3, example_4]

forward_points = 10
backward_points = 15
forward_correct = 0
backward_correct = 0

for example in examples:
    executor = Executor(example["operations"])
    executor.initialize_variables(example["values"])

    # Test forward pass
    try:
        assert abs(executor.forward() - example["output"]) < 0.01
        forward_correct += 1
    except AssertionError:
        pass  # Forward failed

    # Test backward pass
    try:
        derivatives = executor.backward()
        for derivative in derivatives:
            assert abs(derivatives[derivative] - example["gradients"][derivative]) < 0.01
        backward_correct += 1
    except AssertionError:
        pass  # Backward failed

# Calculate scores
factor = 1 if len(executor.node_cache) == 10 else 0.5 # Check whether nodes are properly merged


score_forward = (forward_correct / len(examples)) * forward_points*factor
score_backward = (backward_correct / len(examples)) * backward_points*factor

print(f"Forward Score: {score_forward:.2f} points")
print(f"Backward Score: {score_backward:.2f} points")



###--------- GRADIENT DESCENT ---------####

try:
    operations_example = [[["x", "+", 2], "^", 2], "+", [["y", "+", 10], "^", 2]]
    variables_values_example = {"x":3, "y":2}
    executor = Executor(operations_example)
    executor.initialize_variables(variables_values_example)
    root = executor.gradient_descent(1000,0.01)
    assert abs(root["x"] + 2) < 0.01
    assert abs(root["y"] + 10) < 0.01

    operations_example = [["cos","x"],"+",["sin","y"]]
    variables_values_example = {"x":3, "y":2}
    executor = Executor(operations_example)
    executor.initialize_variables(variables_values_example)
    executor.gradient_descent(100000,0.01)
    assert int(executor.forward()) == -2

    score3 = 5
except AssertionError:
    score3 = 0
print(f"Gradient Score: {score3:.2f} points")



print(f"\n---- Total Score: {score1 + score2 + score3 + score_forward + score_backward:.2f} points ----")