This is my Bachelor's thesis, focused on algorithms for robot dynamics.

This Bachelor's thesis investigates and compares two different approaches for solving the trajectory optimization problem of a robotic manipulator, focusing especially on computational efficiency. The main goal is to examine whether using exclusively the Recursive Newton-Euler Algorithm (RNEA) offers a computational time advantage over a combined approach that also includes the Composite Rigid Body Algorithm (CRBA).

The first part of the thesis covers the theoretical foundations, including spatial algebra, robot modeling, and inverse dynamics algorithms (RNEA and CRBA). Then, these concepts are applied to implement an optimal control solver in Python using the CasADi library, which formulates the problem considering system dynamics and constraints. The ADAM library was used and modified to fully implement the RNEA algorithm.

The file main.py contains an example of solving an optimal control problem, while rbd_algorithms.py is a modified version of the ADAM library file with a working RNEA implementation.

The thesis document is available in Italian.
