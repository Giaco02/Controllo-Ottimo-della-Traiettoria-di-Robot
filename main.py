import numpy as np
# Importa la classe dal file optimal_control_solver.py
from optimal_control_solver import OptimalControlSolver

# Usa la classe
solver = OptimalControlSolver(np.array([6.2, 6.2, 3.1, 6.2, 6.2, 6.2]),'crba',0.02,40) #([6.2, 6.2, 3.1, 6.2, 6.2, 6.2]) [1, 1, 1, 1, 1, 1])
solver.run()
