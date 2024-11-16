import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def plot_constraints(A, b):
    plt.figure(figsize=(8, 8))

    x = np.linspace(0, 10, 400)

    # Plot constraints
    plt.plot(x, (28 - 4*x) / -7, label=r'$4x_1 - 7x_2 \geq 28$')
    plt.plot(x, (10 + 5*x) / 2, label=r'$-5x_1 + 2x_2 \leq 10$')
    
    # Non-negativity constraints
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)
    plt.show()

def solve_linear_programming(c, A, b):
    # Solve the linear programming problem
    res = linprog(c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method='highs')

    # Print the results
    print("Optimal value:", res.fun)
    print("Optimal point:", res.x)

    # Plot the feasible region
    plot_constraints(A, b)

# New linear programming problem
c_new = np.array([3, 4])  # Coefficients of the objective function to be minimized
A_new = np.array([[-4, 7], [5, 2]])  # Coefficients of the inequality constraints
b_new = np.array([-28, 10])  # Right-hand side of the inequality constraints

# Solve the new linear programming problem
solve_linear_programming(c_new, A_new, b_new)
# New linear programming problem
c_new = np.array([3, 4])  # Coefficients of the objective function to be minimized
A_new = np.array([[-4, 7], [5, 2]])  # Coefficients of the inequality constraints
b_new = np.array([-28, 10])  # Right-hand side of the inequality constraints

# Solve the new linear programming problem
res = linprog(c_new, A_ub=A_new, b_ub=b_new, bounds=[(0, None), (0, None)], method='highs')

# Print the results
print("Optimal value:", res.fun)
