import matplotlib.pyplot as plt
import sympy as sp

# Set up the symbols
x, a = sp.symbols('x a')

# Define the expression
expr = x + (1/a) * sp.sin(a*x)**2

# Set up the plot
fig, ax = plt.subplots(figsize=(1.5, 1)) 
ax.axis('off')  # Turn off the axis


# Render the expression as text using Matplotlib's built-in MathText
txt = f'${sp.latex(expr)}$'
ax.text(0.5, 0.5, txt, fontsize=10, ha='center', va='center', transform=ax.transAxes)

# Tight layout and save the figure as SVG
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust the subplot edges

# Minimize the whitespace around the plot
fig.tight_layout(pad=0)
fig.savefig('expression.svg', format='svg', bbox_inches='tight', pad_inches=0.1)  # Tight bounding box with minimal padding


# Optionally display the plot
plt.show()
