import meshio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the mesh file
mesh = meshio.read("model_13630.msh")

# Extract vertices and cells
vertices = mesh.points
cells = mesh.cells

# Plot the mesh
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# Plot cells (assuming only triangular elements for simplicity)
for cell in cells:
        for triangle in cell.data:
            triangle_vertices = vertices[triangle]
            ax.plot(triangle_vertices[:, 0], triangle_vertices[:, 1], triangle_vertices[:, 2], c='b')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()