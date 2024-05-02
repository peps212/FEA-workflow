import jax.numpy as np
import os
import meshio
import jax
import numpy as onp
import matplotlib.pyplot as plt

# Import JAX-FEM specific modules.
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.utils import save_sol
from jax_fem.generate_mesh import box_mesh, get_meshio_cell_type, Mesh
from jax_fem import logger
import logging

logger.setLevel(logging.DEBUG)

# Define constitutive relationship.
class LinearElasticity(Problem):
    # The function 'get_tensor_map' overrides base class method. Generally, JAX-FEM
    # solves -div(f(u_grad)) = b. Here, we have f(u_grad) = sigma.
    def get_tensor_map(self):
        def stress(u_grad):
            E = 66.666e3
            nu = 0.3
            mu = E / (2. * (1. + nu))
            lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
            epsilon = 0.5 * (u_grad + u_grad.T)
            sigma = lmbda * np.trace(epsilon) * np.eye(self.dim) + 2 * mu * epsilon
            return sigma
        return stress

    def get_surface_maps(self):
        def surface_map(u, x):
            return np.array([0., 10., 0.])
        return [surface_map]

    def compute_l2_norm_error(self, sol, true_u_fn):
        cells_sol = sol[self.fes[0].cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.fes[0].shape_vals[None, :, :, None], axis=2)
        physical_quad_points = self.fes[0].get_physical_quad_points() # (num_cells, num_quads, dim)
        true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
        # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
        l2_error = np.sqrt(np.sum((u - true_u)**2 * self.fes[0].JxW[:, :, None]))
        return l2_error

    def compute_h1_norm_error(self, sol, true_u_fn):
        cells_sol = sol[self.fes[0].cells] # (num_cells, num_nodes, vec)
        # (num_cells, 1, num_nodes, vec) * (1, num_quads, num_nodes, 1) -> (num_cells, num_quads, vec)
        u = np.sum(cells_sol[:, None, :, :] * self.fes[0].shape_vals[None, :, :, None], axis=2)
        # (num_cells, 1, num_nodes, vec, 1) * (num_cells, num_quads, num_nodes, 1, dim) -> (num_cells, num_quads, num_nodes, vec, dim)
        u_grads = cells_sol[:, None, :, :, None] * self.fes[0].shape_grads[:, :, :, None, :]
        u_grads = np.sum(u_grads, axis=2) # (num_cells, num_quads, vec, dim)
        physical_quad_points = self.fes[0].get_physical_quad_points() # (num_cells, num_quads, dim)
        true_u = jax.vmap(jax.vmap(true_u_fn))(physical_quad_points) # (num_cells, num_quads, vec)
        true_u_grads = jax.vmap(jax.vmap(jax.jacrev(true_u_fn)))(physical_quad_points) # (num_cells, num_quads, vec, dim)
        # (num_cells, num_quads, vec) * (num_cells, num_quads, 1)
        val_l2_error = np.sqrt(np.sum((u - true_u)**2 * self.fes[0].JxW[:, :, None]))
        # (num_cells, num_quads, vec, dim) * (num_cells, num_quads, 1, 1)
        grad_l2_error = np.sqrt(np.sum((u_grads - true_u_grads)**2 * self.fes[0].JxW[:, :, None, None]))
        h1_error = val_l2_error + grad_l2_error
        return h1_error

def true_u_fn(point):
    """Some arbitrarily created analytical solution
    """
    x, y, z = point
    return np.array([1e2*((x - 0.5)**3 + 2.*(y - 0.5)**3 + 1e1*np.exp(-(z - 0.5)**2))])

def compute_stress_displacement(problem, sol):
    cells_sol = sol[problem.fes[0].cells]  # (num_cells, num_nodes, vec)
    physical_quad_points = problem.fes[0].get_physical_quad_points()  # (num_cells, num_quads, dim)
    
    # Compute displacement
    u = np.sum(cells_sol[:, None, :, :] * problem.fes[0].shape_vals[None, :, :, None], axis=2)
    
    # Compute stress
    #u_grad = np.sum(cells_sol[:, None, :, :, None] * problem.fes[0].shape_grads[:, :, :, None, :], axis=2)
    
    #stress_fn = problem.get_tensor_map()
    #stress = stress_fn(u_grad)
    
    return u

def problem(ele_type, mesh_file, data_dir):
    cell_type = get_meshio_cell_type(ele_type)
    mesh_read = meshio.read(mesh_file)
    Lz = np.max(mesh_read.points[:, 2])
    Lx = np.max(mesh_read.points[:, 0])
    print(Lx)
    cells = mesh_read.cells_dict["tetra"]
    mesh = Mesh(mesh_read.points, mesh_read.cells_dict["tetra"])

    # Define boundary locations.
    def left(point):
        return np.isclose(point[0], Lx, atol=1e-5)

    def right(point):
        return np.isclose(point[2], Lz, atol=1e-5)


    # Define Dirichlet boundary values.
    def zero_dirichlet_val(point):
        return 0.

    dirichlet_bc_info = [[left] * 3, [0, 1, 2], [zero_dirichlet_val] * 3]

    # Define Neumann boundary locations.
    location_fns = [right]

    # Create an instance of the problem.
    problem = LinearElasticity(mesh, vec=3, dim=3, ele_type=ele_type,
                               dirichlet_bc_info=dirichlet_bc_info,
                               location_fns=location_fns)

    # Solve the defined problem.
    sol_list = solver(problem, linear=True, use_petsc=True)
    u = compute_stress_displacement(problem, sol_list[0])

    # Store the solution to local file.
    vtk_path = os.path.join(data_dir, 'vtk/u.vtu')
    save_sol(problem.fes[0], sol_list[0], vtk_path)

    l2_error = problem.compute_l2_norm_error(sol_list[0], true_u_fn)
    h1_error = problem.compute_h1_norm_error(sol_list[0], true_u_fn)
    return l2_error, h1_error, u

def plot_stress_displacement(displacements, stresses, ele_types, mesh_files):
    num_meshes = len(mesh_files)
    x = range(1, num_meshes + 1)

    plt.figure(figsize=(12, 6))

    # Plot displacement
    plt.subplot(1, 2, 1)
    for i, ele_type in enumerate(ele_types):
        plt.plot(x, [onp.mean(disp) for disp in displacements[i]], marker='o', label=ele_type)
    plt.xlabel('Mesh Resolution')
    plt.ylabel('Displacement')
    plt.title('Displacement vs Mesh Resolution')
    plt.legend()
    plt.xticks(x, mesh_files, rotation=45)

    plt.tight_layout()
    plt.show()


def plot_errors(l2_errors_orders, h1_errors_orders, ele_types):
    plt.figure(figsize=(10, 5))

    # Plot L2 errors
    plt.subplot(1, 2, 1)
    for i, ele_type in enumerate(ele_types):
        plt.plot(l2_errors_orders[i], marker='o', label=ele_type)
    plt.xlabel('Mesh Resolution')
    plt.ylabel('L2 Error')
    plt.title('L2 Error Convergence')
    plt.legend()

    # Plot H1 errors
    plt.subplot(1, 2, 2)
    for i, ele_type in enumerate(ele_types):
        plt.plot(h1_errors_orders[i], marker='o', label=ele_type)
    plt.xlabel('Mesh Resolution')
    plt.ylabel('H1 Error')
    plt.title('H1 Error Convergence')
    plt.legend()

    plt.tight_layout()
    plt.show()

def convergence_test():
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, 'data')
    ele_types = ['TET4']
    mesh_files = ['model_14890.msh', 'model_15071.msh', 'model_15494.msh', 'model_15931.msh', 'model_17421.msh','model_18497.msh', 'model_18984.msh', 'model_19294.msh','model_24425.msh', 'model_36688.msh']  # Add mesh files for different resolutions
    l2_errors_orders = []
    h1_errors_orders = []
    displacements = []
    stresses = []
    for ele_type in ele_types:
        l2_errors = []
        h1_errors = []
        displacements_ele = []
        stresses_ele = []
        for mesh_file in mesh_files:
            l2_error, h1_error, u = problem(ele_type, mesh_file, data_dir)
            l2_errors.append(l2_error)
            h1_errors.append(h1_error)
            displacements_ele.append(u)
            #stresses_ele.append(stress)
        l2_errors_orders.append(l2_errors)
        h1_errors_orders.append(h1_errors)
        displacements.append(displacements_ele)
        #stresses.append(stresses_ele)

    l2_errors_orders = onp.array(l2_errors_orders)
    h1_errors_orders = onp.array(h1_errors_orders)

    print(f"l2_errors_orders = \n{l2_errors_orders}")
    print(f"h1_errors_orders = \n{h1_errors_orders}")
    # Print convergence rates
    # ...

    plot_stress_displacement(displacements, stresses, ele_types, mesh_files)
    plot_errors(l2_errors_orders, h1_errors_orders, ele_types)
    

if __name__ == "__main__":
    convergence_test()