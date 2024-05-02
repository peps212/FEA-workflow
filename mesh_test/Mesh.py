import cadquery as cq
import gmsh

step_file = "tavola3.step"

def create_mesh(step_file, mesh_file='model.msh'):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

    # Load STEP file
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    # Fine-tuning mesh parameters
    #gmsh.model.mesh.setOrder(1)
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Delaunay meshing
    #gmsh.option.setNumber("Mesh.DrawSkinOnly", 1)
    #gmsh.option.setNumber("Mesh.Smoothing", 100)
    #gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.1)  # Smaller value for finer mesh
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.1)  # Adjusted maximum length

   
    # Generate mesh
    gmsh.model.mesh.generate(3)  # 2 for surface meshing, 3 for volume meshing if needed

    # Optionally refine mesh
    #gmsh.model.mesh.refine()  # Uncomment if additional refinement is needed after initial generation

    # Save mesh
    gmsh.write(mesh_file)   
    gmsh.finalize()

# Create mesh from STEP file
create_mesh(step_file)
