import cadquery as cq
import gmsh


step_file = "wing.step"



def create_mesh(step_file, mesh_file='model.msh'):
    gmsh.initialize()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)  

    # Load STEP file
    gmsh.model.occ.importShapes(step_file)
    gmsh.model.occ.synchronize()

    # Define meshing parameters
    #gmsh.option.setNumber("Mesh.Algorithm", 6)  # Specify mesh algorithm
    #gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.setOrder(1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1)

    # Generate mesh
    gmsh.model.mesh.generate(3)  # 3 for 3D meshing

    # Save mesh
    gmsh.write(mesh_file)
    gmsh.finalize()

# Create mesh from STEP file
create_mesh(step_file)