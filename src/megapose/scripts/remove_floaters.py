import trimesh

def remove_floating_pieces(mesh_path, output_path):
    """
    Removes floating pieces from a 3D mesh in .ply format.
    
    Parameters:
        mesh_path (str): Path to the input 3D mesh in .ply format.
        output_path (str): Path to save the output mesh with floating pieces removed.
    """
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_path)
    
    # Check if the mesh is watertight (manifold)
    if not mesh.is_watertight:
        print("Warning: The mesh is not watertight. This might affect the removal of floating pieces.")

    # Find the connected components in the mesh
    connected_components = mesh.split(only_watertight=False)
    
    # Find the largest connected component (main body)
    largest_component = None
    max_faces = 0
    for component in connected_components:
        if len(component.faces) > max_faces:
            max_faces = len(component.faces)
            largest_component = component
    
    # Save the largest component to the output file
    largest_component.export(output_path)

# Example usage
input_mesh_path = '/home/andrewg/pose/megapose6d/data/examples/hammer03/meshes/obj_000003/obj_000003.ply'
output_mesh_path = '/home/andrewg/pose/megapose6d/data/examples/hammer03/meshes/obj_000003/obj_000003.ply'

remove_floating_pieces(input_mesh_path, output_mesh_path)
