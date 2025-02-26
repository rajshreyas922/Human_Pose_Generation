import trimesh
import os

# Define the input and output directories
input_dir = "testing_out/Generations_try_14/generated_objs"
output_dir = "testing_out/Generations_try_14/generated_meshes"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through all files from generated_points_0.obj to generated_points_14.obj
for i in range(15):  # From 0 to 14
    input_obj_file = os.path.join(input_dir, f"generated_points_{i}.obj")
    output_obj_file = os.path.join(output_dir, f"generated_mesh_{i}.obj")

    # Check if the input file exists
    if not os.path.exists(input_obj_file):
        print(f"File {input_obj_file} does not exist. Skipping...")
        continue

    # Step 1: Load the .obj file
    mesh = trimesh.load(input_obj_file)

    # Extract the vertices (points) from the loaded mesh
    points = mesh.vertices

    # Step 2: Create a mesh from the points using Convex Hull
    convex_mesh = trimesh.Trimesh(vertices=points).convex_hull

    # Step 3: Save the resulting mesh as a new .obj file
    convex_mesh.export(output_obj_file)

    print(f"Processed and saved mesh to {output_obj_file}")

print("All files processed.")