import open3d as o3d
import os
import numpy as np

# Function to load OBJ as point cloud
def load_obj_as_point_cloud(obj_file):
    vertices = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):  # Vertex line
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))

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

    # Load the point cloud from the OBJ file
    point_cloud = load_obj_as_point_cloud(input_obj_file)

    # Remove statistical outliers
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    point_cloud = point_cloud.select_by_index(ind)

    # Downsample the point cloud (optional, for efficiency)
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)

    # Estimate normals
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    # Orient normals consistently
    point_cloud.orient_normals_consistent_tangent_plane(k=10)

    # Perform Poisson Surface Reconstruction
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=10, width=0, scale=1.1, linear_fit=False
        )
    except RuntimeError as e:
        print(f"Failed to reconstruct mesh for {input_obj_file}: {e}")
        continue

    # Convert densities to a NumPy array
    densities_np = np.asarray(densities)

    # Compute a robust threshold (e.g., 10th percentile)
    threshold = np.quantile(densities_np, 0.1)  # Keep top 90% of vertices

    # Remove low-density vertices
    vertices_to_remove = densities_np < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # # Simplify the mesh (optional)
    # mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)

    # # Smooth the mesh (optional)
    # mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)

    # # Fill small holes in the mesh
    # mesh = mesh.fill_holes(hole_size=0.1)

    # Save the reconstructed mesh to the output file
    o3d.io.write_triangle_mesh(output_obj_file, mesh)

    print(f"Processed and saved mesh to {output_obj_file}")

print("All files processed.")