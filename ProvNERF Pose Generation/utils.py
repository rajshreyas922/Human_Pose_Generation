import torch

def parse_obj_file(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # Vertex data
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return vertices

def obj_to_tensor(file_path):
    vertices = parse_obj_file(file_path)
    vertex_tensor = torch.tensor(vertices)
    return vertex_tensor

def save_obj_file(vertices, file_path, translation):
    with open(file_path, 'w') as file:
        file.write("# Wavefront OBJ file\n")
        for vertex in vertices:
            translated_vertex = vertex + torch.tensor(translation)
            file.write(f"v {translated_vertex[0]} {translated_vertex[1]} {translated_vertex[2]}\n")

def tensor_to_obj_files(tensor, base_file_path, n_columns=5):
    num_objects = tensor.shape[0]
    for i in range(num_objects):
        vertices = tensor[i]
        # Calculate the row and column for the current index
        row = i // n_columns
        col = i % n_columns
        # Translation based on row and column to arrange in a grid
        translation = (col * 3, row * 2, 0)  # Adjust spacing as needed (10 units here)
        file_path = f"{base_file_path}_{i+1}.obj"
        save_obj_file(vertices, file_path, translation)