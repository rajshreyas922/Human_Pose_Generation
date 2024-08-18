from vedo import *
from utils import *
from model import *
epochs = 20000
staleness = 15
num_Z_samples = 40
lr = 0.01
xdim = 2
zdim = 10
num_points = int(np.power(6890, 1/xdim))
min_b = -1
max_b = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t1 = torch.linspace(min_b, max_b, num_points)
t2 = torch.linspace(min_b, max_b, num_points)
grid_t1, grid_t2 = torch.meshgrid((t1, t2), indexing='ij')
t = torch.stack((grid_t1, grid_t2), dim=-1).reshape(-1, xdim).to(device)

provnerf_results_directory = "ProvNERF Pose Generation\\ProvNERF results"
provnerf_pose_directory = "ProvNERF Pose Generation"

meshes = []

colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]

specific_files = ["walk_kinda.obj", "human_body_model.obj"]

H_t = H_theta(input_dim=zdim+xdim, output_dim=3).to(device)
H_t.load_state_dict(torch.load('C:/Users/rajsh/Desktop/Human_Pose_Generation/H_t_30k_epochs.pth'))
H_t.eval()

num_samples = 30
outputs = torch.empty((num_samples, num_points**xdim, 3))
for i in range(num_samples):
    Zs = generate_NN_latent_functions(1, xdim, zdim)[0].to(device)
    Z = Zs(t)
    Z = torch.cat((Z, t), dim = 1)
    out = H_t(Z)
    outputs[i] = out
print("Output shape", outputs.shape)


base_file_path = 'ProvNERF Pose Generation\ProvNERF results/vertex_object'
tensor_to_obj_files(outputs, base_file_path)

for i, filename in enumerate(os.listdir(provnerf_results_directory), start=len(specific_files)):
    if filename.endswith(".obj"):  # Check if the file is an OBJ file
        file_path = os.path.join(provnerf_results_directory, filename)
        mesh = Mesh(file_path)
        mesh.point_size(5)  # Set point size
        
        # Assign a color from the list
        mesh.color(colors[i % len(colors)])
        
        meshes.append(mesh)  # Add the mesh to the list

# Show all meshes in the same scene with axes
show(meshes, axes=True)