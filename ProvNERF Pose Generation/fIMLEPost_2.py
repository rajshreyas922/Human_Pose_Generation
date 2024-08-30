import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from vedo import *
from model import *
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

epochs = 10000
staleness = 15
num_Z_samples = 40
lr = 0.001
xdim = 2
zdim = 15
num_points = int(np.power(6890, 1/xdim))
min_b = 0.5
max_b = 1

H_t = H_theta(input_dim=zdim+xdim, output_dim=3).to(device)
optimizer = optim.Adam(H_t.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
losses = []

if xdim == 1:
    t = torch.linspace(-1, 1, num_points).to(device)
if xdim == 2:
    #(-2 to 2)
    t1 = torch.linspace(min_b, max_b, num_points)
    t2 = torch.linspace(min_b, max_b, num_points)
    grid_t1, grid_t2 = torch.meshgrid((t1, t2), indexing='ij')
    t = torch.stack((grid_t1, grid_t2), dim=-1).reshape(-1, xdim).to(device)
elif xdim == 3:
    t1 = torch.linspace(min_b, max_b, num_points)
    t2 = torch.linspace(min_b, max_b, num_points)
    t3 = torch.linspace(min_b, max_b, num_points)
    grid_t1, grid_t2, grid_t3 = torch.meshgrid((t1, t2, t3), indexing='ij')
    t = torch.stack((grid_t1, grid_t2, grid_t3), dim=-1).reshape(-1, xdim).to(device)

vertex_tensor = obj_to_tensor('ProvNERF Pose Generation\walk_kinda.obj').unsqueeze(0)
#vertex_tensor_2 = obj_to_tensor('ProvNERF Pose Generation\\not_Tpose_pointcloud.obj').unsqueeze(0)
vertex_tensor_3 = obj_to_tensor('ProvNERF Pose Generation\\human_body_model.obj').unsqueeze(0)
vertex_tensor_main = torch.concat((vertex_tensor, vertex_tensor_3), dim=0)
points = vertex_tensor_main[:, :num_points**xdim,:].to(device)
print("t", t.shape)
print("Points", points.shape)


for e in tqdm(range(epochs)):
    with torch.no_grad():
        if e % staleness == 0:
            H_t.eval()
            Zxs = torch.empty((num_Z_samples, num_points**xdim, zdim+xdim)).to(device)

            Zs = generate_NN_latent_functions(num_samples=num_Z_samples, xdim=xdim,zdim=zdim, lambda_value=1)
            for i, model in enumerate(Zs):
                model = model.to(device)
                Zxs[i] = torch.cat((model(t), t), dim=1).to(device)
            #print(Zxs.cpu().numpy())
            #Zxs = matrices_transform_input(x, num_Z_samples, zdim, mean=0.0, std=1.0).to(device)

            generated = H_t(Zxs) 
            generated = generated.to(device)

            imle_nns = torch.tensor([find_nns(d, generated) for d in points], dtype=torch.long)    

            imle_transformed_points = torch.empty((points.shape[0], num_points, zdim+xdim)).to(device)
            imle_transformed_points = Zxs[imle_nns]

            H_t.train()

    optimizer.zero_grad()
    outs = H_t(imle_transformed_points)
    loss = f_loss(outs, points)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    #scheduler.step()

torch.save(H_t.state_dict(), 'C:/Users/rajsh/Desktop/Human_Pose_Generation/H_t_last_epoch.pth')
print(f_loss(outs, points))
plt.figure(figsize=(15, 5))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
print(losses.pop())


if xdim == 2:
    t1 = torch.linspace(min_b,max_b, num_points)
    t2 = torch.linspace(min_b,max_b, num_points)
    grid_t1, grid_t2 = torch.meshgrid((t1, t2), indexing='ij')
    t = torch.stack((grid_t1, grid_t2), dim=-1).reshape(-1, xdim).to(device)
elif xdim == 3:
    t1 = torch.linspace(min_b, max_b, num_points)
    t2 = torch.linspace(min_b, max_b, num_points)
    t3 = torch.linspace(min_b, max_b, num_points)
    grid_t1, grid_t2, grid_t3 = torch.meshgrid((t1, t2, t3), indexing='ij')
    t = torch.stack((grid_t1, grid_t2, grid_t3), dim=-1).reshape(-1, xdim).to(device)

outputs = torch.empty((15, num_points**xdim, 3))
for i in range(15):
    Zs = generate_NN_latent_functions(1, xdim, zdim)[0].to(device)
    Z = Zs(t)
    Z = torch.cat((Z, t), dim = 1)
    out = H_t(Z)
    outputs[i] = out
print("Output shape", outputs.shape)


base_file_path = 'ProvNERF Pose Generation\ProvNERF results/vertex_object'
tensor_to_obj_files(outputs, base_file_path) #save points clouds

# Directories containing the OBJ files
provnerf_results_directory = "ProvNERF Pose Generation\\ProvNERF results"
provnerf_pose_directory = "ProvNERF Pose Generation"

# List to store the meshes
meshes = []

# Define a list of colors (you can add more colors if needed)
colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple"]

# Add specific meshes from the "ProvNERF Pose Generation" directory
specific_files = ["walk_kinda.obj", "human_body_model.obj"]
# for i, filename in enumerate(specific_files):
#     file_path = os.path.join(provnerf_pose_directory, filename)
#     mesh = Mesh(file_path)
#     mesh.point_size(7)
#     mesh.color('black')  # Assign a color
#     meshes.append(mesh)

# Loop through all OBJ files in the "ProvNERF results" directory
for i, filename in enumerate(os.listdir(provnerf_results_directory), start=len(specific_files)):
    if filename.endswith(".obj"):  # Check if the file is an OBJ file
        file_path = os.path.join(provnerf_results_directory, filename)
        mesh = Mesh(file_path)
        mesh.point_size(7)  # Set point size
        
        # Assign a color from the list
        mesh.color(colors[i % len(colors)])
        
        meshes.append(mesh)  # Add the mesh to the list

# Show all meshes in the same scene with axes
show(meshes, axes=True)