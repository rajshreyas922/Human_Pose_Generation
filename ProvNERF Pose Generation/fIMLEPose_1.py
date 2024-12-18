import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from vedo import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)

epochs = 5000
staleness = 15
num_Z_samples = 20
lr = 0.001
xdim = 2
zdim = 5
num_points = int(np.power(6890, 1/xdim))

H_t = H_theta(input_dim=zdim+xdim, output_dim=3).to(device)
optimizer = optim.Adam(H_t.parameters(), lr=lr)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.01)
losses = []


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

min_b = -1
max_b = 1

if xdim == 1:
    t = torch.linspace(-1, -1, num_points).to(device)
if xdim == 2:
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

# x = points[0, 0:1000, 0].to('cpu')
# y = points[0, 0:1000, 1].to('cpu')
# z = points[0, 0:1000, 2].to('cpu')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z, c='b', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# ax.set_title('3D Scatter Plot')

# plt.show()

# exit()



for e in tqdm(range(epochs)):
    with torch.no_grad():
        if e % staleness == 0:
            Zs = generate_NN_latent_functions(num_Z_samples, xdim, zdim, lambda_value=1)
            Zxs = torch.empty((num_Z_samples, num_points**xdim, zdim+xdim)).to(device)
            
            for i, model in enumerate(Zs):
                model = model.to(device)
                Zx = model(t)
                Zx = torch.cat((Zx, t), dim = 1)
                Zxs[i] = Zx
            generated = H_t(Zxs) 
            imle_nns = torch.tensor([find_nns(d, generated) for d in points], dtype=torch.long)
            imle_nn_z = [Zs[idx] for idx in imle_nns]

    optimizer.zero_grad()
    imle_transformed_points = torch.empty((points.shape[0], num_points**xdim, zdim+xdim)).to(device)
    for i, model in enumerate(imle_nn_z):
        model = model.to(device)
        Zx = model(t)  
        Zx = torch.cat((Zx, t), dim = 1)
        imle_transformed_points[i] = Zx
    outs = H_t(imle_transformed_points) 

    loss = f_loss(outs, points)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()
    #scheduler.step()

print(f_loss(outs, points))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.show()
print(losses.pop())



if xdim == 2:
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

outputs = torch.empty((15, num_points**xdim, 3))
for i in range(15):
    Zs = generate_NN_latent_functions(1, xdim, zdim)[0].to(device)
    Z = Zs(t)
    Z = torch.cat((Z, t), dim = 1)
    out = H_t(Z)
    outputs[i] = out

print("Output shape", outputs.shape)

def save_obj_file(vertices, file_path):
    with open(file_path, 'w') as file:
        file.write("# Wavefront OBJ file\n")
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

def tensor_to_obj_files(tensor, base_file_path):
    num_objects = tensor.shape[0]
    for i in range(num_objects):
        vertices = tensor[i]
        file_path = f"{base_file_path}_{i+1}.obj"
        save_obj_file(vertices, file_path)





