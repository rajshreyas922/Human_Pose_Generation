import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
def pos_encoder(x, L):

    _, n = x.shape

    encoding = []
    alpha = 1.0

    for i in range(n):
        for l in range(L):
            if l > 6:
                alpha = 0.5
            encoding.append(alpha*torch.sin(1.1*(2**l) * torch.pi * x[:, i:i+1]))
            encoding.append(alpha*torch.cos(1.1*(2**l) * torch.pi * x[:, i:i+1]))
    encoded_x = torch.cat(encoding, dim=-1)
    return encoded_x

def pos_encoder_decaying(x, L, decay_factor=1.0):

    _, n = x.shape
    encoding = []

    for i in range(n):  # Iterate through each input dimension
        for l in range(L):  # Iterate through each frequency level
            # Calculate the base frequency for this level
            # (Using 1.0 instead of 1.1 here, closer to original NeRF, but adjust if needed)
            # freq = (2**l) * torch.pi
            freq = 1.1 * (2**l) * torch.pi # Keeping your original 1.1 factor

            # Select the i-th dimension
            x_slice = x[:, i:i+1]

            # Calculate the weight for this frequency level
            # Weight decreases exponentially as 'l' increases if decay_factor < 1
            weight = decay_factor ** l

            # Calculate and append weighted sin and cos components
            encoding.append(weight * torch.sin(freq * x_slice))
            encoding.append(weight * torch.cos(freq * x_slice))

    # Concatenate all components along the feature dimension
    encoded_x = torch.cat(encoding, dim=-1)
    return encoded_x

def generate_line(n, sign = "+"):
    t = torch.linspace(-3, 3, n)
    x = t.view(-1, 1)
    if sign == "+":
        y = t.view(-1, 1)
    else:
        y = -t.view(-1, 1)
    return torch.cat((x, y), dim=1).unsqueeze(0)


def generate_line_small(n, sign = "+"):
    t = torch.linspace(-1, 1, n)
    x = t.view(-1, 1)
    if sign == "+":
        y = t.view(-1, 1)
    else:
        y = -t.view(-1, 1)
    return torch.cat((x, y), dim=1).unsqueeze(0)

def generate_parabola(n, sign = "+"):
    t = torch.linspace(-10, 10, n)
    x = (t).view(-1, 1)
    if sign == "+":
        y =((t**2).view(-1, 1) + 5)/5
    else:
        y = (-(t**2).view(-1, 1) - 5)/5
    return torch.cat((x, y), dim=1).unsqueeze(0)

def generate_circle(n, r = 1):
    t = torch.linspace(0, 2 * torch.pi - 1e-3, n)
    x = (r * torch.cos(t)).view(-1, 1)
    y = (r * torch.sin(t)).view(-1, 1)
    return torch.cat((x, y), dim=1).unsqueeze(0)


def generate_sphere(n, a = 1, b = 1, c = 1):
    t1 = torch.linspace(0, 2*torch.pi - 1e-3, n)
    t2 = torch.linspace(0, torch.pi - 1e-3, n)
    grid_t1, grid_t2 = torch.meshgrid((t1, t2), indexing='ij')
    # t = torch.stack((grid_t1, grid_t2), dim=-1).reshape(-1, 2).to(device)
    x = (a * torch.sin(grid_t1) * torch.cos(grid_t2)).view(-1, 1)
    y = (b * torch.sin(grid_t1) *  torch.sin(grid_t2)).view(-1, 1)
    z = (c * torch.cos(grid_t1)).view(-1, 1)
    return torch.cat((x, y, z), dim=1).unsqueeze(0)

def generate_donut(n, a = 1, b = 1, c = 1):
    t1 = torch.linspace(0, 2*torch.pi - 1e-3, n)
    t2 = torch.linspace(0, 2*torch.pi - 1e-3, n)
    grid_t1, grid_t2 = torch.meshgrid((t1, t2), indexing='ij')
    # t = torch.stack((grid_t1, grid_t2), dim=-1).reshape(-1, 2).to(device)
    x = ((torch.sin(grid_t1) + 2) * torch.cos(grid_t2)).view(-1, 1)
    y = ((torch.sin(grid_t1) + 2) *  torch.sin(grid_t2)).view(-1, 1)
    z = 0.5*(torch.cos(grid_t1)).view(-1, 1)
    return torch.cat((x, y, z), dim=1).unsqueeze(0)

def generate_3D_ellipse(n, r1 = 1, r2 = 0.5):
    t = torch.linspace(0, 2 * torch.pi, n+1)
    x = (r1 * torch.cos(t)).view(-1, 1)[:-1]
    y = (r2 * torch.sin(t)).view(-1, 1)[:-1]
    z = (0 * torch.sin(t)).view(-1, 1)[:-1]
    return torch.cat((x, y, z), dim=1).unsqueeze(0)


def generate_ellipse(n, r1 = 1, r2 = 0.5):
    t = torch.linspace(0, 2*torch.pi - 1e-3, n+1)
    x = (r1 * torch.cos(t)).view(-1, 1)[:-1]
    y = (r2 * torch.sin(t)).view(-1, 1)[:-1]
    return torch.cat((x, y), dim=1).unsqueeze(0)

def square_edges_ordered(a, n):
    """
    Generate a (1, n, 2) tensor representing the edges of a square with side length `a`, centered at (0, 0).
    The points are sorted in a counter-clockwise manner, starting from (a/2, 0) and ending at (a/2, 0),
    with the last set of points moving upward.

    Parameters:
        a (float): Length of the square's side.
        n (int): Total number of points to sample along the square's edges.

    Returns:
        torch.Tensor: A tensor of shape (1, n, 2) containing the points along the edges of the square.
    """
    # Ensure n is divisible by 4 (4 edges of the square)
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4 to evenly distribute points across all edges.")

    # Number of points per edge
    points_per_edge = n // 4
    half_a = a / 2

    # Right edge (first half): from (a/2, 0) to (a/2, a/2)
    right_edge_up = torch.stack([
        torch.full((points_per_edge // 2,), half_a),           # x-coordinates
        torch.linspace(0, half_a, points_per_edge // 2)        # y-coordinates
    ], dim=-1)

    # Top edge: from (a/2, a/2) to (-a/2, a/2)
    top_edge = torch.stack([
        torch.linspace(half_a, -half_a, points_per_edge),      # x-coordinates
        torch.full((points_per_edge,), half_a)                 # y-coordinates
    ], dim=-1)

    # Left edge: from (-a/2, a/2) to (-a/2, -a/2)
    left_edge = torch.stack([
        torch.full((points_per_edge,), -half_a),               # x-coordinates
        torch.linspace(half_a, -half_a, points_per_edge)       # y-coordinates
    ], dim=-1)

    # Bottom edge: from (-a/2, -a/2) to (a/2, -a/2)
    bottom_edge = torch.stack([
        torch.linspace(-half_a, half_a, points_per_edge),      # x-coordinates
        torch.full((points_per_edge,), -half_a)                # y-coordinates
    ], dim=-1)

    # Right edge (second half): from (a/2, -a/2) to (a/2, 0) (moving upward)
    right_edge_down = torch.stack([
        torch.full((points_per_edge // 2,), half_a),           # x-coordinates
        torch.linspace(-half_a, 0 + 1e-1, points_per_edge // 2)       # y-coordinates
    ], dim=-1)

    # Concatenate all edges in the desired order:
    # Right edge (up), Top edge, Left edge, Bottom edge, Right edge (down)
    square = torch.cat([right_edge_up, top_edge, left_edge, bottom_edge, right_edge_down], dim=0)

    # Add batch dimension (1, n, 2)
    square = square.unsqueeze(0)

    return square

def square_edges(a, n):
    """
    Generate a (1, n, 2) tensor representing the edges of a square with side length `a`, centered at (0, 0).
    The points are sorted in a counter-clockwise manner, starting from the bottom-left corner (-a/2, -a/2)
    and ending at the bottom-left corner (-a/2, -a/2).

    Parameters:
        a (float): Length of the square's side.
        n (int): Total number of points to sample along the square's edges.

    Returns:
        torch.Tensor: A tensor of shape (1, n, 2) containing the points along the edges of the square.
    """
    # Ensure n is divisible by 4 (4 edges of the square)
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4 to evenly distribute points across all edges.")

    # Number of points per edge
    points_per_edge = n // 4
    half_a = a / 2

    # Bottom edge: from (-a/2, -a/2) to (a/2, -a/2)
    bottom_edge = torch.stack([
        torch.linspace(-half_a+1, half_a, points_per_edge),      # x-coordinates
        torch.full((points_per_edge,), -half_a)                # y-coordinates
    ], dim=-1)

    # Right edge: from (a/2, -a/2) to (a/2, a/2)
    right_edge = torch.stack([
        torch.full((points_per_edge,), half_a),                # x-coordinates
        torch.linspace(-half_a, half_a, points_per_edge)       # y-coordinates
    ], dim=-1)

    # Top edge: from (a/2, a/2) to (-a/2, a/2)
    top_edge = torch.stack([
        torch.linspace(half_a, -half_a, points_per_edge),      # x-coordinates
        torch.full((points_per_edge,), half_a)                 # y-coordinates
    ], dim=-1)

    # Left edge: from (-a/2, a/2) to (-a/2, -a/2)
    left_edge = torch.stack([
        torch.full((points_per_edge,), -half_a),               # x-coordinates
        torch.linspace(half_a, -half_a, points_per_edge)       # y-coordinates
    ], dim=-1)

    # Concatenate all edges in the desired order:
    # Bottom edge, Right edge, Top edge, Left edge
    square = torch.cat([bottom_edge, right_edge, top_edge, left_edge], dim=0)

    # Add batch dimension (1, n, 2)
    square = square.unsqueeze(0)

    return square


def generate_data(n, sign = "+"):
    line = generate_line(n, sign)
    line2 = generate_line(n, sign = '-')
    parabola = generate_parabola(n, sign)
    parabola1 = generate_parabola(n, sign = '-')
    circle = generate_circle(n)
    circle1 = generate_circle(n, r=2)
    e1 = generate_ellipse(n, r1 = 1.5, r2 = 3)
    e2 = generate_ellipse(n, r1 = 3, r2 = 1.5)
    e4 =  generate_ellipse(n, r1 = 5, r2 = 5)
    e5 = square_edges_ordered(a = 10, n = n)
    e3 = generate_ellipse(n, r1 = 2.5, r2 = 2.5)
    curves = torch.concat((e1, e2))
    #curves = torch.concat((curves, e3))

    # start_x = -3.0
    # end_x = 3.0
    # x_coords = torch.linspace(start_x, end_x, steps=40)

    # # The y-coordinate is always 0 for a horizontal line
    # y_coords = torch.zeros_like(x_coords)

    # # Combine x and y coordinates into a (40, 2) tensor
    # points = torch.stack([x_coords, y_coords], dim=1)

    # # Add a batch dimension to make the tensor shape (1, 40, 2)
    # horizontal_line = points.unsqueeze(0)
    # curves = torch.concat((curves, horizontal_line))
    # #curves = torch.cat([e1, e2, curves], dim=0)
    # for i in range(11,30):
    #     # for j in range(1,5):
    #     e1 = generate_ellipse(n, r1 = i/2, r2 = i/4)
    #     curves = torch.concat((curves, e1))
    return curves

def generate_cube(n, a):
    n = n**2
    R = int(round(math.sqrt(n / 6.0)))
    R = max(R, 1)  # ensure at least 1

    half = a / 2
    coords = torch.linspace(-half, half, R)  # shape: (R,)

    Y, Z = torch.meshgrid(coords, coords, indexing='ij')  # (R, R)
    face_x_min = torch.stack([
        torch.full_like(Y, -half),  # x = -half
        Y,
        Z
    ], dim=-1).reshape(-1, 3)  # (R*R, 3)
    face_x_max = torch.stack([
        torch.full_like(Y, half),   # x = +half
        Y,
        Z
    ], dim=-1).reshape(-1, 3)

    # Face y = -half and y = +half (vary x, z)
    X, Z = torch.meshgrid(coords, coords, indexing='ij')
    face_y_min = torch.stack([
        X,
        torch.full_like(X, -half),  # y = -half
        Z
    ], dim=-1).reshape(-1, 3)
    face_y_max = torch.stack([
        X,
        torch.full_like(X, half),   # y = +half
        Z
    ], dim=-1).reshape(-1, 3)

    # Face z = -half and z = +half (vary x, y)
    X, Y = torch.meshgrid(coords, coords, indexing='ij')
    face_z_min = torch.stack([
        X,
        Y,
        torch.full_like(X, -half)   # z = -half
    ], dim=-1).reshape(-1, 3)
    face_z_max = torch.stack([
        X,
        Y,
        torch.full_like(X, half)    # z = +half
    ], dim=-1).reshape(-1, 3)

    # Concatenate all faces
    faces = torch.cat([
        face_x_min, face_x_max,
        face_y_min, face_y_max,
        face_z_min, face_z_max
    ], dim=0)  # shape: (6*R^2, 3)

    # Add batch dimension
    return faces.unsqueeze(0)

def generate_3D_data(n, sign = "+"):
    #e1 = generate_ellipse(n, r1 = 1.5, r2 = 3)

    curves =  generate_cube(n, a = 4)[:,0:int(n**2),:]
    e1 = generate_sphere(n, a = 2, b = 2, c=2)[:,0:int(n**2),:]

    e2 = generate_sphere(n, a = 5, b = 5, c=5)
    # for i in range(11,30):
    #     # for j in range(1,5):
    #     e1 = get_cube_face_points_fixed_spacing(n**2, a = i/2)[:,0:int(n**2),:]
    #     curves = torch.concat((curves, e1))
    # for i in range(10,30):
    #     # for j in range(1,5):
    #     e1 = generate_sphere(n, a = i/4, b = i/4, c=i/4)
    curves = torch.concat((curves, e1))

    return curves

def plot_data(data, mode = '3D'):
    data = data.to("cpu").detach().numpy()

    if mode == '3D':
        fig = plt.figure(figsize=(15, 5))  # Create a wide figure to accommodate three subplots

        # First subplot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.view_init(elev=20, azim=30)  # Set the viewing angle
        for i in range(data.shape[0]):
            ax1.scatter(data[i, :, 0], data[i, :, 1], data[i, :, 2], alpha=1.0, s=0.5)
        ax1.set_title('View 1')

        # Second subplot
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.view_init(elev=40, azim=60)  # Change the viewing angle
        for i in range(data.shape[0]):
            ax2.scatter(data[i, :, 0], data[i, :, 1], data[i, :, 2], alpha=1.0, s=0.5)
        ax2.set_title('View 2')

        # Third subplot
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.view_init(elev=60, azim=90)  # Change the viewing angle again
        for i in range(data.shape[0]):
            ax3.scatter(data[i, :, 0], data[i, :, 1], data[i, :, 2], alpha=1.0, s=0.5)
        ax3.set_title('View 3')

        plt.tight_layout()
        plt.show()
    else:
        for i in range(data.shape[0]):
            plt.scatter(data[i, :, 0], data[i, :, 1], s=1.5)
        plt.show()