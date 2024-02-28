import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_0_0_64 = [
    [4.88E-01, 5.00E+00, -4.75E+01],
    [6.00E-01, 5.00E+00, -3.65E+01],
    [6.75E-01, 5.00E+00, -3.65E+01],
    [5.63E-01, 0.00E+00, -4.20E+01],
    [8.25E-01, 0.00E+00, -5.85E+01],
    [6.75E-01, 5.00E+00, -3.38E+01],
    [4.13E-01, 5.00E+00, -4.48E+01],
    [7.88E-01, 5.00E+00, -4.48E+01],
    [9.00E-01, 1.50E+01, -5.30E+01],
    [6.38E-01, 0.00E+00, -4.20E+01]
]

data_03_24 = [
    [1.05E+00, 4.50E+01, -7.50E+01],
    [8.63E-01, 6.00E+01, -5.85E+01],
    [9.00E-01, 8.00E+01, -4.20E+01],
    [9.00E-01, 7.50E+01, -4.20E+01],
    [7.13E-01, 8.00E+01, -4.20E+01],
    [7.88E-01, 7.50E+01, -4.20E+01],
    [5.25E-01, 8.00E+01, -4.20E+01],
    [9.38E-01, 5.50E+01, -6.40E+01],
    [6.38E-01, 1.05E+02, -4.20E+01],
    [1.05E+00, 7.00E+01, -5.30E+01]
]

data_04_72 = [
    [9.38E-01, 7.00E+01, -7.50E+01],
    [9.38E-01, 9.00E+01, -4.20E+01],
    [9.75E-01, 6.50E+01, -5.30E+01],
    [9.75E-01, 1.10E+02, -4.20E+01],
    [9.75E-01, 8.50E+01, -4.75E+01],
    [9.38E-01, 9.50E+01, -4.20E+01],
    [7.50E-01, 1.05E+02, -4.20E+01],
    [7.50E-01, 9.50E+01, -4.20E+01],
    [9.75E-01, 5.50E+01, -5.85E+01],
    [9.75E-01, 4.50E+01, -6.40E+01]
]

data_04_120 = [
    [9.38E-01, 30.0, -75.0],
    [9.38E-01, 90.0, -42.0],
    [7.50E-01, 75.0, -53.0],
    [7.50E-01, 85.0, -47.5],
    [7.88E-01, 80.0, -53.0],
    [8.63E-01, 85.0, -42.0],
    [8.63E-01, 75.0, -53.0],
    [9.38E-01, 80.0, -42.0],
    [8.25E-01, 60.0, -53.0],
    [9.38E-01, 65.0, -53.0]
]


mean0 = np.mean(data_0_0_64, axis=0)
mean1 = np.mean(data_03_24, axis=0)
mean2 = np.mean(data_04_72, axis=0)
mean3 = np.mean(data_04_120, axis=0)

means = []
means.append(mean0)
means.append(mean1)
means.append(mean2)
means.append(mean3)

print(means)

concatenated_array = np.concatenate((data_0_0_64, data_03_24, data_04_72, data_04_120), axis=0)
all_data = concatenated_array.reshape(4,10,3)

std_values = np.std(all_data, axis=0)
mean_std_deviation = np.mean((std_values))
print(all_data.shape)
# print('msd',mean_std_deviation.shape)

# Function to create a solid 3D sphere centered at a specific point
def create_sphere(radius, center):

    linesforglobe = 12
    u = np.linspace(0, 2 * np.pi, linesforglobe)
    v = np.linspace(0, np.pi, linesforglobe)
    u, v = np.meshgrid(u, v)
    x = center[0] + radius[0] * np.sin(v) * np.cos(u)
    y = center[1] + radius[1] * np.sin(v) * np.sin(u)
    z = center[2] + radius[2] * np.cos(v)
    return x, y, z

# Specify points
data_points = np.array([[0, 0, -64], [0.3, 24, -64], [0.4, 72, -64], [0.4, 120, -64]])

# Create subplots
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='3d'), figsize=(10, 8))

# the sphere are equal to the solution space.
radii = [[1.05, 140, -66], [1.05, 140, -66], [1.05, 140, -66], [1.05, 140, -66]]
spheres = [create_sphere(radius, center) for radius, center in zip(radii, data_points)]
referencepoint=[['g=0.0, b_e=0.0'],['g=0.0, b_e=24'],['g=0.3, b_e=72'],['g=0.4, b_e=120']]

# Plot spheres in subplots
for i in range(4):
    ax = axs[i // 2, i % 2]
    sphere = spheres[i]
    ax.plot_surface(*sphere, color='r', edgecolors=(0, 0, 1, 0.3), alpha=0.05)

    # Scatter plot data point for the current sphere
    ax.scatter(data_points[i, 0], data_points[i, 1], data_points[i, 2], color='red', s=50, label=referencepoint[i])

    # Scatter observed data
    ax.scatter(all_data[i, :, 0], all_data[i, :, 1], all_data[i, :, 2], color='green', label='Best 10 solutions')

    # Set labels
    ax.set_xlabel('g')
    ax.set_ylabel('b_e')
    ax.set_zlabel('Ele/Eli')

    # Set legend
    ax.legend()

plt.show()
