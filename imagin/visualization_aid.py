import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from skimage.transform import resize


def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def show_histogram(values):
    n, bins, patches = plt.hist(values.reshape(-1), 50, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for c, p in zip(normalize(bin_centers), patches):
        plt.setp(p, 'facecolor', cm.viridis(c))

    plt.show()
    
def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def plot_cube(cube, angle=320, img_dim=50):
    cube = normalize(cube)
    
    facecolors = cm.viridis(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=img_dim*2)
    ax.set_ylim(top=img_dim*2)
    ax.set_zlim(top=img_dim*2)
    
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()
    
def preprocess_for_plot3d(img_3d, img_dim=50, show_histogram=False):
    transformed = np.clip(scale_by(np.clip(normalize(img_3d.squeeze())-0.1, 0, 1)**0.4, 2)-0.1, 0, 1)
    if show_histogram:
        show_histogram(transformed)
    resized = resize(transformed, (img_dim, img_dim, img_dim), mode='constant')
    return resized

def scale_by(arr, fac):
    mean = np.mean(arr)
    return (arr-mean)*fac + mean
