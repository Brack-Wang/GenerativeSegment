import numpy as np
from matplotlib import pyplot as plt
from tifffile import TiffFile
import math
from mpl_toolkits.mplot3d import Axes3D 
import os
from skimage.draw import line_nd
from tifffile import imwrite
from plyfile import PlyData, PlyElement
import numpy as np
from tifffile import imwrite
from skimage.draw import line_nd
from collections import defaultdict
import os
import cv2
import plyfile
from skimage.measure import marching_cubes
import plyfile
from collections import defaultdict
from scipy.ndimage import binary_dilation
import numpy as np
from skimage.draw import line_nd
from scipy.ndimage import gaussian_filter


def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)



class NtracerParser:
    def __init__(self, tracing_file):
        """
        Initialize the NtracerParser with the path to the tracing file.

        Args:
            tracing_file (str): Path to the tracing file containing neurite points.
        """
        self.gs_path = tracing_file

    def _parse(self):
        """
        Parse the tracing file and extract neurite points.

        Returns:
            dict: A dictionary with keys as indices and values as dictionaries containing:
                  - 'neuron': Neuron ID
                  - 'type': Type ('neurite' or 'soma')
                  - 'points': A numpy array of points (x, y, z)
        """
        count = 0
        data_dict = {}

        with open(self.gs_path, 'r') as f:
            # Skip the first 6 header lines
            for _ in range(6):
                f.readline()

            while True:
                line = f.readline().strip()
                if not line:
                    break

                if 'Neuron' in line:
                    # Extract neuron ID
                    neuron = int("".join(filter(str.isdigit, line)))

                if 'Neurite' in line or 'Soma' in line:
                    idx = [i for i in findall('POINT:', line)]
                    if idx:
                        points = np.zeros((len(idx), 3), dtype=np.int16)

                        for j in range(len(idx)):
                            # Extract point data between 'POINT:' markers
                            if j == len(idx) - 1:
                                point = line[idx[j]:]
                            else:
                                point = line[idx[j]:idx[j + 1]]

                            # Extract x, y, z coordinates
                            idx_p = [i for i in findall(' ', point)]
                            point_x = int(point[idx_p[1]:idx_p[2]]) - 1
                            point_y = int(point[idx_p[2]:idx_p[3]]) - 1
                            point_z = int(point[idx_p[3]:idx_p[4]]) - 1

                            points[j, :] = [point_x, point_y, point_z]

                        data_dict[count] = {
                            'neuron': neuron,
                            'type': 'soma' if 'Soma' in line else 'neurite',
                            'points': points,
                        }
                        count += 1

        return data_dict

    def load_neurite_points(self):
        """
        Load neurite points from the tracing file.

        Returns:
            dict: A dictionary with keys as indices and values as dictionaries containing:
                  - 'neuron': Neuron ID
                  - 'points': A numpy array of points (x, y, z)
        """
        data_dict = self._parse()
        neurite_dict = {}

        for idx, data in data_dict.items():
            if data['type'] == 'neurite':
                neurite_dict[idx] = {
                    'neuron_id': data['neuron'],
                    'type': data['type'],
                    'gs_points': data['points'],
                }

        return neurite_dict


def adaptive_interpolate_points(start, end, max_distance=2.0):
    """
    Interpolate points adaptively based on the distance between two points.

    Args:
        start (array): Starting point [x, y, z].
        end (array): Ending point [x, y, z].
        max_distance (float): Maximum distance between consecutive points.

    Returns:
        np.array: Array of interpolated points.
    """
    distance = np.linalg.norm(end - start)
    num_points = max(2, int(distance / max_distance))  # Ensure at least 2 points
    return np.linspace(start, end, num_points)



def create_mask_for_neuron_with_tunnels(gs_points, raw_image_shape, radius=5):
    """
    Create a 3D mask for a neuron with tunnels around the connected lines.
    The mask will have the same shape as the raw image.

    Args:
        gs_points (list of np.array): List of arrays containing 3D points in (x, y, z) dimensions.
        raw_image_shape (tuple): Shape of the raw image (Z, C, Y, X).
        radius (int): Radius of the tunnel around the lines in pixels.

    Returns:
        np.array: A 3D binary mask with dimensions (Z, 1, Y, X).
    """
    # Extract the Z, Y, X dimensions from the raw image shape
    Z, C, Y, X = raw_image_shape

    # Initialize an empty binary mask with shape (Z, 1, Y, X)
    mask = np.zeros((Z, 1, Y, X), dtype=np.uint8)

    # Iterate through the list of points for the neuron
    for points in gs_points:
        if len(points) < 2:
            continue  # Skip if there are not enough points to connect

        # Iterate through pairs of points and draw lines
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            # Convert (X, Y, Z) -> (Z, Y, X) for mask indexing
            start = start[[2, 1, 0]]
            end = end[[2, 1, 0]]

            # Interpolate points between start and end
            interpolated_points = adaptive_interpolate_points(start, end, max_distance=0.01)

            # Iterate through interpolated points and draw on the mask
            for point in interpolated_points:
                z, y, x = np.round(point).astype(int)  # Round to nearest integer
                if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:  # Ensure within bounds
                    mask[z, 0, y, x] = 1

    # Create a spherical structuring element for the tunnel
    struct_element = create_spherical_structure(radius)

    # Expand dimensions of struct_element to match (Z, 1, Y, X)
    struct_element = struct_element[:, None, :, :]  # Add the "C" dimension

    # Apply dilation to create tunnels
    mask_with_tunnels = binary_dilation(mask, structure=struct_element).astype(np.uint8)

    return mask_with_tunnels * 255


def create_spherical_structure(radius):
    """
    Create a spherical structuring element with the given radius.

    Args:
        radius (int): Radius of the sphere in pixels.

    Returns:
        np.array: A binary 3D array representing the spherical structuring element.
    """
    diameter = 2 * radius + 1
    struct = np.zeros((diameter, diameter, diameter), dtype=np.uint8)
    center = radius

    for z in range(diameter):
        for y in range(diameter):
            for x in range(diameter):
                if np.sqrt((z - center)**2 + (y - center)**2 + (x - center)**2) <= radius:
                    struct[z, y, x] = 1

    return struct


def create_mask_for_neuron(gs_points, raw_image_shape):
    """
    Create a 3D mask for a neuron by connecting points in gs_points sequentially.
    The mask will have the same shape as the raw image.

    Args:
        gs_points (list of np.array): List of arrays containing 3D points in (x, y, z) dimensions.
        raw_image_shape (tuple): Shape of the raw image (Z, C, Y, X).

    Returns:
        np.array: A 3D binary mask with dimensions (Z, 1, Y, X).
    """
    # Extract the Z, Y, X dimensions from the raw image shape
    Z, C, Y, X = raw_image_shape

    # Initialize an empty binary mask with shape (Z, 1, Y, X)
    mask = np.zeros((Z, 1, Y, X), dtype=np.uint8)

    # Iterate through the list of points for the neuron
    for points in gs_points:
        if len(points) < 2:
            continue  # Skip if there are not enough points to connect

        # Iterate through pairs of points and draw lines
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]

            # Convert (X, Y, Z) -> (Z, Y, X) for mask indexing
            start = start[[2, 1, 0]]
            end = end[[2, 1, 0]]

            # # Generate line coordinates between the two points
            # line_coords = line_nd(start, end, endpoint=True)
            # # Draw the line on the mask (Z, 1, Y, X)
            # mask[(line_coords[0], 0, line_coords[1], line_coords[2])] = 1

            # Interpolate points between start and end
            interpolated_points = adaptive_interpolate_points(start, end, max_distance=0.01)

            # Iterate through interpolated points and draw on the mask
            for point in interpolated_points:
                z, y, x = np.round(point).astype(int)  # Round to nearest integer
                if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:  # Ensure within bounds
                    mask[z, 0, y, x] = 1
    mask = mask * 255
    return mask

def spatial_segmentation_smoothing(generated_image, sigma=2):
    """
    Apply spatial segmentation and smoothing to refine the sharp edges.

    Args:
        generated_image (np.array): The generated image with sharp edges.
        sigma (float): Standard deviation for Gaussian smoothing.

    Returns:
        np.array: Smoothed image with refined edges.
    """
    # Detect edges using Canny
    edges = cv2.Canny(generated_image, 50, 150)
    
    # Dilate the edges to create a mask for sharp regions
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert mask to binary
    edge_mask = (dilated_edges > 0).astype(np.uint8)
    
    # Smooth the entire image
    smoothed_image = gaussian_filter(generated_image.astype(np.float32), sigma=sigma)
    
    # Blend smoothed regions with original image
    refined_image = generated_image * (1 - edge_mask) + smoothed_image * edge_mask
    
    return refined_image.astype(generated_image.dtype)

if __name__ == '__main__':
    # Define input and output paths
    data_dir  = "/data/wangfeiran/code/brainbow/datasets/compress/"
    raw_image_folder = data_dir + "raw/"
    skeleton_folder = data_dir + "skeleton/"
    masks_output_folder = "/data/wangfeiran/code/brainbow/datasets/segmentation_data_8/masks"

    # Create the output folder if it doesn't exist
    os.makedirs(masks_output_folder, exist_ok=True)

    # Get a list of all raw image files
    raw_files = sorted([f for f in os.listdir(raw_image_folder) if f.endswith('.tif')])

    # Process each file
    for raw_file in raw_files:
        # Define the corresponding skeleton file
        skeleton_file = raw_file + "-data.txt"
        masks_output =  os.path.join(masks_output_folder, raw_file[:-4])
        os.makedirs(masks_output, exist_ok=True)
        raw_image_path = os.path.join(raw_image_folder, raw_file)
        skeleton_path = os.path.join(skeleton_folder, skeleton_file)

        # Check if the skeleton file exists
        if not os.path.exists(skeleton_path):
            print(f"Skeleton file not found for {raw_file}, skipping.")
            continue

        # Load the raw image
        with TiffFile(raw_image_path) as tif:
            raw_image = tif.asarray()
        print(f"Raw image {raw_file} loaded with shape: {raw_image.shape}")

        # Initialize the parser
        parser = NtracerParser(skeleton_path)

        # Load neurite points
        neurite_data = parser.load_neurite_points()

        # Combine gs_points with same neuron together
        combined_dict = defaultdict(lambda: {'gs_points': []})
        for idx, neuron_data in neurite_data.items():
            neuron_id = neuron_data['neuron_id']
            if neuron_data['type'] == 'neurite':
                combined_dict[neuron_id]['gs_points'].append(neuron_data['gs_points'])
        combined_dict = dict(combined_dict)

        # Iterate each neuron, save mask
        for neuron_id, neuron_data in combined_dict.items():
            print("Processing neuron_id:", neuron_id)
            gs_points = neuron_data['gs_points']  # List of point arrays

            # Create a 3D mask for the neuron
            # neuron_mask = create_mask_for_neuron(gs_points, raw_image.shape)
            # neuron_mask = spatial_segmentation_smoothing(neuron_mask, sigma=2)
            neuron_mask = create_mask_for_neuron_with_tunnels(gs_points, raw_image.shape, radius=8)
            mask_filename = os.path.join(masks_output, f"neuron_{neuron_id}_mask.tif")
            neuron_mask = neuron_mask.astype(np.uint16)
            imwrite(mask_filename, neuron_mask, dtype=neuron_mask.dtype, imagej=True)
            print(f"Saved mask for neuron_id {neuron_id} to {mask_filename}")
        print(f"Completed processing for {raw_file}")
