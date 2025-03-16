import os
import yaml
import argparse
import imageio
import numpy as np
from attrdict import AttrDict
from collections import OrderedDict
from tqdm import tqdm
import warnings
import torch
import time
import torch.nn as nn
from inference_provider import Provider_valid
from supervised_provider import Provider
from model.Mnet import MNet
import tifffile as tiff
import numpy as np
import os
from skimage.transform import resize
import numpy as np
import tifffile as tiff
import cv2
from skimage.morphology import remove_small_objects, closing, square
from skimage.util import img_as_ubyte

from aicspylibczi import CziFile
import numpy as np
import tifffile as tiff
import skimage.transform


os.environ['CUDA_VISIBLE_DEVICES'] = "6"
warnings.filterwarnings("ignore")


def resize_tiff(root_path, test_data, scale_factor=4):
    print("Step 1: Resizing TIFF file...")
    # Load the TIFF file
    tif_path = os.path.join(root_path, test_data[0])
    tif_data = tiff.imread(tif_path)

    # Print original shape
    print("Original shape:", tif_data.shape)

    tif_data = tif_data[:, 0, :, :] 

    # Resize to 1/4 of the original size
    new_shape = (tif_data.shape[0], tif_data.shape[1] // scale_factor, tif_data.shape[2] // scale_factor)
    resized_data = resize(tif_data, new_shape, anti_aliasing=True, preserve_range=True).astype(tif_data.dtype)

    print("Resized shape:", resized_data.shape)
    # Save the resized image to the same folder
    output_folder = os.path.join(root_path, "resize_result")
    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, "resized_" + os.path.basename(tif_path))
    tiff.imwrite(save_path, resized_data)

def czi_morf3_resize(root_path, test_data, scale_factor=4):
    print("Step 1: Load CZI and resizing...")

    file_path = os.path.join(root_path, test_data[0])
    output_folder = os.path.join(root_path, "resize_result")
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the .czi file
    czi = CziFile(file_path)

    # Get the shape and dimension order
    dims_shape = czi.get_dims_shape()  # Returns a dictionary with shape info

    # Read the image as a NumPy array
    image_array, shp = czi.read_image()

    # Check the type and shape of the loaded array
    print(f"Original Image array shape: {image_array.shape}")
    image_array = image_array[0, 0, 0, 1, :, :, :]   

    # Remove unnecessary dimensions and reshape to (Z, Y, X)
    image_array = image_array.squeeze()  # Removes singleton dimensions

    # Downscale the image to 1/4 of its original size
    scale_factor = 1/ scale_factor  # Reduce to 1/4 of original size
    new_shape = (
        image_array.shape[0],  # Keep the number of Z slices the same
        int(image_array.shape[1] * scale_factor),  # Scale Y
        int(image_array.shape[2] * scale_factor)   # Scale X
    )

    image_array_resized = skimage.transform.resize(
        image_array,
        new_shape,
        anti_aliasing=True,
        preserve_range=True
    ).astype(image_array.dtype)  # Convert back to original dtype

    print(f"Resized Image array shape: {image_array_resized.shape}")

    # Save as a .tif file
    save_path = os.path.join(output_folder, "resized_" + os.path.basename(test_data[0]).rsplit(".", 1)[0] + ".tif")
    tiff.imwrite(save_path, image_array_resized, dtype=image_array.dtype)



def inference(root_path, test_data):    
    print("Step 2: Inference...")
    data_path = os.path.join(root_path, "resize_result")
    base_name = os.path.basename(test_data[0]).rsplit(".", 1)[0]
    test_data = ["resized_" + base_name + ".tif"]
    print("test_data: ", test_data)
    result_path = os.path.join(root_path, "inference_result")
    os.makedirs(result_path, exist_ok=True)
    pth = '/data/wangfeiran/code/brainbow/segmentation/SegNeuron/Train_and_Inference/models/2025-02-05--20-35-55_SegNeuron/model-003000.ckpt'
    # pth = '/data/wangfeiran/code/brainbow/segmentation/SegNeuron/models/SegNeuronModel.ckpt'
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='SegNeuron', help='path to config file')
    args = parser.parse_args()
    cfg_file = args.cfg + '.yaml'
    # print('cfg_file: ' + cfg_file)
    with open('/data/wangfeiran/code/brainbow/segmentation/SegNeuron/Train_and_Inference/config/' + cfg_file, 'r') as f:
        cfg = AttrDict(yaml.safe_load(f))

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU='sub').cuda()
    checkpoint = torch.load(pth)
    new_state_dict = OrderedDict()
    state_dict = checkpoint['model_weights']
    for k, v in state_dict.items():
        name = k.replace('module.', '') if 'module' in k else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()

    model.eval()
    valid_provider = Provider_valid(cfg, data_path, test_data)
    # train_provider = Provider('train', cfg)
    criterion = nn.BCELoss()
    dataloader = torch.utils.data.DataLoader(valid_provider, batch_size=1, num_workers=0,
                                             shuffle=False, drop_last=False, pin_memory=True)

    pbar = tqdm(total=len(valid_provider))
    losses_valid = []
    for k, batch in enumerate(dataloader, 0):
        inputs, target, _ = batch
        inputs = inputs.cuda()
        target = target.cuda()
        with torch.no_grad():
            pred, bound = model(inputs)
        valid_provider.add_vol(np.squeeze(pred.data.cpu().numpy()))
        valid_provider.add_bound(np.squeeze(bound.data.cpu().numpy()))
        pbar.update(1)
    pbar.close()

    out_affs = valid_provider.get_results()
    out_bounds = valid_provider.get_results_bound()

    gt_affs = valid_provider.get_gt_affs()
    gt_seg = valid_provider.get_gt_lb()
    valid_provider.reset_output()

    os.makedirs(result_path, exist_ok=True)
    np.save(result_path + '/neuron' + base_name, out_affs)
    imageio.volwrite(result_path + '/neuron' + base_name + ".tif", out_bounds.squeeze())



def threshold_intensity(root_path, test_data, otsu_ratio):
    print("Step 3: Thresholding intensity...")
    # Load the 3D TIFF file
    base_name = os.path.basename(test_data[0]).rsplit(".", 1)[0]
    input_path = root_path + "/resize_result/resized_" + base_name + ".tif"
    output_folder = root_path + "/intensity_result"
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder + "/intensity_"  + base_name + ".tif"

    image_3d = tiff.imread(input_path)  # Shape: [Z, Y, X]

    # Create an empty array for storing the segmented results
    segmented_3d = np.zeros_like(image_3d, dtype=np.uint16)

    # Process each slice in the Z direction
    for i in range(image_3d.shape[0]):
        img = image_3d[i].astype(np.uint16)  # Ensure it is a NumPy uint16 array

        # Apply Gaussian blur to remove noise
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply Otsu’s thresholding to segment neurons
        _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 计算 Otsu 阈值
        otsu_threshold, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 降低 Otsu 阈值，保留更多的信息；反之，保留更少的信息
        lower_threshold = max(0, int(otsu_threshold * otsu_ratio ))  

        # 重新应用手动阈值
        _, img_thresh = cv2.threshold(img_blur, lower_threshold, 255, cv2.THRESH_BINARY)


        # Convert to a proper NumPy array and apply morphological closing
        img_morph = closing(img_thresh.astype(bool), square(3))  # Convert to binary for skimage operations

        # Convert to NumPy uint16 explicitly
        img_morph = np.array(img_morph, dtype=np.uint16) * 255

        # Store the processed slice
        segmented_3d[i] = img_morph

    # Save the segmented 3D volume as a TIFF file
    tiff.imwrite(output_path, segmented_3d.astype(np.uint16))

def combine_masks(root_path, test_data):
    print("Step 4: Combining masks...")
    # Combine two binary masks using logical OR
    base_name = os.path.basename(test_data[0]).rsplit(".", 1)[0]
    mask_path = root_path +  "/intensity_result/intensity_" + base_name + ".tif"
    image_path = root_path +  "/inference_result/neuron" + base_name + ".tif"
    output_folder = root_path + "/segementation"
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder + "/seg_" + base_name + ".tif"

    # Load the 3D mask and raw image
    mask = tiff.imread(mask_path)  # Mask in uint16
    raw = tiff.imread(image_path) # Raw image in uint32

    mask_binary = (mask > 0).astype(np.uint16)
    raw_min, raw_max = raw.min(), raw.max()

    raw_scaled = ((raw - raw_min) / (raw_max - raw_min) * 65535).astype(np.uint16)

    masked_raw = raw_scaled * mask_binary

    tiff.imwrite(output_path, masked_raw)

def combine_or_masks(root_path, test_data):
    print("Step 4: Combining masks using OR...")

    # Generate file paths
    base_name = os.path.basename(test_data[0]).rsplit(".", 1)[0]
    mask_path = root_path + "/intensity_result/filled_soma_mask.tif"
    image_path = root_path + "/segementation/seg_z1-22.tif"
    output_folder = root_path + "/segmentation"
    os.makedirs(output_folder, exist_ok=True)
    output_path = output_folder + "/seg_" + base_name + "_or.tif"

    # Load the 3D mask and raw image
    mask = tiff.imread(mask_path)  # Mask in uint16
    raw = tiff.imread(image_path)  # Raw image in uint32

    # Convert both to binary masks
    mask_binary = (mask > 0).astype(np.uint16)
    raw_binary = (raw > 0).astype(np.uint16)

    # Apply logical OR operation
    combined_mask = mask_binary | raw_binary  # <-- 改为 OR 操作

    # Normalize raw image
    raw_min, raw_max = raw.min(), raw.max()
    raw_scaled = ((raw - raw_min) / (raw_max - raw_min) * 65535).astype(np.uint16)

    # Apply the combined mask to the raw image
    # masked_raw = raw_scaled * combined_mask

    # Save the output
    tiff.imwrite(output_path, combined_mask)
    
    print(f"Combined mask saved at: {output_path}")



if __name__ == "__main__":
    # root_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/227_morf/"
    # test_data = ['120C-3_20X-15_2048_V5+homer.tif']

    root_path = "/data/wangfeiran/code/brainbow/segmentation/SegNeuron/data/0207_czi"
    test_data = ['z1-22.czi']

    start_time = time.time()

    # resize_tiff(root_path, test_data, scale_factor=1)

    # czi_morf3_resize(root_path, test_data)

    # inference(root_path, test_data)

    # threshold_intensity(root_path, test_data, otsu_ratio=0.2)

    combine_or_masks(root_path, test_data)

    end_time = time.time()
    print(f"Execution time for combine_masks: {end_time - start_time:.2f} seconds")

    
    