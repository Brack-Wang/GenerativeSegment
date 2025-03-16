import numpy as np
import tifffile

def filter_tif(input_file, output_file, center_value=127.5, keep_ratio=0.9):
    # Read the TIFF file
    data = tifffile.imread(input_file)

    # Ensure the file has the required dimensions Z,1,Y,X
    if len(data.shape) != 4 or data.shape[1] != 1:
        raise ValueError("Input file must have dimensions Z,1,Y,X")

    # Calculate the range to keep
    full_range = 255  # Assuming pixel values are 0-255
    range_to_keep = full_range * keep_ratio
    lower_bound = center_value - range_to_keep / 2
    upper_bound = center_value + range_to_keep / 2

    # Apply the filter
    filtered_data = np.where(
        (data >= lower_bound) & (data <= upper_bound), data, 0
    )

    # Save the filtered data to a new file
    tifffile.imwrite(output_file, filtered_data.astype(data.dtype))
    print(f"Filtered file saved to {output_file}")

# Example usage
input_tif_path = "/data/wangfeiran/code/brainbow/output/1209_ddpm_80_seg/ddpm/reconstructions/synth_epoch_2000_sample_1.tif"
output_tif_path = "/data/wangfeiran/code/brainbow/output/1209_ddpm_80_seg/ddpm/reconstructions/synth_epoch_2000_sample_1222.tif"

# Uncomment the following line to run the processing function
filter_tif(input_tif_path, output_tif_path)
