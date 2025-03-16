# SegNeuron

## Training
### 1. Pretraining
```bash
cd Pretrain
```
```bash
# change the path in .config
python pretrain.py
```
### 2. Supervised Training
```bash
cd Train_and_Inference
```
```bash
# change the path in .config
python supervised_train.py
```
## Inference
### 1. Affinity Inference
```
cd Train_and_Inference
```
```
python inference.py
```
### 2. Instance Segmentation
```
cd Postprocess
```
```
python FRMC_post.py
```


# Create synthesis dataset based on single neurons
script/synthesis_dataset.py


# Create segmentation training data from synthesis data
script/createSgData.py


# Create segmentation mask

script/ndl_morf.py
script/morf_2.py



# Others
czi to tif format convert
script/czi2tif.py



## Acknowledgement
This code is based on [SSNS-Net](https://github.com/weih527/SSNS-Net) (IEEE TMI'22) by Huang Wei et al. The postprocessing tools are based on [constantinpape/elf](https://github.com/constantinpape/elf). Should you have any further questions, please let us know. Thanks again for your interest.


