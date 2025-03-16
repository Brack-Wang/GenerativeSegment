# ZECO for the Brainbow Neurons Generation


## Preprocess the data
```bash
# unzip skeleton
cd /brainbow/data_preprocess
python unzip.py
# generate and crop neurons
python data/preprocess/split.py
python data/preprocess/crop.py
```


## Train the model
```bash
conda activate monaimonai

python main.py --gpu 2

```



