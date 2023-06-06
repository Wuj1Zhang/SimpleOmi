## Introduction


### Create environment
-   For pip users
```bash
conda env create -n sim-omi
conda activate sim-omi
pip install -r requirements.txt
```

### Step to run
-   Train and test with the default settings
```bash
python train_test.py
```
-   Check the output files flowing the path in exp parameters
```bash
cd EXP FOLDER PATH/ 
```
-   Visualise the metrics and losses
```bash
tensorboard --logdir=/gscratch/stf/hzhang33/omiExp/ --port=7447
```


## Acknowledgments
Code for a few functions and networks was taken from the repository [OmiEmbed](https://github.com/zhangxiaoyu11/OmiEmbed) and modified as needed.
