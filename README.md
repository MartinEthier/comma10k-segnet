# comma10k-segnet
Training a semantic segmentation network on the comma10k dataset. This network is to be used to filter out dynamic objects when using visual odometry for the comma calibration challenge. The comma calib code can be found at my [calib_challenge repo](https://github.com/MartinEthier/calib_challenge).

## Dataset Setup
The dataset can be downloaded by cloning the comma10k repo. However, simply using git clone will result in an insane download time since we are downloaded each past commit. So limit the depth of the clone to 1 to speed it up:
```
git clone --depth 1 git@github.com:commaai/comma10k.git
```
The only things needed are the imgs and masks folders and the files_trainable file. Feel free to delete everything else.

## Training Setup
The config for my best run is deeplabv3plus.yaml. So to run this, just do:
```
python train.py configs/deeplabv3plus.yaml
```
