# comma10k-segnet
Training a semantic segmentation network on the comma10k dataset. This network is to be used to filter out dynamic objects when using visual odometry for the comma calibration challenge.

## Dataset Setup
The dataset can be downloaded by cloning the comma10k repo. However, simply using git clone will result in an insane download time since we are downloaded each past commit. So limit the depth of the clone to 1 to speed it up:
```
git clone --depth 1 git@github.com:commaai/comma10k.git
```
The only things needed are the imgs and masks folders and the files_trainable file. Feel free to delete everything else.

## Training Setup


## Mask Alpha Channel Issue?
I ended up wasting a ton of time because some of the masks in the dataset have alpha channels, and some don't... I was loading in the masks as numpy arrays and matching the colors to the provided class color list, but not removing the alpha channel because I didn't know about it (most PNGs don't have an alpha channels). This ended up causing the final masks to just be "road" for all the PNGs that have alpha channels and was causing me to have abnormally high cross-entropy loss values.
