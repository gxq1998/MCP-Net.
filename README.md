# MCP-Net: Inter-frame Motion Correction with Patlak Regularization for Whole-body Dynamic PET

Provisional acceptance (top ~13%) in MICCAI 2022, Singapore.

We propose an inter-frame Motion Correction framework with Patlak regularization (MCP-Net) to directly optimize the Patlak fitting error and further improve model performance. 

The MCP-Net contains three modules: a motion estimation module consisting of a multiple-frame 3-D U-Net with a convolutional long short-term memory layer combined at the bottleneck; an image warping module that performs spatial transformation; and an analytical Patlak module that estimates Patlak fitting with the motion-corrected frames and the individual input function. A Patlak loss penalization term using mean squared percentage fitting error is introduced to the loss function in addition to image similarity measurement and displacement gradient loss.


keywords: Inter-frame motion correction, Parametric imaging, Tracer kinetics regularization, Whole-body dynamic PET


# Instructions

## Training

If you would like to train your own model, you will likely need to customize some of the data loading code in `voxelmorph/generators.py` for your own datasets and data formats. However, it is possible to run many of the example scripts out-of-the-box, assuming that you have a directory containing training data files in npz (numpy) format. It's assumed that each npz file in your data folder has a `vol` parameter, which points to the numpy image data to be registered, and an optional `seg` variable, which points to a corresponding discrete segmentation. It's also assumed that the shape of all image data in a directory is consistent.

For a given `/path/to/training/data`, the following script will train the MCP-Net. Model weights will be saved to a path specified by the `--model-dir` flag.

```
./scripts/tf/cv/train_temporalUnet_with_infun_whole.py /path/to/training/data --cvfold 1 --model-dir /path/to/models/output --gpu 0 --image-loss 'ncc' --lambda 0.1 --patlak-lambda 0.1 --log-dir /path/to/log/output --temptype 'convLSTM' --epochs 500 --batch-size 5 

```

## Registration

To correct motion of all the frames, we could run:

```
./scripts/tf/register_temporal_infun_whole.py --batchsize 5 --moving /path/to/moving/frames --infun /path/to/infun --fixedID 11 --moved /path/to/moved/frames --model /path/to/models/output --warp /path/to/warps --gpu 0
```

## Parameter choices

For our data, we found `lambda=0.1` and `patlak-lambda=0.1` with NCC loss to work best.


# Papers

If you use MCP-Net or some part of the code, please cite (see [bibtex](citations.bib)):

Xueqi Guo, Bo Zhou, Xiongchao Chen, Chi Liu, and Nicha Dvornek. MCP-Net: Inter-frame Motion Correction with Patlak Regularization for Whole-body Dynamic PET. In the 25th International Conference on Medical Image Computing and Computer Assisted Intervention, Singapore, Sept 18-22, 2022. 

# Acknowledgments

The code was heavily borrowed from Voxelmorph (https://github.com/voxelmorph/voxelmorph). 

# Notes on Data

We originally used a private dataset at Yale PET Center. If you would like to access the data, please contact chi.liu@yale.edu

We encourage users to download and process their own data. 



# Contact:
For any problems or questions please open an issue or contact xueqi.guo@yale.edu.  
