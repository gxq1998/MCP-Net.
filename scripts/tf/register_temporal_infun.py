#!/usr/bin/env python

"""
Example script to register two volumes with VoxelMorph models.

Please make sure to use trained models appropriately. Let's say we have a model trained to register a
scan (moving) to an atlas (fixed). To register a scan to the atlas and save the warp field, run:

    register.py --moving moving.nii.gz --fixed fixed.nii.gz --model model.h5 --moved moved.nii.gz --warp warp.nii.gz

The source and target input images are expected to be affinely registered.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import numpy as np
import voxelmorph as vxm
import tensorflow as tf
import glob
import time
start_time = time.time()


# parse commandline args
parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--infun', required=True, help='moving image (source) filename')
parser.add_argument('--batchsize', required=True, help='moving image (source) filename')
parser.add_argument('--fixedID', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', required=True, help='warped image output filename')
parser.add_argument('--model', required=True, help='keras model for nonlinear registration')
parser.add_argument('--warp', help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')
args = parser.parse_args()

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)

# load moving and fixed images
add_feat_axis = not args.multichannel
length = int(args.batchsize)
vol_names = args.moving
if os.path.isdir(vol_names):
    vol_names = os.path.join(vol_names, '*_padded.nii')
vol_names = glob.glob(vol_names)
vol_names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-2])) 
ref_frame = [x for x in vol_names if x.split('/')[-1].split('_')[-2]==str(args.fixedID)][0]
vol_names.remove(ref_frame)

# load infun
infun_names = args.infun
infun_names = infun_names + '_pass_*.npz'
infun_names = glob.glob(infun_names)
infun_names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0])) 
print(infun_names)
infun_ref_frame = [x for x in infun_names if x.split('/')[-1].split('_')[-1].split('.')[0]==str(args.fixedID)][0]
infun_names.remove(infun_ref_frame)

for i in range(4,len(vol_names),length):
    if i>len(vol_names)-length:
        i=len(vol_names)-length
    for j in range(length):
        moving_dir = vol_names[i+j]
        infun_dir = infun_names[i+j]
        print(i+j,moving_dir,infun_dir)
        moving_tmp = vxm.py.utils.load_volfile(moving_dir, add_batch_axis=True, add_feat_axis=add_feat_axis)
        infun_tmp = vxm.py.utils.load_volfile(infun_dir, add_batch_axis=True, add_feat_axis=add_feat_axis,np_var='sum_infun')
        this_infun_tmp = vxm.py.utils.load_volfile(infun_dir, add_batch_axis=True, add_feat_axis=add_feat_axis,np_var='this_infun')
        fixed_tmp, fixed_affine = vxm.py.utils.load_volfile(ref_frame, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)
        if j==0:
            moving=moving_tmp
            fixed=fixed_tmp
            infun=infun_tmp
            this_infun=this_infun_tmp
        else:
            moving=np.concatenate((moving,moving_tmp),axis=0)
            fixed=np.concatenate((fixed,fixed_tmp),axis=0)
            infun=np.concatenate((infun,infun_tmp),axis=0)
            this_infun=np.concatenate((this_infun,this_infun_tmp),axis=0)
    print('sum moving',np.sum(moving))
    print('sum fixed',np.sum(fixed))
    print('sum infun',np.sum(infun))
    print('sum this_infun',np.sum(this_infun))
    
    inshape = moving.shape[1:-1]
    nb_feats = moving.shape[-1]
    print('inshape',inshape, 'nb_feats', nb_feats)
    with tf.device(device):
        # load model and predict
        warp = vxm.networks.VxmDenseTemporalwithPatlak.load(args.model).register(moving, fixed, infun, this_infun)
        
        #warp = vxm.networks.VxmDenseTemporalwithWholePatlak.load(args.model).register(moving, fixed, infun, this_infun)
        print('warp.shape',warp.shape)
        moved = vxm.networks.Transform(inshape, interp_method='linear', nb_feats=nb_feats).predict([moving, warp])
        print('moved.shape',moved.shape)
        #patlak = vxm.networks.VxmDenseTemporalwithPatlak.load(args.model).get_patlak(moving, fixed, infun, this_infun)
        #print('patlak.shape',patlak.shape)
        #x_t_ = vxm.networks.VxmDenseTemporalwithPatlak.load(args.model).test_x_t(moving, fixed, infun, this_infun)
        #print('test x_t_',np.sum(x_t_))
        #print('patlak slope',vxm.networks.VxmDenseTemporalwithPatlak.load(args.model).layers[-3].get_weights()[0])
        #print('patlak intercept',vxm.networks.VxmDenseTemporalwithPatlak.load(args.model).layers[-3].get_weights()[1])
    
    for j in range(length):
        moving_dir = vol_names[i+j]
        file_name = moving_dir.split('/')[-1].split('.')[0]
        # save warp
        if args.warp:
            warp_file = os.path.join(args.warp, (file_name+'.npz'))
            vxm.py.utils.save_volfile(warp[j].squeeze(), warp_file, fixed_affine)
        moved_file = os.path.join(args.moved, (file_name+'.nii'))
        vxm.py.utils.save_volfile(moved[j].squeeze(), moved_file, fixed_affine)
        #ki_file = os.path.join(args.moved, (file_name+'_ki.nii'))
        #print('ki',j, 'sum',np.sum(patlak[j].squeeze()[0,:,:,:]))

        #vxm.py.utils.save_volfile(patlak[j].squeeze()[0,:,:,:], ki_file, fixed_affine)
        #vb_file = os.path.join(args.moved, (file_name+'_vb.nii'))
        #print('vb',j, 'sum',np.sum(patlak[j].squeeze()[1,:,:,:]))
        #vxm.py.utils.save_volfile(patlak[j].squeeze()[1,:,:,:], vb_file, fixed_affine)

print("--- Elapsed time %s seconds ---" % (time.time() - start_time))