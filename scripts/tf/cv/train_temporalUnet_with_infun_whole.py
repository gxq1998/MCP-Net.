#!/usr/bin/env python

"""
Example script to train MCP-Net.

The data are all arranged in datadir, with npz format.
The dynamic frame is saved with key 'vol', the input function is with key 'this_infun', and the sum of infun is with 'sum_infun'
For a cross-validation fold, the training and testing subject splits are loaded.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import sys
sys.setrecursionlimit(100000)

import argparse
import glob
import numpy as np
import tensorflow as tf
import voxelmorph as vxm


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('datadir', help='base data directory')
parser.add_argument('--cvfold', help='The Xth fold of cv')
parser.add_argument('--temptype', help='temp unit type')
parser.add_argument('--model-dir', default='models', help='model output directory (default: models)')
parser.add_argument('--log-dir', default='logs', help='log output directory (default: logs)')
parser.add_argument('--multichannel', action='store_true', help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100, help='frequency of model saves (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0, help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--update_cycle', default='50', help='update cycle of the remaining ')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+', help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+', help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7, help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2, help='flow downsample factor for integration (default: 2)')
parser.add_argument('--use-probs', action='store_true', help='enable probabilities')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse', help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='lambda_weight', default=0.01, help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--patlak-lambda', type=float, dest='lambda_patlak', default=1, help='weight of gradient or KL loss (default: 0.01)')
parser.add_argument('--legacy-image-sigma', dest='image_sigma', type=float, default=1.0,
                    help='image noise parameter for miccai 2018 network (recommended value is 0.02 when --use-probs is enabled)')
args = parser.parse_args()

# load and prepare training data
cv_index = args.cvfold
lists=np.load('./data/cv/cv_split_{0}.npz'.format(cv_index))
train_list = lists['train']
valid_list = lists['valid']
train_subjects = [x.split('_')[-1] for x in train_list]
valid_subjects = [x.split('_')[-1] for x in valid_list]

print('valid_subjects',valid_subjects)
print('train_subjects',train_subjects)
train_vol_names = [y for x in train_subjects for y in glob.glob(os.path.join(args.datadir, '{0}_*.npz'.format(x)))]
assert len(train_vol_names) > 0, 'Could not find any training data'

# load and prepare validation data
valid_vol_names = [y for x in valid_subjects for y in glob.glob(os.path.join(args.datadir, '{0}_*.npz'.format(x)))]
assert len(valid_vol_names) > 0, 'Could not find any validation data'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# extract shape and number of features from sampled input
sample_shape = next(generator)[0][1].shape


inshape = sample_shape[1:-1]
print('inshape',inshape)
nfeats = sample_shape[-1]

enc_nf = args.enc if args.enc else [16, 32, 32, 32] 
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device(args.gpu)
print('nb_devices',nb_devices)
assert np.mod(args.batch_size, nb_devices) == 0, 'Batch size (%d) should be a multiple of the number of gpus (%d)' % (args.batch_size, nb_devices)

with tf.device(device):
    print('tf.device(device)',tf.device(device))
    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    # prepare model checkpoint save path
    save_filename = os.path.join(model_dir, '{epoch:04d}.tf')
    
    model = vxm.networks.VxmDenseTemporalwithWholePatlak(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=args.bidir,
        use_probs=args.use_probs,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
        src_feats=nfeats,
        trg_feats=nfeats,
        temp_type=args.temptype,
        batch_size=args.batch_size
    )
    
    with open('report.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    # load initial weights (if provided)
    if args.load_weights:
        model.load_weights(args.load_weights)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses_infun.NCC_whole().loss
    elif args.image_loss == 'nmi':
        bin_centers, vol_size = np.linspace(0.02, 0.98, num=64, retstep=True)
        image_loss_func = vxm.losses_infun.NMI(bin_centers, vol_size, crop_background=True).loss
    else:
        raise ValueError('Image loss should be "mse" or "nmi", but found "%s"' % args.image_loss)

    # need two image loss functions if bidirectional
    if args.bidir:
        losses  = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses  = [image_loss_func]
        weights = [1]

    # patlak loss
    patlak_loss_func = vxm.losses_infun.MSE_whole(crop_background=True).loss
    losses += [patlak_loss_func]
    weights += [args.lambda_patlak]

    # prepare deformation loss
    if args.use_probs:
        flow_shape = model.outputs[-1].shape[1:-1]
        losses += [vxm.losses_infun.KL(args.kl_lambda, flow_shape).loss]
    else:
        losses += [vxm.losses_infun.Grad('l2', loss_mult=args.int_downsize).loss]

    weights += [args.lambda_weight]

    print('model.inputs',model.inputs)
    print('model.outputs',model.outputs)
    print(losses,weights)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=weights)

    # save starting weights
    model.save(save_filename.format(epoch=args.initial_epoch))
    save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename)
    log_dir = args.log_dir
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    for ii in range(args.initial_epoch,args.epochs):
        generator = vxm.generators.scan_whole_series_with_infun(train_vol_names, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
        v_generator = vxm.generators.scan_whole_series_with_infun(valid_vol_names, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
        model.fit_generator(generator,
                initial_epoch=ii,
                epochs=ii+1,
                steps_per_epoch=int(len(train_vol_names)/args.batch_size),
                validation_data=v_generator,
                validation_steps=int(len(valid_vol_names)/args.batch_size),
                callbacks=[save_callback,tensorboard_callback],
                verbose=1)

        if ii+1 > args.update_cycle-1 and (ii+1)%args.update_cycle == 0 and ii+1 < args.epochs:
            # load moving and fixed images
            fixedID = 11
            for patient in train_subjects:
                length = int(args.batch_size)
                vol_names = args.datadir
                if os.path.isdir(vol_names):
                    vol_names = os.path.join(vol_names, '{0}_pass_*.npz'.format(patient))

                vol_names = glob.glob(vol_names)
                vol_names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                ref_frame = [x for x in vol_names if x.split('/')[-1].split('_')[-1].split('.')[0]==str(fixedID)][0]
                vol_names.remove(ref_frame)
                whole_names = vol_names
                whole_names.append(ref_frame)

                # load infun
                infun_names = args.datadir
                infun_names = os.path.join(infun_names, '{0}_pass_*.npz'.format(patient))
                infun_names = glob.glob(infun_names)
                infun_names.sort(key = lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]))
                infun_ref_frame = [x for x in infun_names if x.split('/')[-1].split('_')[-1].split('.')[0]==str(fixedID)][0]
                infun_names.remove(infun_ref_frame)

                for i in range(4,len(vol_names),length):
                    whole_dir_ = [x for x in vol_names[4:]]
                    if i>len(vol_names)-length-1:
                        i=len(vol_names)-length-1
                    for j in range(length):
                        moving_dir = vol_names[i+j]
                        infun_dir = infun_names[i+j]
                        whole_dir_.remove(moving_dir)
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

                    for k in range(len(whole_dir_)):
                        whole_dir = whole_dir_[k]
                        infun_dir = infun_names[k]
                        infun_tmp = vxm.py.utils.load_volfile(infun_dir, add_batch_axis=True, add_feat_axis=add_feat_axis,np_var='sum_infun')
                        this_infun_tmp = vxm.py.utils.load_volfile(infun_dir, add_batch_axis=True, add_feat_axis=add_feat_axis,np_var='this_infun')
                        infun=np.concatenate((infun,infun_tmp),axis=0)
                        this_infun=np.concatenate((this_infun,this_infun_tmp),axis=0)

                        whole_tmp = vxm.py.utils.load_volfile(whole_dir, add_batch_axis=True, add_feat_axis=add_feat_axis)
                        if k==0:
                            whole = whole_tmp
                        else:
                            whole = np.concatenate((whole,whole_tmp),axis=0)
                    moving = moving[np.newaxis,:,:,:,:,:]
                    fixed = fixed[np.newaxis,:,:,:,:,:]
                    infun = infun[np.newaxis,:,:,:,:,:]
                    this_infun = this_infun[np.newaxis,:,:,:,:,:]
                    whole = whole[np.newaxis,:,:,:,:,:]

                    inshape = moving.shape[1:-1]
                    nb_feats = moving.shape[-1]

                    # load model and predict
                    warp = model.register(moving, fixed, whole, infun, this_infun)

                    moved = vxm.networks.Transform(inshape[1:], interp_method='linear', nb_feats=nb_feats).predict([moving[0,:,:,:,:,:], warp[0,:,:,:,:,:]])
                    print('moved.shape',moved.shape)

                    warp = warp[0,:,:,:,:,:]
                    for jj in range(length):
                        moving_dir = vol_names[i+jj]
                        vxm.py.utils.save_volfile_for_whole(moved[jj].squeeze(),infun[0,jj,:,:,:,:].squeeze(), this_infun[0,jj,:,:,:,:].squeeze(), moving_dir, moving[0,jj,:,:,:,:].squeeze(), fixed_affine)

