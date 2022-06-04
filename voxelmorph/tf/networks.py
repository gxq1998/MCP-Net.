"""
tensorflow/keras networks for voxelmorph

If you use this code, please cite:

 and one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

License: GPLv3
"""

# internal python imports
from collections.abc import Iterable

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

# local imports
import neurite as ne
from .. import default_unet_features
from . import layers

from .modelio import LoadableModel, store_config_args


# make ModelCheckpointParallel directly available from vxm
ModelCheckpointParallel = ne.callbacks.ModelCheckpointParallel

class VxmDenseTemporalwithWholePatlak(LoadableModel):
    """
    MCP-Net with whole-sequence Patlak estimation
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=7,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            trg_feats=1,
            unet_half_res=False,
            input_model=None,
            name='vm',
            temp_type='LSTM',
            dynamic_length = 15,
            batch_size=1,
            hyp_model=None):

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3,4], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), batch_size=1,name='%s_source_input' % name)
            print('source shape',source.shape)
            rest_source = tf.keras.Input(shape=(10,inshape[1], inshape[2], inshape[3], src_feats), batch_size=1,name='%s_rest_source_input' % name)
            print('rest source shape',rest_source.shape)

            target_frame = tf.keras.Input(shape=(*inshape, src_feats), batch_size=1,name='%s_target_input_frame' % name)
            infun = tf.keras.Input(shape=(dynamic_length,inshape[1], inshape[2], inshape[3], src_feats), batch_size=1,name='%s_infun' % name)
            this_infun = tf.keras.Input(shape=(dynamic_length,inshape[1], inshape[2], inshape[3],  src_feats), batch_size=1,name='%s_this_infun' % name)

            print('infun shape',infun.shape)
            input_model = tf.keras.Model(inputs=[source, target_frame], outputs=[source[0,:,:,:,:,:], target_frame[0,:,:,:,:,:]])
        else:
            source, target = input_model.outputs[:2]
            
        # configure inputs
        inputs = input_model.inputs
        if hyp_model is not None:
            hyp_input = hyp_model.input
            hyp_tensor = hyp_model.output
            inputs = (*inputs, hyp_input)
        else:
            hyp_input = None
            hyp_tensor = None

        # build core unet model and grab inputs
        if temp_type == 'serialconvLSTM':
            unet_model = temporalConvNet(
                input_model=input_model,
                nb_features=nb_unet_features,
                nb_levels=nb_unet_levels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=nb_unet_conv_per_level,
                half_res=unet_half_res,
                name='%s_unet' % name,
                hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
                temp_type=temp_type
            )
        else:
            unet_model = temporalNet(
                input_model=input_model,
                nb_features=nb_unet_features,
                nb_levels=nb_unet_levels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=nb_unet_conv_per_level,
                half_res=unet_half_res,
                name='%s_unet' % name,
                hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
                temp_type=temp_type,
                batch_size=1
            )
        

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % (ndims-1))
        flow_mean = Conv(ndims-1, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)

        print('flow_mean.shape',flow_mean.get_shape())
        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow

        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss', name='%s_neg_flow_int' % name, int_steps=int_steps)(neg_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)
                if bidir:
                    neg_flow = layers.RescaleTransform(int_downsize, name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        print('pos_flow after int_steps',pos_flow.get_shape())
        y_source = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_transformer' % name)([source[0,:,:,:,:,:], pos_flow])
        if bidir:
            y_target = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_neg_transformer' % name)([target, neg_flow])
            
        print('warped shape',y_source.get_shape())

        y_source_whole = tf.concat([y_source,rest_source[0,:,:,:,:,:]], axis = 0)
        print('y_source_whole.shape',y_source_whole.get_shape())
        weights = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(y_source_whole,-1),-1),-1),-1)
        print('weights.shape',weights.get_shape())
        weights_norm_diag = tf.linalg.diag(tf.div(weights,tf.reduce_max(weights)))
        print('weights_norm_diag.shape',weights_norm_diag.get_shape())
        
        # linear fitting
        X = tf.stack([infun[0,:,0,0,0,0], this_infun[0,:,0,0,0,0]], axis=1)
        print('stack shape',X.get_shape())
        X_T = tf.transpose(X)
        print('transpose shape',X_T.get_shape())
        #X_T_ = K.dot(tf.linalg.inv(K.dot(X_T,X)+ tf.eye(2) * 10e-4),X_T)
        X_T_ = K.dot(tf.linalg.inv(K.dot(K.dot(X_T,weights_norm_diag),X)),K.dot(X_T,weights_norm_diag))
        print('inv shape',X_T_.get_shape())
        
        y_source_shape = y_source_whole.get_shape()[1:]
            
        patlak = K.dot(X_T_,KL.Flatten()(y_source_whole))
        patlak = KL.Reshape((y_source_shape[0],y_source_shape[1],y_source_shape[2],y_source_shape[3]))(patlak)
        print('patlak shape',patlak.get_shape())
        
        ki = patlak[0,:,:,:,0]
        vb = patlak[1,:,:,:,0]
        
        estimation = (tf.expand_dims(ki, axis=-1)*infun + tf.expand_dims(vb, axis=-1)*this_infun - y_source_whole) / (y_source_whole+1)

        print('estimation shape',estimation.get_shape())
        
        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source[tf.newaxis,:,:,:,:,:], estimation]

        if use_probs:
            # compute loss on flow probabilities
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            outputs += [preint_flow]

        super().__init__(name=name, inputs=[source,target_frame,rest_source,infun,this_infun], outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.patlak = patlak
        self.references.infun = infun
        self.references.this_infun = this_infun
        self.references.target_frame = target_frame
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = pos_flow[tf.newaxis,:,:,:,:,:]
        self.references.neg_flow = neg_flow if bidir else None
        self.references.hyp_input = hyp_input

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def get_patlak_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, tf.stack([self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0]], axis=0))
    
    def get_patlak(self, src, trg, infun, this_infun):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_patlak_model().predict([src, trg, infun, this_infun])
    
    def register(self, src, trg, rest_source, infun, this_infun):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg, rest_source, infun, this_infun])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])
    
    
class VxmDenseTemporalwithPatlak(LoadableModel):
    """
    MCP-Net with partial Patlak estimation
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=7,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            unet_half_res=False,
            input_model=None,
            name='vm',
            temp_type='LSTM',
            batch_size=5,
            hyp_model=None):

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3,], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), batch_size=batch_size,name='%s_source_input' % name)
            target_frame = tf.keras.Input(shape=(*inshape, src_feats), batch_size=batch_size,name='%s_target_input_frame' % name)
            infun = tf.keras.Input(shape=(*inshape, src_feats), batch_size=batch_size,name='%s_infun' % name)
            this_infun = tf.keras.Input(shape=(*inshape, src_feats), batch_size=batch_size,name='%s_this_infun' % name)

            print('infun shape',infun.shape)
            input_model = tf.keras.Model(inputs=[source, target_frame, infun, this_infun], outputs=[source, target_frame, infun, this_infun])
        else:
            source, target = input_model.outputs[:2]
            
        # configure inputs
        inputs = input_model.inputs
        if hyp_model is not None:
            hyp_input = hyp_model.input
            hyp_tensor = hyp_model.output
            inputs = (*inputs, hyp_input)
        else:
            hyp_input = None
            hyp_tensor = None

        # build core unet model and grab inputs
        if temp_type == 'serialconvLSTM':
            unet_model = temporalConvNet(
                input_model=input_model,
                nb_features=nb_unet_features,
                nb_levels=nb_unet_levels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=nb_unet_conv_per_level,
                half_res=unet_half_res,
                name='%s_unet' % name,
                hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
                temp_type=temp_type
            )
        else:
            unet_model = temporalNet(
                input_model=input_model,
                nb_features=nb_unet_features,
                nb_levels=nb_unet_levels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=nb_unet_conv_per_level,
                half_res=unet_half_res,
                name='%s_unet' % name,
                hyp_input=hyp_input,
            hyp_tensor=hyp_tensor,
                temp_type=temp_type,
                batch_size=batch_size
            )
        

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow

        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss', name='%s_neg_flow_int' % name, int_steps=int_steps)(neg_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)
                if bidir:
                    neg_flow = layers.RescaleTransform(int_downsize, name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        print('pos_flow after int_steps',pos_flow.get_shape())
        y_source = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_transformer' % name)([source, pos_flow])
        if bidir:
            y_target = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_neg_transformer' % name)([target, neg_flow])

        # linear fitting
        X = tf.stack([infun[:,0,0,0,0], this_infun[:,0,0,0,0]], axis=1)
        X_T = tf.transpose(X)
        X_T_ = K.dot(tf.linalg.inv(K.dot(X_T,X)),X_T)

        y_source_shape = y_source.get_shape()[1:]
            
        patlak = K.dot(X_T_,KL.Flatten()(y_source))
        patlak = KL.Reshape((y_source_shape[0],y_source_shape[1],y_source_shape[2],y_source_shape[3]))(patlak)

        ki = patlak[0,:,:,:,0]
        vb = patlak[1,:,:,:,0]
        
        estimation = (tf.expand_dims(ki, axis=-1)*infun + tf.expand_dims(vb, axis=-1)*this_infun - y_source) / (y_source+1)

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source, estimation]

        if use_probs:
            # compute loss on flow probabilities
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            outputs += [preint_flow]

        super().__init__(name=name, inputs=[source,target_frame,infun,this_infun], outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.patlak = patlak
        self.references.infun = infun
        self.references.this_infun = this_infun
        self.references.target_frame = target_frame
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None
        self.references.hyp_input = hyp_input

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def get_patlak_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, tf.stack([self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0],  self.references.patlak[:,:,:,:,0]], axis=0))
    
    def get_patlak(self, src, trg, infun, this_infun):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_patlak_model().predict([src, trg, infun, this_infun])
    
    def register(self, src, trg, infun, this_infun):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg, infun, this_infun])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmDenseTemporal(LoadableModel):
    """
    Spatiotemporal model
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=7,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            trg_feats=1,
            unet_half_res=False,
            input_model=None,
            name='vxm_dense',
            batch_size=5,
            temp_type='LSTM'):

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), batch_size=batch_size, name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), batch_size=batch_size, name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = temporalNet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            name='%s_unet' % name,
            temp_type=temp_type
        )
        

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow

        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss', name='%s_neg_flow_int' % name, int_steps=int_steps)(neg_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)
                if bidir:
                    neg_flow = layers.RescaleTransform(int_downsize, name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        print('pos_flow after int_steps',pos_flow.get_shape())
        y_source = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_transformer' % name)([source, pos_flow])
        if bidir:
            y_target = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_neg_transformer' % name)([target, neg_flow])

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source]

        if use_probs:
            # compute loss on flow probabilities
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            outputs += [preint_flow]

        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])
    
    
class VxmDenseConvTemporal(LoadableModel):
    """
    Spatiotemporal model with S-convLSTM as the estimation network
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=7,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            trg_feats=1,
            unet_half_res=False,
            input_model=None,
            name='vxm_dense',
            temp_type='LSTM'):

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = temporalConvNet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            name='%s_unet' % name,
            temp_type=temp_type
        )
        

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow

        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss', name='%s_neg_flow_int' % name, int_steps=int_steps)(neg_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)
                if bidir:
                    neg_flow = layers.RescaleTransform(int_downsize, name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        print('pos_flow after int_steps',pos_flow.get_shape())
        y_source = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_transformer' % name)([source, pos_flow])
        if bidir:
            y_target = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_neg_transformer' % name)([target, neg_flow])

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source]

        if use_probs:
            # compute loss on flow probabilities
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            outputs += [preint_flow]

        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])
    
class VxmDense(LoadableModel):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    @store_config_args
    def __init__(self,
            inshape,
            nb_unet_features=None,
            nb_unet_levels=None,
            unet_feat_mult=1,
            nb_unet_conv_per_level=1,
            int_steps=7,
            int_downsize=2,
            bidir=False,
            use_probs=False,
            src_feats=1,
            trg_feats=1,
            unet_half_res=False,
            input_model=None,
            name='vxm_dense'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. Default is False.
            input_model: Model to replace default input layer before concatenation. Default is None.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            name='%s_unet' % name
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)

        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        # optionally negate flow for bidirectional model
        pos_flow = flow
        if bidir:
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)
            if bidir:
                neg_flow = layers.VecInt(method='ss', name='%s_neg_flow_int' % name, int_steps=int_steps)(neg_flow)

            # resize to final resolution
            if int_downsize > 1:
                pos_flow = layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)
                if bidir:
                    neg_flow = layers.RescaleTransform(int_downsize, name='%s_neg_diffflow' % name)(neg_flow)

        # warp image with flow field
        y_source = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_transformer' % name)([source, pos_flow])
        if bidir:
            y_target = layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_neg_transformer' % name)([target, neg_flow])

        # initialize the keras model
        outputs = [y_source, y_target] if bidir else [y_source]

        if use_probs:
            # compute loss on flow probabilities
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            outputs += [preint_flow]

        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


###############################################################################
# Utility/Core Networks
###############################################################################

class Transform(tf.keras.Model):
    """
    Simple transform model to apply dense or affine transforms.
    """

    def __init__(self,
        inshape,
        affine=False,
        interp_method='linear',
        rescale=None,
        fill_value=None,
        nb_feats=1):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            affine: Enable affine transform. Default is False.
            interp_method: Interpolation method. Can be 'linear' or 'nearest'. Default is 'linear'.
            rescale: Transform rescale factor. Default is None.
            fill_value: Fill value for SpatialTransformer. Default is None.
            nb_feats: Number of source image features. Default is 1.
        """

        # configure inputs
        ndims = len(inshape)
        scan_input = tf.keras.Input((*inshape, nb_feats), name='scan_input')

        if affine:
            trf_input = tf.keras.Input((ndims * (ndims + 1),), name='trf_input')
        else:
            trf_shape = inshape if rescale is None else [int(d / rescale) for d in inshape]
            trf_input = tf.keras.Input((*trf_shape, ndims), name='trf_input')

        trf_scaled = trf_input if rescale is None else layers.RescaleTransform(rescale)(trf_input)

        # transform and initialize the keras model
        trf_layer = layers.SpatialTransformer(interp_method=interp_method,
                                              name='transformer',
                                              fill_value=fill_value)
        y_source = trf_layer([scan_input, trf_scaled])
        super().__init__(inputs=[scan_input, trf_input], outputs=y_source)


class Unet(tf.keras.Model):
    """
    A unet architecture that builds off either an input keras model or input shape. Layer features can be
    specified directly as a list of encoder and decoder features or as a single integer along with a number
    of unet levels. The default network features per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self,
                 inshape=None,
                 input_model=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 do_res=False,
                 half_res=False,
                 name='unet'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            unet_input = KL.Input(shape=inshape, name='%s_input' % name)
            model_inputs = [unet_input]
        else:
            unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool]*nb_levels

        # configure encoder (down-sampling path)
        enc_layers = []
        last = unet_input
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res)
            enc_layers.append(last)
            
            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

        # configure decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res)
            if not half_res or level < (nb_levels - 2):
                layer_name = '%s_dec_upsample_%d' % (name, real_level)
                last = _upsample_block(last, enc_layers.pop(), factor=max_pool[real_level], name=layer_name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            layer_name = '%s_dec_final_conv_%d' % (name, num)
            last = _conv_block(last, nf, name=layer_name)

        super().__init__(inputs=model_inputs, outputs=last, name=name)
        
        
class temporalNet(tf.keras.Model):
    """
    A multiple-frame U-Net with a temporal layer integrated at the bottleneck
    """

    def __init__(self,
                 inshape=None,
                 input_model=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 do_res=False,
                 half_res=False,
                 temp_type='LSTM',
                 hyp_input=None,
                 hyp_tensor=None,
                 batch_size=1,
                 name='unet'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            unet_input = KL.Input(shape=inshape, batch_size=batch_size, name='%s_input' % name)
            model_inputs = [unet_input]
        else:
            print('input_model.outputs',input_model.outputs)
            if len(input_model.outputs[0].get_shape())>5:
                unet_input = KL.concatenate([input_model.outputs[0][:,0,:,:,:,:],input_model.outputs[1]], name='%s_input_concat' % name)
            else:
                unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs

        # add hyp_input tensor if provided
        if hyp_input is not None:
            model_inputs = model_inputs + [hyp_input]
            
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool]*nb_levels

        # configure encoder (down-sampling path)
        enc_layers = []
        if len(unet_input.get_shape())>5:
            last = unet_input[0]
        else:
            last = unet_input#[0] #[:,:,:,:,:2]
            #infuns = unet_input[:,:,:,:,2:]
        print('last',last.get_shape())
        #print('infuns',infuns.get_shape())
        last = KL.Lambda(lambda x :K.clip(x, 0, 2.5))(last)
        mask = last >= 2.5
        mask = tf.cast(mask, tf.float32)
        noise = tf.random.normal(mask.get_shape(), mean=0.0, stddev=0.01, dtype=tf.dtypes.float32, seed=None, name=None)
        print('mask shape',mask.get_shape())
        print('noise shape',noise.get_shape())

        last = last + noise * mask
        #last = KL.concatenate([last,infuns])
        #print('last',last.get_shape())
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
                print(layer_name)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res)
                print(last.get_shape())
            enc_layers.append(last)
            
            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

            print('level',level)
            print(last.get_shape())
            
        bottle_shape = last.get_shape()[1:]
        print('bottle_shape',bottle_shape)
        if temp_type=='convLSTM':
            last = tf.expand_dims(last, axis=0)
            print('convLSTM bottleneck input shape',last.get_shape())
            last = tf.squeeze(CR.ConvLSTM3D(32,3,return_sequences=True)(last))
            print('convLSTM bottleneck shape',last.get_shape())
            last = KL.Reshape((bottle_shape[0],bottle_shape[1],bottle_shape[2],32))(last)
            print('reshape to 3d',last.get_shape())
        elif temp_type=='multiframe':
            print('bottleneck shape',last.get_shape())
            pass
        else:
            flattened = KL.Flatten()(last)
            print('flattened shape',flattened.get_shape().as_list())
            flattened = tf.expand_dims(flattened, axis=0)
            print('expand dims flattened shape',flattened.get_shape())

            nf_lstm = nf
            if temp_type=='LSTM':
                lstm = KL.LSTM(int(flattened.get_shape().as_list()[-1]/nf_lstm),return_sequences=True)
            elif temp_type=='RNN':
                lstm = KL.SimpleRNN(int(flattened.get_shape().as_list()[-1]/nf_lstm),return_sequences=True)
            elif temp_type=='GRU':
                lstm = KL.GRU(int(flattened.get_shape().as_list()[-1]/nf_lstm),return_sequences=True)
            last = tf.squeeze(lstm(flattened))
            print(temp_type, 'output shape', last.get_shape())

            last = KL.Reshape((bottle_shape[0],bottle_shape[1],bottle_shape[2],1))(last)
            print('reshape to 3d',last.get_shape())
            last = _conv_block(last, nf, name='conv_after_lstm', do_res=do_res)
            print('conv after lstm output shape', last.get_shape())
            last = KL.Reshape(bottle_shape)(last)
            print('decoder input shape', last.get_shape())
        
        # configure decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
                print(layer_name)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res)
                print(last.get_shape())
            if not half_res or level < (nb_levels - 2):
                layer_name = '%s_dec_upsample_%d' % (name, real_level)
                print(layer_name)
                last = _upsample_block(last, enc_layers.pop(), factor=max_pool[real_level], name=layer_name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            layer_name = '%s_dec_final_conv_%d' % (name, num)
            print(layer_name)
            last = _conv_block(last, nf, name=layer_name)
            print(last.get_shape())

        super().__init__(inputs=model_inputs, outputs=last, name=name)

        
class temporalConvNet(tf.keras.Model):
    """
    A multiple-frame U-Net with a temporal layer integrated following the U-net
    """


    def __init__(self,
                 inshape=None,
                 input_model=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 do_res=False,
                 half_res=False,
                 temp_type='LSTM',
                 hyp_input=None,
                 hyp_tensor=None,
                 name='unet'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            unet_input = KL.Input(shape=inshape, name='%s_input' % name)
            model_inputs = [unet_input]
        else:
            unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs

        # add hyp_input tensor if provided
        if hyp_input is not None:
            model_inputs = model_inputs + [hyp_input]
        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()
            

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool]*nb_levels

        # configure encoder (down-sampling path)
        enc_layers = []
        last = unet_input
        print('unet_input',unet_input.get_shape())
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
                print(layer_name)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res)
                print(last.get_shape())
            enc_layers.append(last)
            
            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

            print('level',level)
            print(last.get_shape())
            
        # configure decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
                print(layer_name)
                last = _conv_block(last, nf, name=layer_name, do_res=do_res)
                print(last.get_shape())
            if not half_res or level < (nb_levels - 2):
                layer_name = '%s_dec_upsample_%d' % (name, real_level)
                print(layer_name)
                last = _upsample_block(last, enc_layers.pop(), factor=max_pool[real_level], name=layer_name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            layer_name = '%s_dec_final_conv_%d' % (name, num)
            print(layer_name)
            last = _conv_block(last, nf, name=layer_name)
            print(last.get_shape())
            
        # feed into a convlstm 
        last = _conv_block(last, 16, name='conv_be4_convlstm', do_res=do_res)
        last = MaxPooling(max_pool[level], name='conv_be4_convlstm_pooling')(last)
        last = tf.expand_dims(last, axis=0)
        #print(last.get_shape())
        last_shape = last.get_shape()[2:]
        last = CR.ConvLSTM3D(32,3,return_sequences=True)(last)
        print('CR.ConvLSTM3D',last.get_shape())
        last = tf.squeeze(last)
        last = KL.Reshape((last_shape[0],last_shape[1],last_shape[2],32))(last)
        print('clstm_layer_1',last.get_shape())
        last = _upsample_only_block(last,name='upsample_back')
        print('final last',last.get_shape())

        super().__init__(inputs=model_inputs, outputs=last, name=name)


###############################################################################
# Private functions
###############################################################################

def _conv_block(x, nfeat, strides=1, name=None, do_res=False,hyp_tensor=None):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    extra_conv_params = {}
    if hyp_tensor is not None:
        Conv = getattr(ne.layers, 'HyperConv%dDFromDense' % ndims)
        conv_inputs = [x, hyp_tensor]
    else:
        Conv = getattr(KL, 'Conv%dD' % ndims)
        extra_conv_params['kernel_initializer'] = 'he_normal'
        conv_inputs = x

    convolved = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(x)
    name = name + '_activation' if name else None

    if do_res:
        add_layer = x
        print('note: this is a weird thing to do, since its not really residual training anymore')
        if nfeat != x.get_shape().as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', name='resfix_'+name)(x)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    return KL.LeakyReLU(0.2, name=name)(convolved)


def _upsample_only_block(x, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    #print('x',x.get_shape())
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)
    
    upsampled = UpSampling(size=(factor,) * ndims, name=name)(x)
    #print('upsampled',upsampled.get_shape())
    name = name if name else None
    #print('concatenated',KL.concatenate([upsampled, connection]).get_shape())
    return upsampled


def _upsample_block(x, connection, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    #print('x',x.get_shape())
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)
    
    upsampled = UpSampling(size=(factor,) * ndims, name=name)(x)
    #print('upsampled',upsampled.get_shape())
    name = name + '_concat' if name else None
    #print('concatenated',KL.concatenate([upsampled, connection]).get_shape())
    return KL.concatenate([upsampled, connection], name=name)

def _concat_block(x, connection, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    name = name + '_concat' if name else None
    return KL.concatenate([x, connection], name=name)

def _conv_trans_block(x, nfeat, strides=1, name=None, do_res=False,hyp_tensor=None):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    extra_conv_params = {}
    if hyp_tensor is not None:
        Conv = getattr(ne.layers, 'HyperConv%dDFromDense' % ndims)
        conv_inputs = [x, hyp_tensor]
    else:
        Conv = getattr(KL, 'Conv%dDTranspose' % ndims)
        extra_conv_params['kernel_initializer'] = 'he_normal'
        conv_inputs = x

    convolved = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', strides=strides, name=name)(x)
    name = name + '_activation' if name else None

    if do_res:
        # assert nfeat == x.get_shape()[-1], 'for residual number of features should be constant'
        add_layer = x
        print('note: this is a weird thing to do, since its not really residual training anymore')
        if nfeat != x.get_shape().as_list()[-1]:
            add_layer = Conv(nfeat, kernel_size=3, padding='same', kernel_initializer='he_normal', name='resfix_'+name)(x)
        convolved = KL.Lambda(lambda x: x[0] + x[1])([add_layer, convolved])

    return KL.LeakyReLU(0.2, name=name)(convolved)
