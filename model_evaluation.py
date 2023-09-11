# check import modules
import os
import h5py
import yaml
import random
import argparse

import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib import pylab as plt
import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from torch.nn.functional import l1_loss, binary_cross_entropy

# define which GPU we want to use
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
GPU_DEVICE = [1]

from src.image_processing import \
    NormalizeData, ClipData, PatchPairedImage, \
    AugmentRandomRotation, AugmentHorizonalFlip

from src.data_utils import LOFT4DMRData, LOFT4DMRDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

from Models.edsr import edsr_hr
from Models.network_swinir_ver2 import SwinIR_V2
from torch.utils.data import Dataset, DataLoader


#######################################################################################################################
## Model Parameters

# structure of EXP_NAME, MODEL_TYPE, SLICE_NUM and USE_M0
model_list = [('baseline_2D_EDSR' ,  'EDSR',   1,  False), 
              ('pseudo3d_3_EDSR' ,   'EDSR',   3,  False),
              ('pseudo3D_5_EDSR' ,   'EDSR',   5,  False),
              ('pseudo3D_7_EDSR' ,   'EDSR',   7,  False),
              ('EDSR_2D_withM0',     'EDSR',   1,  True),
              ('pseudo3d_3_EDSR_M0channel','EDSR',3,True),
              ('2D_SWINIR' ,  'SWINIR',   1,  False), 
              ('pseudo3d_3_SWINIR' ,   'SWINIR',   3,  False),
              ('pseudo3d_5_SWINIR' ,   'SWINIR',   5,  False),
              ('pseudo3d_7_SWINIR' ,   'SWINIR',   7,  False),
              ('SWINIR_2D_withM0',     'SWINIR',   1,  True),
              ('SWINIR_slice3_withM0','SWINIR',3,True),]
######################################################################################################################
for model in model_list[9:]:

    EXP_NAME = model[0]
    SLICE_NUM = model[2] # input channels
    THREE_DIM = False # For 3D models
    USE_M0 = model[3]
    MODEL_TYPE = model[1] # EDSR / SWINIR 
    tmp_output_path = os.path.join('/ifs/loni/groups/loft/qinyang/KWIA_DL_project/model_predictions_2',EXP_NAME)
    with open('/ifs/loni/groups/loft/qinyang/KWIA_DL_project/trained_models_4/' + EXP_NAME + '/best_model_checkpoint.txt','r') as file:
        MODEL_CHECKPOINT_PATH = file.read()

    print(MODEL_CHECKPOINT_PATH)
    os.makedirs(tmp_output_path, exist_ok='True')
    ########################################################################################################################
    # define model
    if MODEL_TYPE == 'EDSR':
        class TestEDSR(edsr_hr, pl.LightningModule):
            def __init__(self, *args, **kwargs):
                super(TestEDSR, self).__init__(*args, **kwargs)

            def predict_step(self, batch, batch_idx):
                # split data and convert to fp32
                input = batch[0].type(torch.float32)

                # forward step
                prediction = self(input)

                return prediction

            # initialize model

        LAYERS = 20
        edsr_model = TestEDSR.load_from_checkpoint(
            MODEL_CHECKPOINT_PATH,
            layers = LAYERS,
            img_channel = SLICE_NUM + USE_M0,
        )
        test_model = edsr_model

    elif MODEL_TYPE == 'SWINIR':
        class TestSwinIR(SwinIR_V2, pl.LightningModule):
            def __init__(self, *args, **kwargs):
                super(TestSwinIR, self).__init__(*args, **kwargs)

            def predict_step(self, batch, batch_idx):
                # split data and convert to fp32
                input = batch[0].type(torch.float32)

                # forward step
                prediction = self(input)

                return prediction

                # initialize model
        SWIN_MODEL_PARAMS = {
            "upscale": 1,
            "in_chans": SLICE_NUM + USE_M0,
            "img_size": 48,
            "window_size": 8,
            "img_range": 1.,
            "depths": [6] * 6,
            "embed_dim": 84,
            "num_heads": [6] * 6,
            "mlp_ratio": 2.,
            "upsampler": None,
            "resi_connection": "1conv",
        }

        swin_model = TestSwinIR.load_from_checkpoint(
            # model checkpoint
            MODEL_CHECKPOINT_PATH,
            # defines training hyperparameters
            **SWIN_MODEL_PARAMS,
        ) 
        test_model = swin_model

    from src.data_utils import LOFT4DMRPredDataset, LOFT4DMRPredDataset_multislice
    class ModelPredictor(object):
        """ class to contain logic for prediction
        :params: model: pl model
        :params: preproc_lst: list of preprocessing steps
        :params: postproc_lst: list of postprocessing steps
        :params: batch_size: prediction batch size

        :note: if objects needs information from preprocessing, we need to initialize
        object first then and pass the initialized object to each list
        """


        def __init__(self, model, slice_num = 1, preproc_lst=None, postproc_lst=None, batch_size=None):
            # save data
            self.model = model
            self.preproc_lst = preproc_lst if preproc_lst is not None else []
            self.postproc_lst = postproc_lst if postproc_lst is not None else []
            self.batch_size = batch_size
            self.slice_num = slice_num

        def _predict_on_pred_dataset(self, pred_dset):

            """ predict from a hdf5 path
            :params: smr_dataset: SubtleMRDataset object
            :returns: predicted image list
            """
            # determine batch
            if self.batch_size is None:
                curr_batch_size = 1

            # otherwise just set it
            else:
                curr_batch_size = self.batch_size

            # predict
            rslt_img_lst = pl.Trainer(accelerator = 'gpu', devices = GPU_DEVICE).predict(
                # define model
                self.model,

                # define dataloader
                dataloaders=DataLoader(pred_dset, batch_size=curr_batch_size),
            )

            rslt_img = np.concatenate(rslt_img_lst)
            # print(np.min(rslt_img), np.max(rslt_img))
            # apply post processing

            #for curr_postproc in postproc_lst:
            for curr_postproc in self.postproc_lst:
                if hasattr(curr_postproc, "post_process"):
                    rslt_img = curr_postproc.post_process(rslt_img)

            rslt_img_mtx = rslt_img
            return rslt_img_mtx

        def get_prediction(self, input_mtx):
            """ runs prediction
            :params: input_mtx, input matrix for each case
            :returns: predicted image for that case
            """
            # make dataset
            if self.slice_num == 1:
                pred_dat = LOFT4DMRPredDataset(input_mtx, self.preproc_lst)
            elif self.slice_num > 1:
                pred_dat = LOFT4DMRPredDataset_multislice(input_mtx, num_channels = self.slice_num, preproc_lst=self.preproc_lst)

            return self._predict_on_pred_dataset(pred_dat)   

        def get_prediction_withM0(self, input_mtx, input_M0):
            """ runs prediction
            :params: input_mtx, input matrix for each case
            :params: input_M0, input M0 image as an additional channel to the image
            :returns: predicted image for that case
            """
            ## to be implemented
            # make dataset
            pred_dat = LOFT4DMRPredDataset_multislice(input_mtx, input_M0, num_channels = self.slice_num, preproc_lst=self.preproc_lst)

            return self._predict_on_pred_dataset(pred_dat) 

        def __call__(self, input_dicom_path):
            """ call prediction method
            :params: input_dicom_path: directory which contains dicom files
            :returns: predicted image
            """
            return self.get_prediction(input_dicom_path)

    from src.image_processing import NormalizePostProcess, NormalizePostProcessSlice, ClipDataSlice
    norm_post_proc = NormalizePostProcess(use_M0 = USE_M0)
    clip = ClipData(0.1)
    predictor = ModelPredictor(
        test_model,
        preproc_lst=[norm_post_proc],
        postproc_lst=[norm_post_proc],
        slice_num = SLICE_NUM + USE_M0,
        batch_size=4,
    )
    ################################# Start of evaluation ##################################################

    import glob
    from tqdm import tqdm
    import nibabel as nib
    from src.evaluation_metrics import cal_ssim_cases, cal_psnr_cases, im_scaling, im_scaling_slice
    study_pre = 'GE750W'
    os.makedirs(os.path.join(tmp_output_path,study_pre),exist_ok=True)
    input_data_path = '/ifs/loni/groups/loft/KWIA_DL/test_data/GE_750W'
    case_list = [os.path.basename(x) for x in glob.glob("/ifs/loni/groups/loft/KWIA_DL/test_data/GE_750W/*")]
    print(case_list)
    # number of subjects
    for case in tqdm(case_list): ## subject level

        print(case)   
        img = nib.load(os.path.join(input_data_path,case,'PWI_mean.nii'))    
        input_img = np.array(img.get_fdata())
        input_img = input_img.squeeze()
        input_mtx = np.transpose(input_img,(2,0,1))

        mask_path = os.path.join(input_data_path,case,'brain_mask.nii')
        mask_img = nib.load(mask_path).get_fdata()
        mask_img[mask_img>0]=1
        mask_mtx = np.transpose(mask_img,(2,0,1))

        input_mtx = input_mtx

        if USE_M0:
            M0_path = os.path.join(input_data_path,case,'M0.nii')
            M0_img = nib.load(M0_path).get_fdata()
            M0_img = np.squeeze(M0_img)
            print(M0_img.shape)
            M0_mtx = np.transpose(M0_img,(2,0,1))               
            result_mtx = predictor.get_prediction_withM0(
                input_mtx,
                M0_mtx,
            )
        else:    
            result_mtx = predictor.get_prediction(
                # input path
                input_mtx,
            )



        result_mtx = np.squeeze(result_mtx)
        result_mtx = im_scaling_slice(result_mtx, input_mtx, mask_mtx)
        result_img = np.transpose(result_mtx,(1,2,0))

        # apply mask        
        masked_img = result_img * mask_img[:,:,:]        
        # save input and prediction images
        ni_img = nib.Nifti1Image(result_img, affine=nib.load(mask_path).affine)
        nib.save(ni_img, os.path.join(tmp_output_path,study_pre,case+'_pred.nii'))

        input_img = np.transpose(input_mtx,(1,2,0))
        ni_img = nib.Nifti1Image(input_img, affine=nib.load(mask_path).affine)
        nib.save(ni_img, os.path.join(tmp_output_path,study_pre,case+'_input.nii'))

   


           
            

   