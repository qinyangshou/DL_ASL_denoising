# model evaluation for the test data in hdf5
import os
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import nibabel as nib
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICE"] = "0"
GPU_DEVICE = [0]

#############################################
MODEL_TYPE = 'SWINIR'
SLICE_NUM  = 3
USE_M0     = False
INPUT_DIM = 2 # compared to 3d
IMAGE_SHAPE = (96,96,48) # only used in 3D case
EXP_NAME = 'pseudo3d_3_SWINIR'
#############################################

tmp_output_path = os.path.join('./Model_predictions',EXP_NAME)
MODEL_CHECKPOINT_PATH = './Trained_models/' + EXP_NAME + '/best_model_checkpoint.ckpt'

if MODEL_TYPE == 'EDSR':
    from Models.edsr import edsr_hr as target_model
    MODEL_PARAMS = {"layers":20, "in_channels":SLICE_NUM, "global_conn":True,"main_channel":int(SLICE_NUM/2)}
elif MODEL_TYPE == 'SWINIR':
    from Models.network_swinir_ver2 import SwinIR_V2 as target_model
    MODEL_PARAMS = {
        "upscale": 1,
        "in_chans": SLICE_NUM,
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
elif MODEL_TYPE == 'DWAN':
    from Models.DWAN import DWAN_network as target_model 
    MODEL_PARAMS = {"img_channel": SLICE_NUM}

elif MODEL_TYPE == 'UNETR':
    from Models.unetr import UNETR_mini as target_model
    MODEL_PARAMS = {"img_shape":IMAGE_SHAPE, }
elif MODEL_TYPE == 'TransUnet':
    from Models.vit_seg_modeling import VisionTransformer_restore as target_model
    import Models.vit_seg_configs as configs
    MODEL_PARAMS = {"img_size":96,"config":configs.get_r50_b16_mini_config()}

else:
    print("No modules found!!")

class TestModel(target_model, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super(TestModel, self).__init__(*args, **kwargs)
    
    def predict_step(self, batch, batch_idx):
        input = batch[0].type(torch.float32)

        prediction = self(input)

        return prediction
    
test_model = TestModel.load_from_checkpoint(
    MODEL_CHECKPOINT_PATH,
    **MODEL_PARAMS,
)
print('## Model loaded successfully ##')

from src.data_utils import LOFT4DMRPredDataset, LOFT4DMRPredDataset_multislice
from src.data_utils_3D import LOFT4DMRPredDataset_Volume
from torch.utils.data import DataLoader
# class for 2D data prediction
class ModelPredictor(object):
    """ class to contain logic for prediction
    :params: model: pl model
    :params: preproc_lst: list of preprocessing steps
    :params: postproc_lst: list of postprocessing steps
    :params: batch_size: prediction batch size

    :note: if objects needs information from preprocessing, we need to initialize
    object first then and pass the initialized object to each list
    """
    def __init__(self, model, slice_num = 1, preproc_lst=None, postproc_lst=None, batch_size=None, input_dim=2, img_shape = None):
        # save data
        self.model = model
        self.preproc_lst = preproc_lst if preproc_lst is not None else []
        self.postproc_lst = postproc_lst if postproc_lst is not None else []
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.input_dim = input_dim
        self.img_shape = img_shape

    def _predict_on_pred_dataset(self, pred_dset):
        
        """ predict from a hdf5 path
        :params: pred_dset: LOFTMRPredDataset object
        :returns: predicted image list
        """
        # determine batch
        if self.batch_size is None:
            curr_batch_size = 1
        else:
            curr_batch_size = self.batch_size

        # predict
        rslt_img_lst = pl.Trainer(accelerator = 'gpu', devices = GPU_DEVICE).predict(
            # define model
            self.model,

            # define dataloader
            dataloaders=DataLoader(pred_dset, batch_size=curr_batch_size),
        )
        print(rslt_img_lst[0].shape)
        if(self.input_dim==2):
            rslt_img = np.concatenate(rslt_img_lst)
        elif(self.input_dim==3):
            rslt_img = rslt_img_lst[0]
        # apply post processing
        for curr_postproc in self.postproc_lst:
            if hasattr(curr_postproc, "post_process"):
                rslt_img = curr_postproc.post_process(rslt_img)

        return rslt_img
            
    def get_prediction(self, input_mtx):
        """ runs prediction
        :params: input_mtx, input matrix for each case
        :returns: predicted image for that case
        """
        # make dataset
        if self.input_dim==2 and self.slice_num == 1:
            pred_dat = LOFT4DMRPredDataset(input_mtx, self.preproc_lst)
        elif self.input_dim==2 and self.slice_num > 1:
            pred_dat = LOFT4DMRPredDataset_multislice(input_mtx, num_channels = self.slice_num, preproc_lst=self.preproc_lst)
        elif self.input_dim==3 and (self.img_shape is not None):
            pred_dat = LOFT4DMRPredDataset_Volume(input_mtx, img_shape = self.img_shape, preproc_lst=self.preproc_lst)
        else:
            print('Please check your predictor input arguments')

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
           
from src.image_processing import NormalizePostProcess
norm_post_proc = NormalizePostProcess(use_M0 = USE_M0)
predictor = ModelPredictor(
    test_model,
    preproc_lst = [norm_post_proc],
    postproc_lst = [norm_post_proc],
    slice_num = SLICE_NUM + USE_M0,
    input_dim= INPUT_DIM,
    img_shape = IMAGE_SHAPE,
    batch_size = 4,
)

from src.evaluation_metrics import im_scaling_slice
study_pre = 'Example'
os.makedirs(os.path.join(tmp_output_path,study_pre),exist_ok=True)
input_data_path = "./Data/EvaluationData"
case_list = [os.path.basename(x) for x in glob.glob(input_data_path + "/*")]

for case in tqdm(case_list): 
    input_data = [os.path.basename(x) for x in glob.glob(input_data_path + '/' + case + "/input*.nii")]
    os.makedirs(os.path.join(tmp_output_path,study_pre,case),exist_ok=True)
    for data in input_data : 
        
        img = nib.load(os.path.join(input_data_path,case,data))    
        input_img = np.array(img.get_fdata())
        input_img = input_img.squeeze()
        input_mtx = np.transpose(input_img,(2,0,1))

        mask_path = os.path.join(input_data_path,case,'mask.nii')
        mask_img = nib.load(mask_path).get_fdata()
        mask_img[mask_img>0]=1
        mask_mtx = np.transpose(mask_img,(2,0,1))

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
      
        # save input and prediction images
        ni_img = nib.Nifti1Image(result_img, affine=nib.load(mask_path).affine)
        nib.save(ni_img, os.path.join(tmp_output_path,study_pre,case+'/pred_'+data))
