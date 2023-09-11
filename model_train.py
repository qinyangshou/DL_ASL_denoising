# Code written by Qinyang Shou: qinyangs@usc.edu
import os
import h5py
import yaml
import random
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_DEVICE = [1]
N_WORKERS = 24 

# define hyperparameters
SLICE_NUM = 3
PATCH_SIZE = (48, 48)
TRAIN_BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 10
SEED = 80085 

MODEL_BASE_PATH = 'Trained_models'
EXP_NAME = 'test_training'
USE_CKPT = False
if USE_CKPT:
    with open('Trained_models/' + EXP_NAME + '/best_model_checkpoint.txt','r') as file:
        MODEL_CHECKPOINT_PATH = file.read()

MODEL_TYPE = 'SWINIR'
USE_M0= False

# set random seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# define data augmentation

from src.image_processing import PatchPairedImage, AugmentRandomRotation, AugmentHorizonalFlip
AUG_ROT_RANGE = (-60, 60)
train_preproc_lst = [
    AugmentRandomRotation(AUG_ROT_RANGE),
    AugmentHorizonalFlip(),        
    PatchPairedImage(patch_size=PATCH_SIZE,apply_before_ksp = True),
]
valid_preproc_lst = [PatchPairedImage(patch_size=PATCH_SIZE,apply_before_ksp = True)]
from src.data_utils import LOFT4DMRData

############### data loading #############
data_path = './Data/ExampleTrainData/Example_train_valid_data.hdf5'
data_conn = h5py.File(data_path,"r")
csv_path = './Data/ExampleTrainData/Example_train_valid_data.csv'
csv = pd.read_csv(csv_path)
data_splits = {
    "train": csv[csv["train_val_test"] == "TRAIN"]["subject_ID"].tolist(),
    "validation": csv[csv["train_val_test"] == "VALID"]["subject_ID"].tolist(),
}
data = LOFT4DMRData(data_conn, data_splits, prop_cull=[0.1,0.1],slice_axis=-2, time_axis=-1, remove_spike=True)
train_data = data.generate_dataset_multislice("train", preproc_lst=train_preproc_lst, input_name = "input/data_input", target_name = "target_ave/dset_target_ave",slice_num=SLICE_NUM,  use_M0 = USE_M0)
valid_data = data.generate_dataset_multislice("validation", preproc_lst = valid_preproc_lst, input_name = "input/data_input", target_name = "target_ave/dset_target_ave",  slice_num=SLICE_NUM, use_M0 = USE_M0) 


# define loss function
from torch.nn.functional import l1_loss
from src.loss_functions import CombinedLoss, SSIM_Loss
combined_loss = CombinedLoss(
    loss_lst = [
        l1_loss,
        SSIM_Loss(data_range=1.0, channel=1),
    ],
)

# define common training parameters
if USE_M0:
    SLICE_NUM = SLICE_NUM + 1   

COMMON_PARAMS = {"loss_fn": combined_loss}
# define model and model specific parameters
if MODEL_TYPE == 'EDSR':
    from Models.edsr import edsr_hr as target_model
    MODEL_PARAMS = {"layers":20, "in_channels":SLICE_NUM, "global_conn":True,"main_channel":int(SLICE_NUM/2)}
elif MODEL_TYPE == 'SWINIR':
    from Models.network_swinir_ver2 import SwinIR_V2 as target_model
    MODEL_PARAMS = {
        "upscale": 1,
        "in_chans": SLICE_NUM,
        "img_size": PATCH_SIZE[0],
        "window_size": 8,
        "img_range": 1.,
        "depths": [6] * 6,
        "embed_dim": 84,
        "num_heads": [6] * 6,
        "mlp_ratio": 2.,
        "upsampler": None,
        "resi_connection": "1conv",
        "drop_path_rate":0,
    }
elif MODEL_TYPE == 'DWAN':
    from Models.DWAN import DWAN_network as target_model 
    MODEL_PARAMS = {"img_channel": SLICE_NUM}
elif MODEL_TYPE == 'TransUnet':
    from Models.vit_seg_modeling import VisionTransformer_restore as target_model
    import Models.vit_seg_configs as configs
    MODEL_PARAMS = {"img_size":96,"config":configs.get_r50_b16_mini_config()}
else:
    print("No modules found!!")

class TrainingModule(target_model, pl.LightningModule):
    def __init__(self, loss_fn, *args, **kwargs):
        super(TrainingModule, self).__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.step_times = 0

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True)
        
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "frequency": 10,
            }
        }

    def training_step(self, batch, batch_nb):
        input = batch[0].type(torch.float32)
        target = batch[1].type(torch.float32)

        # forward step
        output = self(input)

        # calculate loss
        loss = self.loss_fn(output, target) 
        self.log("train_loss_123", loss, on_epoch=True, on_step=None)
        
        if batch_nb % 20 == 0:
            num_channels = input.shape[1]
            input_img  = input[0,int(num_channels/2),:,:].detach().cpu().numpy().squeeze() 
            pred_img   = output[0,0,:,:].detach().cpu().numpy().squeeze()
            target_img = target[0,0,:,:].detach().cpu().numpy().squeeze()
            self.logger.experiment.add_image("train input img" ,input_img,dataformats='HW')
            self.logger.experiment.add_image("train pred img"  ,pred_img,dataformats='HW')
            self.logger.experiment.add_image("train target img",target_img,dataformats='HW')
                
        return loss

    def validation_step(self, batch, batch_idx):
        # split data and convert to fp32
        input = batch[0].type(torch.float32)
        target = batch[1].type(torch.float32)

        # forward step
        prediction = self(input)

        # calculate loss and log
        loss = self.loss_fn(prediction, target)
        self.log("val_loss", loss, on_epoch=True, on_step=None)

        # log images
        num_channels = input.shape[1]
        if batch_idx % 20 == 0:
            input_img  = input[0,int(num_channels/2),:,:].cpu().numpy().squeeze() 
            pred_img   = prediction[0,0,:,:].cpu().numpy().squeeze()
            target_img = target[0,0,:,:].cpu().numpy().squeeze()
            self.logger.experiment.add_image("valiation input img",input_img,dataformats='HW')
            self.logger.experiment.add_image("validation pred img",pred_img,dataformats='HW')
            self.logger.experiment.add_image("validation target img",target_img,dataformats='HW')

        return loss

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log("train_loss_epoch", avg_loss, on_epoch=True, on_step = None)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log("valid_loss_epoch", avg_loss, on_epoch=True, on_step = None)
        
if USE_CKPT:
    model = TrainingModule.load_from_checkpoint(
        # set loss
        MODEL_CHECKPOINT_PATH,
        **COMMON_PARAMS,
        **MODEL_PARAMS,
    )
else:    
    model = TrainingModule(
        **COMMON_PARAMS,
        **MODEL_PARAMS,
    )
   
# print(model)

MODEL_SAVE_PATH = os.path.join(MODEL_BASE_PATH,EXP_NAME)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# construct model parameters
hyperparam_dict = {
    "model_class": MODEL_TYPE,
    "loss_function": combined_loss.get_losses_and_weights(),
    "epochs": EPOCHS,
    "batch_size": TRAIN_BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "experiment_name": EXP_NAME,
    "train_process_lst": train_preproc_lst,
    "patch_size": PATCH_SIZE,
    "aug_rot_range": AUG_ROT_RANGE,
}
# save hyperparameters
with open(os.path.join(MODEL_SAVE_PATH, "hyperparams.yaml"), "w") as file:
    yaml.dump(hyperparam_dict, file)

last_checkpoint_callback = ModelCheckpoint(
    every_n_epochs=5,
    dirpath=MODEL_SAVE_PATH,
    filename="model_checkpoint",
    save_top_k=5,
    monitor = 'val_loss',
)
logger = TensorBoardLogger("TensorboardLoggers", name =  EXP_NAME)

# initialize trainer and start training
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator = 'gpu',
    devices = GPU_DEVICE,    
    callbacks=[
        last_checkpoint_callback,
        RichProgressBar(),
    ],
    check_val_every_n_epoch=5,
    logger = logger,
    log_every_n_steps=10,
    auto_lr_find=True,
)

# train
trainer.fit(
    model=model,
    train_dataloaders=torch.utils.data.DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE,
        num_workers = N_WORKERS, drop_last=True),
    val_dataloaders=torch.utils.data.DataLoader(valid_data, batch_size=4, drop_last=False),
)

# save the best model checkpoint
last_checkpoint_callback.best_model_path
save_path = MODEL_SAVE_PATH
name_of_file = "best_model_checkpoint"
completeName = os.path.join(save_path, name_of_file+".txt")         
file1 = open(completeName, "w")
file1.write(last_checkpoint_callback.best_model_path)
file1.close()
print(last_checkpoint_callback.best_model_path)
