# load libraries
import os
import datetime

import numpy as np
import pandas as pd
import multiprocessing as mp
import pytorch_lightning as pl

from matplotlib import pylab as plt
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import nibabel as nib


# classes
class ImagePredictionCallback(pl.Callback):
    """ Callback class for evaluation on validation data
    :params: eval_data: PyTorch Dataset object
    :params: experiment_name: name of experiment:
    :params: result_dir: where to save directory
    :params: curr_epoch: current epoch
    :params: eval_internval: the interval of epochs to evulate data
    """
    def __init__(self, eval_data, experiment_name, result_dir, curr_epoch=0, batch_size=4, plot_items = [0], main_channel = 0):
        # costruct dir and save
        experiment_name = "{}__{}_{}_{}".format(
            experiment_name,
            datetime.datetime.now().strftime('%Y'),
            datetime.datetime.now().strftime('%m'),
            datetime.datetime.now().strftime('%d'),
        )


        # make new directory (and increment if necessary)
        curr_iter = 2
        curr_dir = os.path.join(result_dir, experiment_name)
        while os.path.exists(curr_dir):
            curr_dir = os.path.join(result_dir, experiment_name + " - " + str(curr_iter))
            curr_iter += 1
        self.base_dir = curr_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # save information into loader
        self.eval_loader = DataLoader(eval_data, batch_size=batch_size, drop_last=True)
        self.plot_items = plot_items
        self.main_channel = main_channel
        # set input and gt images
        
        # check all the images of shape and do zero padding to make all the images to the same size
        self.input_imgs = None
        self.gt_imgs    = None
        '''
        for x in eval_data:
            if not x[0].shape==given_shape:
                pad_width0 = (given_shape[1]-x[0].shape[1])//2
                pad_width1 = (given_shape[2]-x[0].shape[2])//2
                padded_input = np.pad(x[0],pad_width=((0,0),(pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values=0)
                
                padded_gt = np.pad(x[1],pad_width=((0,0),(pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values=0)
                
                if self.input_imgs is None:
                    self.input_imgs = padded_input
                    self.gt_imgs    = padded_gt
                else:
                    self.input_imgs = np.concatenate([self.input_imgs, padded_input])
                    self.gt_imgs = np.concatenate([self.gt_imgs, padded_gt])
        '''  
        self.input_imgs = np.stack([x[0] for x in eval_data])
        self.input_imgs = np.squeeze(self.input_imgs)
               
        if isinstance(eval_data[0][1], dict):            
            self.gt_imgs = np.stack([x[1]["target"] for x in eval_data])
            self.gt_imgs = np.squeeze(self.gt_imgs)
        else:
            self.gt_imgs = np.stack([x[1] for x in eval_data])
            self.gt_imgs = np.squeeze(self.gt_imgs)
        
        # get dict of indexed slices for each key
        
        da = pd.DataFrame(eval_data.id_slice_time_lst, columns=["key", "slice_indx", "time_indx"])
        # save the case unique names
        unique_keys = da["key"].unique()
        self.data_slice_dict = dict(zip(
            unique_keys,
            [eval_data.get_slice_time_indicies(x) for x in unique_keys],
        ))
        plot_unique_keys = [unique_keys[x] for x in self.plot_items]
        self.plot_data_slice_dict = dict(zip(
            plot_unique_keys,
            [eval_data.get_slice_time_indicies(x) for x in plot_unique_keys],
        ))
        #print(self.data_slice_dict)
        
        # initialize information to use later
        self.curr_epoch = curr_epoch
        self._curr_epoch_dir = None
        self._val_batch_lst = None

    def on_validation_epoch_start(self, trainer, pl_module):
        """ method hook for when epoch ends
        :params: trainer: PL trainer object that defines GPU/multiprocessing/etc.
        :params: pl_module: PL module object that defines model forward pediction
        """
        # set epoch dir
        self._curr_epoch_dir = self._make_epoch_dir(self.curr_epoch)

    def _make_epoch_dir(self, epoch):
        """ constructs epoch directory
        """
        # make new epoch directory
        base_rslt_dir = os.path.join(
            self.base_dir,
            "epoch_{}".format(str(epoch + 1).zfill(5)),
        )
        os.makedirs(base_rslt_dir, exist_ok=True)

        return base_rslt_dir

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """ method hook for when validation batch ends
        :params: trainer: PL trainer object that defines GPU/multiprocessing/etc.
        :params: pl_module: PL module object that defines model forward pediction
        :params: outputs: output from validation
        :params: batch: validation batch
        :params: batch_idx: batch index
        :params: dataloader_idx: the dataloader index
        """
        # test if this is none, if it is, we want to initialize a new list
        if self._val_batch_lst is None:
            self._val_batch_lst = []

        # add to list
        self._val_batch_lst.append(outputs.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """ method hook for when validation epoch ends
        :params: trainer: PL trainer object that defines GPU/multiprocessing/etc.
        :params: pl_module: PL module object that defines model forward pediction
        """
        # Only process if we are on valid epoch
        
        if self.curr_epoch > 0:

            # combine data into a numpy array
            pred_imgs = np.concatenate(self._val_batch_lst)
            pred_imgs = np.squeeze(pred_imgs)

            # print("pred_imgs shape is", pred_imgs.shape)
            # save predicted images in .png slice by slice
            
            # self._save_prediction_images(pred_imgs)
            # sqy testing
            self._save_prediction_nifti(pred_imgs)

            # get result dataframe
            eval_df = self._get_eval_df(pred_imgs)

            # print results and save
            self._print_eval_df(eval_df)
            eval_df.to_csv(os.path.join(self._curr_epoch_dir, "performance.csv"), index=False)
        

        # clean up
        self._val_batch_lst = None

        # increment epoch
        self.curr_epoch += 1

    def _get_eval_df(self, pred_imgs):
        """ method to eval predicted images
        :params: pred_imgs: numpy array of stacked predicted images (order defined by self.data_slice_dict)
        :returns: dataframe of evualated metrics
        """
        # initialize DataFrame
        eval_df = pd.DataFrame()

        # get slice indicies and make matrices
        for curr_id, time_window_indx in self.data_slice_dict.items():
            # get images
            for t in range(len(time_window_indx)):
                window_indices = time_window_indx[t]
                if not window_indices:
                    continue
                #print(curr_id, window_idx)
                window_indx = [window_indices[0],window_indices[-1]]
                pred_mtx = pred_imgs[slice(*window_indx)]
                gt_mtx = self.gt_imgs[slice(*window_indx)]


                # iterate over slices
                for curr_slice in range(window_indx[1] - window_indx[0]):
                    eval_df = eval_df.append(pd.DataFrame({
                        "TestID": [curr_id],
                        "TimePoint": t,
                        "SliceNum": [curr_slice + 1],
                        "ssim": [ssim(
                                    pred_mtx[curr_slice],
                                    gt_mtx[curr_slice],
                                    data_range=1.,
                                ).round(4)],
                        "psnr": [psnr(
                                    pred_mtx[curr_slice],
                                    gt_mtx[curr_slice],
                                    data_range=1.,
                                ).round(4)],
                    })).reset_index(drop=True)
                
        return eval_df

    def _print_eval_df(self, eval_df):
        """ method to construct print statement
        :params: eval_df: evualation dataframe
        """
        print("\n\n\n\n")
        print("Current Performance: ")
        metric_msg = " -- ".join([
            "ssim: {}".format(eval_df["ssim"].mean().round(4)),
            "psnr: {}".format(eval_df["psnr"].mean().round(4)),
        ])
        print(metric_msg)
        print("\n\n\n\n")

    @staticmethod
    def _save_img(curr_save_path, curr_x, curr_y, curr_pred):
        """ save images
        :params: curr_save_path: save path for file
        :params: curr_x: input image
        :params: curr_y: target image
        :params: curr_pred: predicted image
        """
        # define number of rows and columns
        nrows = 1
        ncols = 3


        # create post filtering
        # filtered_img = curr_pred.copy()

        """
        # sharpen
        filtered_img = unsharp_mask(filtered_img)

        # filter through some noise to get better texture
        diff_img = filtered_img - curr_pred
        diff_img *= BLEND_PROP
        filtered_img = curr_pred + diff_img
        """

        # create subplot
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), dpi=175)

        # turn off ticks
        for n in range(ncols):
            ax[n].axis('off')
        
        if curr_x.ndim==3:
            center = int((curr_x.shape[0]-1)/2)
            curr_x = curr_x[center,:,:].squeeze()
            
        try:
            ax[0].imshow(curr_x, cmap='gray')
            ax[0].set_title('Input')

            ax[1].imshow(curr_y, cmap='gray')
            ax[1].set_title('Target')

            ax[2].imshow(curr_pred, cmap='gray')
            ax[2].set_title('Prediction')

            #ax[3].imshow(filtered_img, cmap='gray')
            #ax[3].set_title('Filtered')
        except IndexError:
            print("Problem with {}.".format(curr_save_path))

        # save figure
        fig.savefig(curr_save_path)
        plt.close(fig)
        
    def _save_prediction_nifti(self, pred_imgs):
        assert self._curr_epoch_dir is not None

        # iterate over unique ids and add them to tuple list
        arg_tpl_lst = []
        #for curr_id, window_indx in self.data_slice_dict.items():
        for curr_id, time_window_indx in self.plot_data_slice_dict.items(): #data_slice_dict
            # construct save path
            save_path = os.path.join(self._curr_epoch_dir, "{}".format(curr_id))
            os.makedirs(save_path, exist_ok=True)
            
            time_point = 0
            while not time_window_indx[time_point]: # check if the current time point is empty
                time_point += 1
                
            window_indx = [time_window_indx[time_point][0],time_window_indx[time_point][-1]]  
            # sqy: just save the prediction image for the first time point
            
            # get images
            input_mtx = self.input_imgs[slice(*window_indx)]
            gt_mtx = self.gt_imgs[slice(*window_indx)]
            pred_mtx = pred_imgs[slice(*window_indx)]
            
            if input_mtx.ndim == 4:
                #center = int((input_mtx.shape[1]-1)/2)
                
                input_mtx = input_mtx[:,self.main_channel,:,:].squeeze()
            
            # make the image shape of x y z from slice x y
            
            input_mtx = np.transpose(input_mtx, (1,2,0))
            gt_mtx = np.transpose(gt_mtx, (1,2,0))
            pred_mtx = np.transpose(pred_mtx, (1,2,0))
            
            ni_img = nib.Nifti1Image(pred_mtx, affine=np.eye(4))
            nib.save(ni_img, os.path.join(save_path,'pred.nii'))
            ni_img = nib.Nifti1Image(input_mtx, affine=np.eye(4))
            nib.save(ni_img, os.path.join(save_path,'input.nii'))        
            ni_img = nib.Nifti1Image(gt_mtx, affine=np.eye(4))
            nib.save(ni_img, os.path.join(save_path,'gt.nii'))        
        
        
    def _save_prediction_images(self, pred_imgs):
        """ method to eval predicted images
        :params: pred_imgs: numpy array of stacked predicted images (order defined by self.data_slice_dict)
        """
        assert self._curr_epoch_dir is not None


        # iterate over unique ids and add them to tuple list
        arg_tpl_lst = []
        #for curr_id, window_indx in self.data_slice_dict.items():
        for curr_id, time_window_indx in self.plot_data_slice_dict.items(): #data_slice_dict
            # construct save path
            save_path = os.path.join(self._curr_epoch_dir, "{}".format(curr_id))
            os.makedirs(save_path, exist_ok=True)
            
            time_point = 0
            while not time_window_indx[time_point]: # check if the current time point is empty
                time_point += 1
            window_indx = [time_window_indx[time_point][0],time_window_indx[time_point][-1]]  
            # sqy: just save the prediction image for the first time point
            
            # get images
            input_mtx = self.input_imgs[slice(*window_indx)]
            gt_mtx = self.gt_imgs[slice(*window_indx)]
            pred_mtx = pred_imgs[slice(*window_indx)]


            # iterate over slices
            for curr_slice in range(window_indx[1] - window_indx[0]):
                # make data slices
                curr_tpl = (
                    os.path.join(save_path, "{}.png".format(curr_slice)),
                    input_mtx[curr_slice],
                    gt_mtx[curr_slice],
                    pred_mtx[curr_slice],
                )

                # save into tuple and append
                arg_tpl_lst.append(curr_tpl)


        # starting
        mp.set_start_method('fork', force=True)
        with mp.Pool() as pool:
            pool.starmap(self._save_img, arg_tpl_lst)
            
#############################################################################
class ImagePredictionCallback_MD(pl.Callback):
    """ Callback class for evaluation on validation data
    :params: eval_data: PyTorch Dataset object
    :params: experiment_name: name of experiment:
    :params: result_dir: where to save directory
    :params: curr_epoch: current epoch
    :params: eval_internval: the interval of epochs to evulate data
    """
    def __init__(self, eval_data, experiment_name, result_dir, curr_epoch=0, batch_size=4, plot_items = [0], main_channel = 0):
        # costruct dir and save
        experiment_name = "{}__{}_{}_{}".format(
            experiment_name,
            datetime.datetime.now().strftime('%Y'),
            datetime.datetime.now().strftime('%m'),
            datetime.datetime.now().strftime('%d'),
        )


        # make new directory (and increment if necessary)
        curr_iter = 2
        curr_dir = os.path.join(result_dir, experiment_name)
        while os.path.exists(curr_dir):
            curr_dir = os.path.join(result_dir, experiment_name + " - " + str(curr_iter))
            curr_iter += 1
        self.base_dir = curr_dir
        os.makedirs(self.base_dir, exist_ok=True)
        
        # save information into loader
        self.eval_loader = DataLoader(eval_data, batch_size=batch_size, drop_last=True)
        self.plot_items = plot_items
        self.main_channel = main_channel
        # set input and gt images
        
        # check all the images of shape and do zero padding to make all the images to the same size
        self.input_imgs = None
        self.gt_imgs    = None
        '''
        for x in eval_data:
            if not x[0].shape==given_shape:
                pad_width0 = (given_shape[1]-x[0].shape[1])//2
                pad_width1 = (given_shape[2]-x[0].shape[2])//2
                padded_input = np.pad(x[0],pad_width=((0,0),(pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values=0)
                
                padded_gt = np.pad(x[1],pad_width=((0,0),(pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values=0)
                
                if self.input_imgs is None:
                    self.input_imgs = padded_input
                    self.gt_imgs    = padded_gt
                else:
                    self.input_imgs = np.concatenate([self.input_imgs, padded_input])
                    self.gt_imgs = np.concatenate([self.gt_imgs, padded_gt])
        '''  
        print(len(eval_data))
        self.input_imgs = np.stack([x[0] for x in eval_data])
        self.input_imgs = np.squeeze(self.input_imgs)
               
        if isinstance(eval_data[0][1], dict):            
            self.gt_imgs = np.stack([x[1]["target"] for x in eval_data])
            self.gt_imgs = np.squeeze(self.gt_imgs)
        else:
            self.gt_imgs = np.stack([x[1] for x in eval_data])
            self.gt_imgs = np.squeeze(self.gt_imgs)
        
        # get dict of indexed slices for each key
        
        da = pd.DataFrame(eval_data.id_slice_time_lst, columns=["key", "slice_indx", "time_indx", "pld_indx"])
        # save the case unique names
        unique_keys = da["key"].unique()
        self.data_slice_dict = dict(zip(
            unique_keys,
            [eval_data.get_slice_time_indicies(x) for x in unique_keys],
        ))
        
        plot_unique_keys = [unique_keys[x] for x in self.plot_items]
        
        self.plot_data_slice_dict = dict(zip(
            plot_unique_keys,
            [eval_data.get_slice_time_indicies(x) for x in plot_unique_keys],
        ))
        
        # initialize information to use later
        self.curr_epoch = curr_epoch
        self._curr_epoch_dir = None
        self._val_batch_lst = None

    def on_validation_epoch_start(self, trainer, pl_module):
        """ method hook for when epoch ends
        :params: trainer: PL trainer object that defines GPU/multiprocessing/etc.
        :params: pl_module: PL module object that defines model forward pediction
        """
        # set epoch dir
        self._curr_epoch_dir = self._make_epoch_dir(self.curr_epoch)

    def _make_epoch_dir(self, epoch):
        """ constructs epoch directory
        """
        # make new epoch directory
        base_rslt_dir = os.path.join(
            self.base_dir,
            "epoch_{}".format(str(epoch + 1).zfill(5)),
        )
        os.makedirs(base_rslt_dir, exist_ok=True)

        return base_rslt_dir

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """ method hook for when validation batch ends
        :params: trainer: PL trainer object that defines GPU/multiprocessing/etc.
        :params: pl_module: PL module object that defines model forward pediction
        :params: outputs: output from validation
        :params: batch: validation batch
        :params: batch_idx: batch index
        :params: dataloader_idx: the dataloader index
        """
        # test if this is none, if it is, we want to initialize a new list
        if self._val_batch_lst is None:
            self._val_batch_lst = []

        # add to list
        self._val_batch_lst.append(outputs.cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """ method hook for when validation epoch ends
        :params: trainer: PL trainer object that defines GPU/multiprocessing/etc.
        :params: pl_module: PL module object that defines model forward pediction
        """
        # Only process if we are on valid epoch
        
        if self.curr_epoch > 0:

            # combine data into a numpy array
            pred_imgs = np.concatenate(self._val_batch_lst)
            pred_imgs = np.squeeze(pred_imgs)

            # print("pred_imgs shape is", pred_imgs.shape)
            # save predicted images in .png slice by slice
            
            # self._save_prediction_images(pred_imgs)
            # sqy testing
            self._save_prediction_nifti(pred_imgs)

            # get result dataframe
            eval_df = self._get_eval_df(pred_imgs)

            # print results and save
            self._print_eval_df(eval_df)
            eval_df.to_csv(os.path.join(self._curr_epoch_dir, "performance.csv"), index=False)
        

        # clean up
        self._val_batch_lst = None

        # increment epoch
        self.curr_epoch += 1

    def _get_eval_df(self, pred_imgs):
        """ method to eval predicted images
        :params: pred_imgs: numpy array of stacked predicted images (order defined by self.data_slice_dict)
        :returns: dataframe of evualated metrics
        """
        # initialize DataFrame
        eval_df = pd.DataFrame()

        # get slice indicies and make matrices
        for curr_id, time_window_indx in self.data_slice_dict.items():
            # get images
            for pld in range(len(time_window_indx)):
                for t in range(len(time_window_indx[pld])):
                    window_indices = time_window_indx[pld][t]
                    if not window_indices:
                        continue
                    #print(curr_id, window_idx)
                    window_indx = [window_indices[0],window_indices[-1]]
                    pred_mtx = pred_imgs[slice(*window_indx)]
                    gt_mtx = self.gt_imgs[slice(*window_indx)]

                    # iterate over slices
                    for curr_slice in range(window_indx[1] - window_indx[0]):
                        eval_df = eval_df.append(pd.DataFrame({
                            "TestID": [curr_id],
                            "TimePoint": t,
                            "SliceNum": [curr_slice + 1],
                            "PostLabelingDelay": pld,
                            "ssim": [ssim(
                                        pred_mtx[curr_slice],
                                        gt_mtx[curr_slice],
                                        data_range=1.,
                                    ).round(4)],
                            "psnr": [psnr(
                                        pred_mtx[curr_slice],
                                        gt_mtx[curr_slice],
                                        data_range=1.,
                                    ).round(4)],
                        })).reset_index(drop=True)

        return eval_df

    def _print_eval_df(self, eval_df):
        """ method to construct print statement
        :params: eval_df: evualation dataframe
        """
        print("\n\n\n\n")
        print("Current Performance: ")
        metric_msg = " -- ".join([
            "ssim: {}".format(eval_df["ssim"].mean().round(4)),
            "psnr: {}".format(eval_df["psnr"].mean().round(4)),
        ])
        print(metric_msg)
        print("\n\n\n\n")

    @staticmethod
    def _save_img(curr_save_path, curr_x, curr_y, curr_pred):
        """ save images
        :params: curr_save_path: save path for file
        :params: curr_x: input image
        :params: curr_y: target image
        :params: curr_pred: predicted image
        """
        # define number of rows and columns
        nrows = 1
        ncols = 3


        # create post filtering
        # filtered_img = curr_pred.copy()

        """
        # sharpen
        filtered_img = unsharp_mask(filtered_img)

        # filter through some noise to get better texture
        diff_img = filtered_img - curr_pred
        diff_img *= BLEND_PROP
        filtered_img = curr_pred + diff_img
        """

        # create subplot
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), dpi=175)

        # turn off ticks
        for n in range(ncols):
            ax[n].axis('off')
        
        if curr_x.ndim==3:
            center = int((curr_x.shape[0]-1)/2)
            curr_x = curr_x[center,:,:].squeeze()
            
        try:
            ax[0].imshow(curr_x, cmap='gray')
            ax[0].set_title('Input')

            ax[1].imshow(curr_y, cmap='gray')
            ax[1].set_title('Target')

            ax[2].imshow(curr_pred, cmap='gray')
            ax[2].set_title('Prediction')

            #ax[3].imshow(filtered_img, cmap='gray')
            #ax[3].set_title('Filtered')
        except IndexError:
            print("Problem with {}.".format(curr_save_path))

        # save figure
        fig.savefig(curr_save_path)
        plt.close(fig)
        
    def _save_prediction_nifti(self, pred_imgs):
        assert self._curr_epoch_dir is not None

        # iterate over unique ids and add them to tuple list
        arg_tpl_lst = []
        
        # need to change later, plot for all five PLDs for each subject
        for curr_id, time_window_indx in self.plot_data_slice_dict.items(): #data_slice_dict
            save_path = os.path.join(self._curr_epoch_dir, "{}".format(curr_id))
            os.makedirs(save_path, exist_ok=True)
            for pld in range(len(time_window_indx)):
                # construct save path
                window_indx = [time_window_indx[pld][0][0],time_window_indx[pld][0][-1]]  
                # sqy: just save the prediction image for the first time point

                # get images
                input_mtx = self.input_imgs[slice(*window_indx)]
                gt_mtx = self.gt_imgs[slice(*window_indx)]
                pred_mtx = pred_imgs[slice(*window_indx)]

                if input_mtx.ndim == 4:
                    #center = int((input_mtx.shape[1]-1)/2)

                    input_mtx = input_mtx[:,self.main_channel,:,:].squeeze()

                # make the image shape of x y z from slice x y

                input_mtx = np.transpose(input_mtx, (1,2,0))
                gt_mtx = np.transpose(gt_mtx, (1,2,0))
                pred_mtx = np.transpose(pred_mtx, (1,2,0))

                ni_img = nib.Nifti1Image(pred_mtx, affine=np.eye(4))
                nib.save(ni_img, os.path.join(save_path,'pred_PLD' + str(pld+1) + '.nii'))
                ni_img = nib.Nifti1Image(input_mtx, affine=np.eye(4))
                nib.save(ni_img, os.path.join(save_path,'input_PLD' + str(pld+1) + '.nii'))        
                ni_img = nib.Nifti1Image(gt_mtx, affine=np.eye(4))
                nib.save(ni_img, os.path.join(save_path,'gt_PLD' + str(pld+1) + '.nii'))        
        
        
    def _save_prediction_images(self, pred_imgs):
        """ method to eval predicted images
        :params: pred_imgs: numpy array of stacked predicted images (order defined by self.data_slice_dict)
        """
        assert self._curr_epoch_dir is not None


        # iterate over unique ids and add them to tuple list
        arg_tpl_lst = []
        #for curr_id, window_indx in self.data_slice_dict.items():
        for curr_id, time_window_indx in self.plot_data_slice_dict.items(): #data_slice_dict
            # construct save path
            save_path = os.path.join(self._curr_epoch_dir, "{}".format(curr_id))
            os.makedirs(save_path, exist_ok=True)
            
            time_point = 0
            while not time_window_indx[time_point]: # check if the current time point is empty
                time_point += 1
            window_indx = [time_window_indx[time_point][0],time_window_indx[time_point][-1]]  
            # sqy: just save the prediction image for the first time point
            
            # get images
            input_mtx = self.input_imgs[slice(*window_indx)]
            gt_mtx = self.gt_imgs[slice(*window_indx)]
            pred_mtx = pred_imgs[slice(*window_indx)]


            # iterate over slices
            for curr_slice in range(window_indx[1] - window_indx[0]):
                # make data slices
                curr_tpl = (
                    os.path.join(save_path, "{}.png".format(curr_slice)),
                    input_mtx[curr_slice],
                    gt_mtx[curr_slice],
                    pred_mtx[curr_slice],
                )

                # save into tuple and append
                arg_tpl_lst.append(curr_tpl)


        # starting
        mp.set_start_method('fork', force=True)
        with mp.Pool() as pool:
            pool.starmap(self._save_img, arg_tpl_lst)
            
   