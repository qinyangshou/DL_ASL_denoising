import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from image_processing import fft_im, ifft_im
from scipy.ndimage import zoom
import time

class LOFT4DMRData3D(object):
    
    def __init__(self, h5_data, split_dict, prop_cull=None, slice_axis=-2, time_axis=-1, remove_spike=True):
        
        self.data = h5_data
        self.split_dict = split_dict
        self.prop_cull = [0., 0.] if prop_cull is None else prop_cull
        self.slice_axis = slice_axis
        self.time_axis = time_axis
        self.remove_spike = remove_spike
        
    def _get_id_time_lst(self, id_lst):
        
        # make range list of time points 
        time_range_lst = [(0, np.array(self.data[x + '/input/data_input']).shape[self.time_axis]) for x in id_lst]       

        # make slice lists and unlist nested list
        id_time_lst = [[(x, y) for y in range(*y)] for x, y in zip(id_lst, time_range_lst)]
        
        refined_id_time_lst = []
        for id in range(len(id_time_lst)):
            curr_id = id_lst[id]
            #print(curr_id)
            curr_4Dvolume = np.array(self.data[curr_id + '/input/data_input'])
            curr_mask = np.array(self.data[curr_id + '/mask/dset_mask'])
            mean_map   = np.mean(curr_4Dvolume, axis=-1)
            mean_value = np.mean(mean_map[curr_mask>0])
            time_value_lst = []
            for t in range(len(id_time_lst[id])):
                tmp_volume = curr_4Dvolume[:,:,:,t].squeeze()
                time_value_lst.append(np.mean(tmp_volume[curr_mask>0]))
            #print(time_value_lst)
            std_value = np.std(time_value_lst)
            
            if self.remove_spike:            
                for t in range(len(id_time_lst[id])):               
                    if not time_value_lst[t]>mean_value + 2*std_value or time_value_lst[t]<mean_value-2*std_value:
                        refined_id_time_lst.append((curr_id,t))
            else:
                for t in range(len(id_time_lst[id])):               
                    refined_id_time_lst.append((curr_id,t))

        # remove input if necessary
        #id_slice_lst = [(x[0].replace("/input", ""), x[1]) for x in id_slice_lst]
        return refined_id_time_lst
    
    def generate_dataset(self, mode, *args, **kwargs):
        
        assert mode in self.split_dict
        
        id_lst = self.split_dict[mode]
        id_time_lst = self._get_id_time_lst(id_lst)
        
        return LOFT4DMRDataset_Volume(self.data, id_time_lst, slice_axis=self.slice_axis, *args, **kwargs)
    

class LOFT4DMRDataset_Volume(Dataset):
    
    def __init__(self, h5_data, id_time_lst, input_name = "input/data_input", target_name = 'target/data_target', 
                 mask_name = None, preproc_lst = None, ksp_sim_lst = None, slice_axis=-2, time_axis=-1, 
                 global_norm = False, input_shape = (96,96,48)):
        self.data        = h5_data
        self.id_time_lst = id_time_lst
        self.input_name  = input_name
        self.target_name = target_name
        self.mask_name   = mask_name
        self.preproc_lst = preproc_lst
        self.ksp_sim_lst = ksp_sim_lst
        self.slice_axis  = slice_axis
        self.time_axis   = time_axis
        self.global_norm = global_norm
        self.input_shape = input_shape

    def _apply_preproc(self, input_img, target_img, apply_before_ksp):
        """ applies k-space based simulations
        :params: input_img: input 2D numpy matrix
        :params: target_img: target 2D numpy matrix
        :params: curr_apply_before_ksp: bool to determine if we apply objects with or without _apply_before_ksp tag
        :returns: [0] preprocessed input image
        :returns: [1] preprocessed target image
        """
    # apply preprocessing steps after ksp simulations
        for curr_preproc in self.preproc_lst:            
            if curr_preproc._apply_before_ksp == apply_before_ksp:
                input_img, target_img = curr_preproc(input_img, target_img)

        return input_img, target_img
    
    def _apply_ksp_simulations(self, input_img):
        """ applies k-space based simulations
        :params: input_image: input 2D numpy matrix
        :returns: k-space simulation result 2D numpy matrix
        """
        # determine if we can skip
        if not len(self.ksp_sim_lst):
            return input_img

        # apply k-space based simulations
        # convert input data to k-space
        ksp_input_img = fft_im(input_img)

        # apply simulations
        for curr_sim in self.ksp_sim_lst:
            ksp_input_img = curr_sim(ksp_input_img)

        # convert back to image space
        input_img = ifft_im(ksp_input_img)
        input_img = np.real(input_img)
        input_img = np.clip(input_img, 0., None)

        return input_img

    def _process_data(self, curr_id, time_indx):
        """ method to process data
        :params: curr_id: the dataset id key (str) to use
        :params: f_conn: the hdf5 data connection
        :returns: [0] input image
        :returns: [1] target image
        """
        # load initial data
        # make correct keys
        curr_input_key = curr_id + "/" + self.input_name
        curr_target_key = curr_id + "/" + self.target_name
        if self.mask_name is not None:
            curr_mask_key = curr_id + "/" + self.mask_name

        # load data
        if self.slice_axis == -2 and self.time_axis == -1:
            input_img = self.data[curr_input_key][..., time_indx]
            target_img = self.data[curr_target_key][..., time_indx]
            # first normalize data to the defined shape

            input_img = zoom(input_img, (self.input_shape[0]/input_img.shape[0], self.input_shape[1]/input_img.shape[1], self.input_shape[2]/input_img.shape[2]))
            target_img = zoom(target_img, (self.input_shape[0]/target_img.shape[0], self.input_shape[1]/target_img.shape[1], self.input_shape[2]/target_img.shape[2]))

            if self.mask_name is not None:
                mask_img = self.data[curr_mask_key]
            if self.global_norm:
                # global normalization, if used global norm, maybe not using the slice wise normalization again
                input_global_min = np.min(input_img)
                input_global_max = np.max(input_img)
                target_global_min = np.min(target_img)
                target_global_max = np.max(target_img)       
                input_img = (input_img - input_global_min) / (input_global_max - input_global_min)
                target_img = (target_img - target_global_min)/ (target_global_max - target_global_min)                  
        else:
            print("please check data shape of" + curr_input_key)
        
        # apply preprocessing
        if self.mask_name is not None:
            #print(input_img.shape, mask_img.shape)
            mask_img[mask_img>0] = 1 
            input_img = input_img * mask_img
            target_img = target_img * mask_img

        input_img, target_img = self._apply_preproc(input_img, target_img, apply_before_ksp=True)
        
        # apply ksp simulations
        input_img = self._apply_ksp_simulations(input_img)
        
        
        # apply preprocessing
        input_img, target_img = self._apply_preproc(input_img, target_img, apply_before_ksp=False)
        
        
        # add channel and return
        input_img = np.expand_dims(input_img, axis=0)
        target_img = np.expand_dims(target_img, axis=0)

        # convert to float32
        input_img = input_img.astype("float32")
        target_img = target_img.astype("float32")

        # return data
        return input_img, target_img
        
    def __getitem__(self, indx):
        input_img, target_img = self._process_data(*self.id_time_lst[indx]) 
        return input_img, target_img  
    
    def __len__(self):
        return len(self.id_time_lst)
    
    
class MultipleLOFT4DMRDataset_Volume(Dataset):
    """ class for containing multiple datasets
    :params: dataset_lst: list of datasets
    :probs: list of probabilities to set for data
    """
    def __init__(self, dataset_lst, probs=None):
        # save data
        self.dataset_lst = dataset_lst
        self.id_time_lst = [x.id_time_lst for x in dataset_lst]
        self.id_time_lst = [x for y in self.id_time_lst for x in y]
        
        # save probs
        if probs == None:
            self.probs = np.repeat(1/len(dataset_lst), len(dataset_lst))
        elif len(dataset_lst) == len(probs):
            self.probs = np.array(probs) / np.array(probs).sum()
        else:
            raise AssertionError("length of dataset_lst does not equal length of probs.")

    def get_weight_lst(self):
        """ gets list of weights
        """
        # make repeats of weights in list of list and then unlist
        weights_lst = [[x] * len(y) for x, y in zip(self.probs, self.dataset_lst)]
        weights_lst = [x for y in weights_lst for x in y]

        # normalize to 1
        weights_lst = (np.array(weights_lst) / np.array(weights_lst).sum()).tolist()

        return weights_lst

    def __len__(self):
        """ len hook
        """
        len_lst = [len(x) for x in self.dataset_lst]

        return sum(len_lst)

    def __getitem__(self, indx):
        """ getter hook
        """
        # get list of lengths and make cumulative array of indicies
        len_lst = [len(x) for x in self.dataset_lst]
        cum_ary = np.cumsum([0]+len_lst[:-1])
        # find min index
        dat_indx = np.where(indx >= cum_ary)[0].max()
        # reindex by subtracting the cumulative lengths
        new_indx = indx - cum_ary[dat_indx]
        # get data using index of data and of new index
        # this will use each data set's own getter method for returning data
        return self.dataset_lst[dat_indx][new_indx]    

class LOFT4DMRPredDataset_Volume(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, M0_mtx = None, img_shape = (96,96,48), preproc_lst = None):
        # save data
        self.preproc_lst = preproc_lst if preproc_lst is not None else []
        self.img_shape = img_shape
        self.pxl_mtx = input_mtx
        self.M0_mtx = M0_mtx
        # first preprocess, then split channels
        self.pxl_mtx = self._preprocess_data(self.pxl_mtx)            
        self.pxl_mtx = self.pxl_mtx.astype('float32')
        # resize the image to the fixed size
        self.pxl_mtx = zoom(self.pxl_mtx, (self.img_shape[0]/self.pxl_mtx.shape[0], self.img_shape[1]/self.pxl_mtx.shape[1], self.img_shape[2]/self.pxl_mtx.shape[2]))
        self.pxl_mtx = np.expand_dims(self.pxl_mtx, axis=0) # for the channel dimension
        self.pxl_mtx = np.expand_dims(self.pxl_mtx, axis=0) # for the batch dimension 

    def _preprocess_data(self, input_mtx):
        """ preprocesses dicom data
        :params: input_mtx: input numpy matrix
        :returns: applied preprocessing steps
        """
        # apply preprocessing steps
        for curr_preproc in self.preproc_lst:
            input_mtx = curr_preproc(input_mtx)[0]

        return input_mtx


    def __len__(self):
        """ calculates total number of steps
        :returns: length of self.data_lst
        """
        return self.pxl_mtx.shape[0]

    def __getitem__(self, indx):
        """ primary getter method for Dataset
        :params: indx: index to return
        :returns: [0] input image
        """
        # get index of image and then
        input_mtx = self.pxl_mtx[indx]
        return input_mtx,

