"""
Created on September 11th 2023
@author: Qinyang Shou

"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('./src/')
from image_processing import fft_im, ifft_im


class LOFT4DMRData(object):
    
    def __init__(self, h5_data, split_dict, prop_cull=None, slice_axis=-2, time_axis=-1, remove_spike=True):
        
        self.data = h5_data
        self.split_dict = split_dict
        self.prop_cull = [0., 0.] if prop_cull is None else prop_cull
        self.slice_axis = slice_axis
        self.time_axis = time_axis
        self.remove_spike = remove_spike
        
    def _get_id_slice_time_lst(self, id_lst):
        
        # make range list of time points 
        time_range_lst = [(0, np.array(self.data[x + '/input/data_input']).shape[self.time_axis]) for x in id_lst]       
        # make range list of slices
        slice_range_lst = [(0,np.array( self.data[x + '/input/data_input']).shape[self.slice_axis]) for x in id_lst] #-1
        slice_range_lst = [(
            np.floor(self.prop_cull[0] * x[1]).astype("int"), 
            np.floor(x[1] - self.prop_cull[1] * x[1]).astype("int"), 
        ) for x in slice_range_lst] # cull slices since they may be air

        # make slice lists and unlist nested list
        id_slice_lst = [[(x, y) for y in range(*y)] for x, y in zip(id_lst, slice_range_lst)]
        time_lst = [[(x, z) for z in range(*z)] for x,z in zip(id_slice_lst, time_range_lst)]

        id_slice_time_lst = []
        for id in range(len(id_slice_lst)):
            curr_id = id_lst[id]
            curr_4Dvolume = np.array(self.data[curr_id + '/input/data_input'])
            curr_mask = np.array(self.data[curr_id + '/mask/dset_mask'])
            mean_map   = np.mean(curr_4Dvolume, axis=-1)
            mean_value = np.mean(mean_map[curr_mask>0])
            time_value_lst = []
            for t in range(len(time_lst[id])):
                tmp_volume = curr_4Dvolume[:,:,:,t].squeeze()
                time_value_lst.append(np.mean(tmp_volume[curr_mask>0]))
            std_value = np.std(time_value_lst)
            
            for t in range(len(time_lst[id])):
                if self.remove_spike:
                    if not time_value_lst[t]>mean_value + 2*std_value or time_value_lst[t]<mean_value-2*std_value:
                        for slice in range(len(time_lst[id][t][0])):
                            id_slice_time_lst.append(time_lst[id][t][0][slice] + (t,))
                    #else:
                        #print(curr_id + str(t) + ' removed')
                else:
                    for slice in range(len(time_lst[id][t][0])):
                        id_slice_time_lst.append(time_lst[id][t][0][slice] + (t,))

        # remove input if necessary
        #id_slice_lst = [(x[0].replace("/input", ""), x[1]) for x in id_slice_lst]
        return id_slice_time_lst
    
    def generate_dataset(self, mode, soft_label = None, *args, **kwargs):
        
        assert mode in self.split_dict
        
        id_lst = self.split_dict[mode]
        id_slice_time_lst = self._get_id_slice_time_lst(id_lst)
        
        return LOFT4DMRDataset(self.data, id_slice_time_lst, soft_label = soft_label, slice_axis=self.slice_axis, *args, **kwargs)

    def generate_dataset_multislice(self, mode, soft_label = None, slice_num = 3, *args, **kwargs):
        
        assert mode in self.split_dict
        
        id_lst = self.split_dict[mode]
        id_slice_time_lst = self._get_id_slice_time_lst(id_lst)
        
        return LOFT4DMRDataset_MS(self.data, id_slice_time_lst, soft_label = soft_label, slice_axis=self.slice_axis, slice_num = slice_num, *args, **kwargs)
    
    def generate_dataset_spatiotemporal(self, mode, soft_label = None, slice_num=3, time_frame = 3, *args, **kwargs):
        
        assert mode in self.split_dict
        
        id_lst = self.split_dict[mode]
        id_slice_time_lst = self._get_id_slice_time_lst(id_lst)
        
        return LOFT4DMRDataset_MS(self.data, id_slice_time_lst, soft_label = soft_label, slice_axis=self.slice_axis, slice_num = slice_num, time_frame = time_frame,*args, **kwargs)
        
# For 2D dataset
class LOFT4DMRDataset(Dataset):
    ## h5_data: The hdf5 file that contains input and target data matrics
    ## id_slice_time_lst: generated from the LOFT4DMRData Object that contains separation of slices and time points
    ## input_name: set according to your h5 file keys for the input
    ## target_name: set according to your h5 file keys for the target
    ## mask_name : set if you have brain mask and want to apply to the training data
    ## preproc_lst: preprocessing list to be applied to the training data
    ## ksp_sim_lst: k space based augmentation methods if applicable
    ## slice_axis: the slice axis according to the data organization e.g., if data is [Width Height Slice Time] slice_axis=-2
    ## time_axis:  the time axis according to the data organization e.g., if data is [Width Height Slice Time] time_axis=-1
    ## soft_label: if you have another soft_label to be used as an alternative target
    ## global_norm: globally normalize 3D data to [0,1]
    
    def __init__(self, h5_data, id_slice_time_lst, input_name = "input/data_input", target_name = "target/data_target", mask_name = None, preproc_lst = None, ksp_sim_lst = None, slice_axis = -2, time_axis = -1, soft_label = None, average=1, fixed_average = False, global_norm = True):
        
        self.data = h5_data 
        self.id_slice_time_lst = id_slice_time_lst
        self.input_name = input_name
        self.target_name = target_name
        self.mask_name = mask_name
        self.preproc_lst = preproc_lst
        self.ksp_sim_lst = ksp_sim_lst if ksp_sim_lst is not None else []
        self.slice_axis = slice_axis
        self.time_axis = time_axis
        self.soft_label = soft_label
        self.global_norm = global_norm
        
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
    def _process_data(self, curr_id, slice_indx, time_indx):
        """ method to process data
        :params: curr_id: the dataset id key (str) to use
        :params: slice_indx: the slice index (int) to use
        :params: f_conn: the hdf5 data connection
        :returns: [0] input image
        :returns: [1] target image
        """
        # make correct keys
        curr_input_key = curr_id + "/" + self.input_name
        curr_target_key = curr_id + "/" + self.target_name
        if self.mask_name is not None:
            curr_mask_key = curr_id + "/" + self.mask_name

        # load data
        if self.slice_axis == -2 and self.time_axis == -1:
            input_img = self.data[curr_input_key][..., slice_indx, time_indx]
            target_img = self.data[curr_target_key][..., slice_indx, time_indx]
            if self.mask_name is not None:
                mask_img = self.data[curr_mask_key][..., slice_indx]
            if self.global_norm:
                # global normalization, if used global norm, maybe not using the slice wise normalization again
                input_global_min = np.min(self.data[curr_input_key][...,time_indx])
                input_global_max = np.max(self.data[curr_input_key][...,time_indx])
                target_global_min = np.min(self.data[curr_target_key][...,time_indx])
                target_global_max = np.max(self.data[curr_target_key][...,time_indx])       
                input_img = (input_img - input_global_min) / (input_global_max - input_global_min)
                target_img = (target_img - target_global_min)/ (target_global_max - target_global_min)  
                
        elif self.slice_axis == 0 and self.time_axis== -1:
            input_img = self.data[curr_input_key][slice_indx,...,time_indx]
            target_img = self.data[curr_target_key][slice_indx,...,time_indx]
            if self.mask_name is not None:
                mask_img = self.data[curr_mask_key][slice_indx,...]
            if self.global_norm:
                # global normalization
                input_global_min = np.min(self.data[curr_input_key][...,time_indx])
                input_global_max = np.max(self.data[curr_input_key][...,time_indx])       
                input_img = (input_img - input_global_min) / (input_global_max - input_global_min)
                target_img = (target_img - input_global_min)/ (input_global_max - input_global_min)  
        else:
            print("please check data shape of" + curr_input_key)
            input_img = np.take(self.data[curr_input_key], slice_indx, self.slice_axis)
            target_img = np.take(self.data[curr_target_key], slice_indx, self.slice_axis)
                
        # apply preprocessing
        if self.mask_name is not None:
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
    
    def _process_data_average(self, curr_id, slice_indx, time_indx_lst):
        """ method to process data
        :params: curr_id: the dataset id key (str) to use
        :params: slice_indx: the slice index (int) to use
        :params: f_conn: the hdf5 data connection
        :returns: [0] input image
        :returns: [1] target image
        """
        # make correct keys
        curr_input_key = curr_id + "/" + self.input_name
        curr_target_key = curr_id + "/" + self.target_name
        if self.mask_name is not None:
            curr_mask_key = curr_id + "/" + self.mask_name

        # load data
        if self.slice_axis == -2 and self.time_axis == -1:
            input_img = np.mean(self.data[curr_input_key][..., slice_indx, time_indx_lst],axis=-1)
            target_img = np.mean(self.data[curr_target_key][..., slice_indx, time_indx_lst],axis=-1)
            if self.mask_name is not None:
                mask_img = self.data[curr_mask_key][..., slice_indx]
        elif self.slice_axis == 0 and self.time_axis== -1:
            input_img = np.mean(self.data[curr_input_key][slice_indx,...,time_indx_lst],axis = 0)
            target_img = np.mean(self.data[curr_target_key][slice_indx,...,time_indx_lst], axis = 0)
            if self.mask_name is not None:
                mask_img = self.data[curr_mask_key][slice_indx,...]
        else:
            print("please check data shape of" + curr_input_key)


        # apply preprocessing
        if self.mask_name is not None:
            mask_img[mask_img>0]=1
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
        """ primary getter method for Dataset
        :params: indx: index to return
        :returns: [0] input image
        :returns: [1] target image
        """
        # process data

        input_img, target_img = self._process_data(*self.id_slice_time_lst[indx])
                               
        if self.soft_label is None:
            return input_img, target_img
        else:

            teacher_model = self.soft_label
            for param in teacher_model.parameters():
                param.requires_grad = False
            tensor_input_img = torch.tensor(input_img)
            soft_target = teacher_model.forward(tensor_input_img).numpy()
            return input_img, {'target':target_img,'soft_label':soft_target}
    
    def __len__(self):
        """ calculates total number of steps
        :returns: length of self.data_lst
        """
        if not self.fixed_average:
            return len(self.id_slice_time_lst)
        else:
            return len(self.id_slice_time_lst) // self.average
    
    def get_slice_time_indicies(self, current_sub): 

        df = pd.DataFrame(self.id_slice_time_lst, columns=["key", "slice_indx","time"])
        df = df[df["key"] == current_sub]
        
        split_indices = []
        for t in range(len(df["time"].unique())):
            slice_indices = df.index[df['time'] == t].tolist()
            split_indices.append(slice_indices)

        return split_indices


class LOFT4DMRDataset_MS(Dataset):
    ## h5_data: The hdf5 file that contains input and target data matrics
    ## id_slice_time_lst: generated from the LOFT4DMRData Object that contains separation of slices and time points
    ## input_name: set according to your h5 file keys for the input
    ## target_name: set according to your h5 file keys for the target
    ## mask_name : set if you have brain mask and want to apply to the training data
    ## preproc_lst: preprocessing list to be applied to the training data
    ## ksp_sim_lst: k space based augmentation methods if applicable
    ## slice_axis: the slice axis according to the data organization e.g., if data is [Width Height Slice Time] slice_axis=-2
    ## time_axis:  the time axis according to the data organization e.g., if data is [Width Height Slice Time] time_axis=-1
    ## soft_label: if you have another soft_label to be used as an alternative target
    ## global_norm: globally normalize 3D data to [0,1]
    ## slice_num: number of slices in the pesudo3D input
    ## use_M0: whether to use M0 as another channel in the input
    ## three_dim: organize data for 3D input
    def __init__(self, h5_data, id_slice_time_lst, input_name = "input/data_input", target_name = "target/data_target", mask_name = None, preproc_lst = None, ksp_sim_lst = None, slice_axis = -2, time_axis = -1, soft_label = None, average=1, fixed_average = False, slice_num=3, time_frame = 1, global_norm = False, three_dim = False, use_M0 = False):
        
        self.data = h5_data
        self.id_slice_time_lst = id_slice_time_lst
        self.input_name = input_name
        self.target_name = target_name
        self.mask_name = mask_name 
        self.preproc_lst = preproc_lst
        self.ksp_sim_lst = ksp_sim_lst if ksp_sim_lst is not None else []
        self.slice_axis = slice_axis
        self.time_axis = time_axis   
        self.soft_label = soft_label 
        self.slice_num = slice_num
        self.global_norm = global_norm
        self.three_dim = three_dim
        self.use_M0 = use_M0
            
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
    
    # fixed for the 3D multi-slice case
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
        if input_img.ndim==2:
            ksp_input_img = fft_im(input_img)

            # apply simulations
            for curr_sim in self.ksp_sim_lst:
                ksp_input_img = curr_sim(ksp_input_img)

            input_img = ifft_im(ksp_input_img)
            input_img = np.real(input_img)
            input_img = np.clip(input_img, 0., None)
            return input_img
        elif input_img.ndim==3:
            ksp_input_img = np.zeros_like(input_img)
            for j in range(input_img.shape[0]):
                ksp_input_img[j,:,:] = fft_im(input_img[j,:,:])
                for curr_sim in self.ksp_sim_lst:
                    ksp_input_img[j,:,:] = curr_sim(ksp_input_img[j,:,:])
                input_img[j,:,:]     = ifft_im(ksp_input_img[j,:,:])
            return input_img
     
    def _process_data(self, curr_id, slice_indx, time_indx, num_slice, num_time = 1):
        """ method to process data
        :params: curr_id: the dataset id key (str) to use
        :params: slice_indx: the slice index (int) to use
        :params: f_conn: the hdf5 data connection
        :params: time_indx: can be a list or a single time point
        :returns: [0] input image
        :returns: [1] target image
        """
        # make correct keys
        curr_input_key = curr_id + "/" + self.input_name
        curr_target_key = curr_id + "/" + self.target_name
        curr_M0_key    = curr_id + "/M0/dset_M0" 
        curr_mask_key = curr_id + "/mask/dset_mask"
        if self.mask_name is not None:
            curr_mask_key = curr_id + "/" + self.mask_name
                        
        input_volume = self.data[curr_input_key][...] 
        target_volume =self.data[curr_target_key][...] 
        (size_x, size_y, size_z, size_t) = input_volume.shape       
        M0_volume    = self.data[curr_M0_key][...]
        mask_volume  = self.data[curr_mask_key][...]
        
        input_stacked_img = []
        # specify the slice indices of the neighbouring multislice
        slice_indices = [*range(slice_indx-int((num_slice-1)/2), slice_indx+int((num_slice-1)/2)+1, 1)]
        for slc in slice_indices:            
            if slc>=0 and slc<size_z:
                if isinstance(time_indx, int):
                    input_slice = input_volume[..., slc,time_indx]
                elif isinstance(time_indx, list):
                    input_slice = np.mean(input_volume[..., slc,time_indx], axis=-1)                
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx]
                    input_slice = input_slice * mask_img                    
                input_stacked_img.append(input_slice.squeeze())            
            elif slc<0:
                if isinstance(time_indx, int):
                    input_slice = input_volume[..., 0,time_indx]
                elif isinstance(time_indx, list):
                    input_slice = np.mean(input_volume[..., 0,time_indx], axis=-1)
                if self.mask_name is not None:
                    mask_img = mask_volume[..., 0]
                    input_slice = input_slice * mask_img             
                input_stacked_img.append(input_slice.squeeze())                                
            elif slc>=size_z:
                if isinstance(time_indx, int):
                    input_slice = input_volume[..., size_z-1,time_indx]
                elif isinstance(time_indx, list):
                    input_slice = np.mean(input_volume[..., size_z-1,time_indx], axis=-1)                   
                if self.mask_name is not None:
                    mask_img = mask_volume[..., size_z-1]
                    input_slice = input_slice * mask_img                    
                input_stacked_img.append(input_slice.squeeze())
        
        # specify the time indices for temporal denoising
        time_indices = [*range(time_indx-int((num_time-1)/2), time_indx+int((num_time-1)/2)+1, 1)]
        time_indices.remove(time_indx)
        for t in time_indices:            
            if t>=0 and t<size_t:
                input_slice = input_volume[..., slice_indx,t]           
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx]
                    input_slice = input_slice * mask_img                    
                input_stacked_img.append(input_slice.squeeze())            
            elif t<0:
                input_slice = input_volume[..., slice_indx,0]
                if self.mask_name is not None:
                    mask_img = mask_volume[..., 0]
                    input_slice = input_slice * mask_img             
                input_stacked_img.append(input_slice.squeeze())                                
            elif t>=size_t:
                input_slice = input_volume[..., slice_indx,size_t-1]                 
                if self.mask_name is not None:
                    mask_img = mask_volume[..., size_z-1]
                    input_slice = input_slice * mask_img                    
                input_stacked_img.append(input_slice.squeeze())
        
        # combine all input channels
        input_img = np.stack(input_stacked_img)
        
        if isinstance(time_indx, int):
            target_img = target_volume[..., slice_indx, time_indx]  
        elif isinstance(time_indx, list):
            target_img = np.mean(target_volume[..., slice_indx], axis=-1) 
            
        if self.mask_name is not None:
            mask_img = mask_volume[..., slice_indx]
            target_img = target_img * mask_img
            
        if self.global_norm:
            # global normalization, if used global norm, maybe not using the slice wise normalization again
            input_global_min  = np.min(self.data[curr_input_key][...,time_indx])
            input_global_max  = np.max(self.data[curr_input_key][...,time_indx])      
            input_img = (input_img - input_global_min) / (input_global_max - input_global_min)
            target_img = (target_img - input_global_min)/ (input_global_max - input_global_min)
        
        # include M0 as an additional channel
        if self.use_M0:                        
            M0 = np.array(self.data[curr_M0_key][..., slice_indx])
            M0_min = np.min(self.data[curr_M0_key])
            M0_max = np.max(self.data[curr_M0_key])
            M0_normed = (M0 - M0_min) / (M0_max - M0_min)
            M0_normed = np.expand_dims(M0_normed, axis=0)
            input_img = np.concatenate((input_img, M0_normed), axis=0)
         

        input_img, target_img = self._apply_preproc(input_img, target_img, apply_before_ksp=True)
        
        
        input_img = self._apply_ksp_simulations(input_img)
                      
        if self.three_dim:
            input_img = np.expand_dims(input_img, axis=0) # for 3D dataset
           
        target_img = np.expand_dims(target_img, axis=0)

        # convert to float32
        input_img = input_img.astype("float32")
        target_img = target_img.astype("float32")

        return input_img, target_img
    
    def __getitem__(self, indx):
        """ primary getter method for Dataset
        :params: indx: index to return
        :returns: [0] input image
        :returns: [1] target image
        """
        # process data
        input_img, target_img = self._process_data(*self.id_slice_time_lst[indx], num_slice=self.slice_num)           

        input_img = np.nan_to_num(input_img)
        target_img = np.nan_to_num(target_img)
        
        if self.soft_label is None:
            return input_img, target_img
        else:
            teacher_model = self.soft_label
            for param in teacher_model.parameters():
                param.requires_grad = False
            tensor_input_img = torch.tensor(input_img)
            soft_target = teacher_model.forward(tensor_input_img).numpy()
            return input_img, {'target':target_img,'soft_label':soft_target}
    
    def __len__(self):
        """ calculates total number of steps
        :returns: length of self.data_lst
        """
        return len(self.id_slice_time_lst) 
    
    def get_slice_time_indicies(self, current_sub): 
        df = pd.DataFrame(self.id_slice_time_lst, columns=["key", "slice_indx","time"])
        df = df[df["key"] == current_sub]
        
        split_indices = []
        for t in range(len(df["time"].unique())):
            slice_indices = df.index[df['time'] == t].tolist()
            split_indices.append(slice_indices)

        return split_indices
      
class LOFT4DMRPredDataset(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, preproc_lst = None):
        self.preproc_lst = preproc_lst if preproc_lst is not None else []
        self.pxl_mtx = input_mtx
        self.pxl_mtx = self._preprocess_data(self.pxl_mtx)
        self.pxl_mtx = self.pxl_mtx.astype('float32')


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
        input_mtx = self.pxl_mtx[indx]
        input_mtx = np.expand_dims(input_mtx, 0)
        if input_mtx.ndim==4:
            input_mtx = np.squeeze(input_mtx, axis=0)

        return input_mtx,

#########################################################################################################################    
class LOFT4DMRPredDataset_multislice(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, M0_mtx = None, num_channels = 3, preproc_lst = None):
        self.preproc_lst = preproc_lst if preproc_lst is not None else []
        self.num_channels = num_channels
        self.pxl_mtx = input_mtx
        self.M0_mtx = M0_mtx
        self.pxl_mtx = self._preprocess_data(self.pxl_mtx)            
        self.pxl_mtx = self.pxl_mtx.astype('float32')
          
        size_z = input_mtx.shape[0]
       
        stack_mtx = []
        for slice_indx in range(size_z):
            slice_indices = [*range(slice_indx-int((self.num_channels-1)/2), slice_indx+int((self.num_channels-1)/2)+1, 1)]
            input_stacked_img = []
            for slc in slice_indices:
                
                if slc>=0 and slc<size_z:
                    input_slice = self.pxl_mtx[slc,:,:]              
                    input_stacked_img.append(input_slice.squeeze())
            
                elif slc<0:
                    input_slice = self.pxl_mtx[0, :, :]                    
                    input_stacked_img.append(input_slice.squeeze())                              
                elif slc>=size_z:

                    input_slice = self.pxl_mtx[size_z-1, :,:]                    
                    input_stacked_img.append(input_slice.squeeze())                
            input_stack = np.stack(input_stacked_img) 
            
            if self.M0_mtx is not None:
                M0_slice = M0_mtx[slice_indx, :, :]
                M0_max, M0_min = np.max(M0_slice), np.min(M0_slice)
                M0_slice = (M0_slice - M0_min) / (M0_max - M0_min)
                M0_slice = np.expand_dims(M0_slice, axis=0)
                input_stack = np.concatenate((input_stack, M0_slice), axis=0)
                
            stack_mtx.append(input_stack)
        
        self.pxl_mtx = np.stack(stack_mtx) 
        


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


