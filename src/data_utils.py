import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
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
            np.floor(self.prop_cull[0] * x[1]).astype("int"), # start index
            np.floor(x[1] - self.prop_cull[1] * x[1]).astype("int"), # end index
        ) for x in slice_range_lst] # cull slices since they may be air

        # make slice lists and unlist nested list
        id_slice_lst = [[(x, y) for y in range(*y)] for x, y in zip(id_lst, slice_range_lst)]
        time_lst = [[(x, z) for z in range(*z)] for x,z in zip(id_slice_lst, time_range_lst)]
        #id_slice_time_lst = [x for y in id_slice_time_lst for x in y]

        id_slice_time_lst = []
        for id in range(len(id_slice_lst)):
            curr_id = id_lst[id]
            #print(curr_id)
            curr_4Dvolume = np.array(self.data[curr_id + '/input/data_input'])
            curr_mask = np.array(self.data[curr_id + '/mask/dset_mask'])
            mean_map   = np.mean(curr_4Dvolume, axis=-1)
            mean_value = np.mean(mean_map[curr_mask>0])
            time_value_lst = []
            for t in range(len(time_lst[id])):
                tmp_volume = curr_4Dvolume[:,:,:,t].squeeze()
                time_value_lst.append(np.mean(tmp_volume[curr_mask>0]))
            #print(time_value_lst)
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

    # SQY: to implement
    def generate_dataset_multislice(self, mode, soft_label = None, slice_num = 3, *args, **kwargs):
        
        assert mode in self.split_dict
        
        id_lst = self.split_dict[mode]
        id_slice_time_lst = self._get_id_slice_time_lst(id_lst)
        
        return LOFT4DMRDataset_MS(self.data, id_slice_time_lst, soft_label = soft_label, slice_axis=self.slice_axis, slice_num = slice_num, *args, **kwargs)
    
    # SQY 2/9 
    def generate_dataset_spatiotemporal(self, mode, soft_label = None, slice_num=3, time_frame = 3, *args, **kwargs):
        
        assert mode in self.split_dict
        
        id_lst = self.split_dict[mode]
        id_slice_time_lst = self._get_id_slice_time_lst(id_lst)
        
        return LOFT4DMRDataset_MS(self.data, id_slice_time_lst, soft_label = soft_label, slice_axis=self.slice_axis, slice_num = slice_num, time_frame = time_frame,*args, **kwargs)
        
# For 2D dataset
class LOFT4DMRDataset(Dataset):
    
    def __init__(self, h5_data, id_slice_time_lst, input_name = "input/data_input", target_name = "target/data_target", mask_name = None, preproc_lst = None, ksp_sim_lst = None, slice_axis = -2, time_axis = -1, soft_label = None, average=1, fixed_average = False, global_norm = False):
        
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
        self.average = average
        self.fixed_average = fixed_average
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
        # load initial data
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
            input_img = self.data[curr_input_key][slice_indx,...,time_index]
            target_img = self.data[curr_target_key][slice_indx,...,time_index]
            if self.mask_name is not None:
                mask_img = self.data[curr_mask_key][slice_indx,...]
            if self.global_norm:
                # global normalization
                input_global_min = np.min(self.data[curr_input_key][...,time_indx])
                input_global_max = np.max(self.data[curr_input_key][...,time_indx])
                #target_global_min = np.min(self.data[curr_target_key][...,time_indx])
                #target_global_max = np.max(self.data[curr_target_key][...,time_indx])       
                input_img = (input_img - input_global_min) / (input_global_max - input_global_min)
                target_img = (target_img - input_global_min)/ (input_global_max - input_global_min)  
        else:
            print("please check data shape of" + curr_input_key)
            input_img = np.take(self.data[curr_input_key], slice_indx, self.slice_axis)
            target_img = np.take(self.data[curr_target_key], slice_indx, self.slice_axis)
        

        
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
    
    def _process_data_average(self, curr_id, slice_indx, time_indx_lst):
        """ method to process data
        :params: curr_id: the dataset id key (str) to use
        :params: slice_indx: the slice index (int) to use
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

        # apply simulations and augmentations, currently disabled

        # apply preprocessing
        if self.mask_name is not None:
            #print(input_img.shape, mask_img.shape)
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
        # print(self.id_slice_time_lst)
        if self.average == 1:
            input_img, target_img = self._process_data(*self.id_slice_time_lst[indx])
        else:
            # get the list for average
            current_id, slice_index, time_index = self.id_slice_time_lst[indx]
            # generate the list for average
            total_time_points = self.id_slice_time_lst[-1][-1] + 1 # get the total number for average
            if self.average > total_time_points:
                print("too many average, check the data!")
                return False
            # select the random timepoints from the list, must include the current index
            
            if not self.fixed_average:
                time_range_lst = [*range(total_time_points)]
                time_range_lst.remove(time_index)
                random_time_lst = list(np.random.choice(time_range_lst,self.average-1,replace=False))
                random_time_lst.append(time_index)
                random_time_lst.sort()
                input_img, target_img = self._process_data_average(current_id, slice_index, random_time_lst)
            else:
                time_range_lst = [*range(total_time_points)]
                res = len(time_range_lst) - len(time_range_lst) % self.average                
                time_range_lst = time_range_lst[0:res]
                fixed_time_lst = time_range_lst[time_index::self.average]
                input_img, target_img = self._process_data_average(current_id, slice_index, fixed_time_lst)
                
                
        if self.soft_label is None:
            return input_img, target_img
        else:
            # use the model pred, # will change to preloaded soft_labels in the future
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
    
    def get_slice_time_indicies(self, current_sub): ##sqy used for prediction
        # make into df and caluclate max index
        # make into df
        df = pd.DataFrame(self.id_slice_time_lst, columns=["key", "slice_indx","time"])
        df = df[df["key"] == current_sub]
        
        split_indices = []
        for t in range(len(df["time"].unique())):
            slice_indices = df.index[df['time'] == t].tolist()
            split_indices.append(slice_indices)

        return split_indices

###### SQY to be implemented #####################   #####################   #####################   #####################       
class LOFT4DMRDataset_MS(Dataset):
    
    def __init__(self, h5_data, id_slice_time_lst, input_name = "input/data_input", target_name = "target/data_target", mask_name = None, preproc_lst = None, ksp_sim_lst = None, slice_axis = -2, time_axis = -1, soft_label = None, average=1, fixed_average = False, slice_num=3, time_frame = 1, global_norm = False, three_dim = False, use_M0 = False):
        
        self.data = h5_data
        self.id_slice_time_lst = id_slice_time_lst
        self.input_name = input_name
        self.target_name = target_name
        self.mask_name = mask_name #
        self.preproc_lst = preproc_lst
        self.ksp_sim_lst = ksp_sim_lst if ksp_sim_lst is not None else []
        self.slice_axis = slice_axis
        self.time_axis = time_axis   
        self.soft_label = soft_label #
        self.average = average # 
        self.fixed_average = fixed_average #
        self.slice_num = slice_num
        self.time_frame = time_frame ## sqy
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

            # convert back to image space
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
        # load initial data
        # make correct keys
        curr_input_key = curr_id + "/" + self.input_name
        curr_target_key = curr_id + "/" + self.target_name
        curr_M0_key    = curr_id + "/M0/dset_M0" # use the M0 included hdf5 data
        curr_mask_key = curr_id + "/mask/dset_mask"
        if self.mask_name is not None:
            curr_mask_key = curr_id + "/" + self.mask_name
                        
        input_volume = self.data[curr_input_key][...] # 4D volume
        target_volume =self.data[curr_target_key][...] # 4D volume
        (size_x, size_y, size_z, size_t) = input_volume.shape       
        M0_volume    = self.data[curr_M0_key][...]
        mask_volume  = self.data[curr_mask_key][...]
        
        input_stacked_img = []
        # specify the slice indices of the neighbouring multislice
        slice_indices = [*range(slice_indx-int((num_slice-1)/2), slice_indx+int((num_slice-1)/2)+1, 1)]
        #print("slice_indices: ", slice_indices)
        for slc in slice_indices:            
            if slc>=0 and slc<size_z:
                # average for the time frames for the case of random average
                if isinstance(time_indx, int):
                    input_slice = input_volume[..., slc,time_indx]
                elif isinstance(time_indx, list):
                    input_slice = np.mean(input_volume[..., slc,time_indx], axis=-1)                
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx]
                    input_slice = input_slice * mask_img                    
                input_stacked_img.append(input_slice.squeeze())            
            elif slc<0:
                # currently use same padding
                if isinstance(time_indx, int):
                    input_slice = input_volume[..., 0,time_indx]
                elif isinstance(time_indx, list):
                    input_slice = np.mean(input_volume[..., 0,time_indx], axis=-1)
                if self.mask_name is not None:
                    mask_img = mask_volume[..., 0]
                    input_slice = input_slice * mask_img             
                input_stacked_img.append(input_slice.squeeze())                                
            elif slc>=size_z:
                # currently use same padding
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
        # exclude current time index (duplicated)
        time_indices.remove(time_indx)
        for t in time_indices:            
            if t>=0 and t<size_t:
                input_slice = input_volume[..., slice_indx,t]           
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx]
                    input_slice = input_slice * mask_img                    
                input_stacked_img.append(input_slice.squeeze())            
            elif t<0:
                # currently use same padding
                input_slice = input_volume[..., slice_indx,0]
                if self.mask_name is not None:
                    mask_img = mask_volume[..., 0]
                    input_slice = input_slice * mask_img             
                input_stacked_img.append(input_slice.squeeze())                                
            elif t>=size_t:
                # currently use same padding
                input_slice = input_volume[..., slice_indx,size_t-1]                 
                if self.mask_name is not None:
                    mask_img = mask_volume[..., size_z-1]
                    input_slice = input_slice * mask_img                    
                input_stacked_img.append(input_slice.squeeze())
        
        # combine all input channels
        input_img = np.stack(input_stacked_img)
        
        # this part may need to think more carefully if want to use other type of labels instead of average
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
            #target_global_min = np.min(self.data[curr_target_key][...,time_indx])
            #target_global_max = np.max(self.data[curr_target_key][...,time_indx])       
            input_img = (input_img - input_global_min) / (input_global_max - input_global_min)
            target_img = (target_img - input_global_min)/ (input_global_max - input_global_min)
        
        # include M0 as an additional channel
        if self.use_M0:                        
            M0 = np.array(self.data[curr_M0_key][..., slice_indx])
            M0_min = np.min(self.data[curr_M0_key])
            M0_max = np.max(self.data[curr_M0_key])
            M0_normed = (M0 - M0_min) / (M0_max - M0_min)
            M0_normed = np.expand_dims(M0_normed, axis=0)
            # input is num_slice by x by y, M0 is 1 by x by y
            input_img = np.concatenate((input_img, M0_normed), axis=0)
         

        input_img, target_img = self._apply_preproc(input_img, target_img, apply_before_ksp=True)
        
        # apply ksp simulations
        
        input_img = self._apply_ksp_simulations(input_img)
              
        # apply preprocessing
        #input_img, target_img = self._apply_preproc(input_img, target_img, apply_before_ksp=False)
        
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
        # print(self.id_slice_time_lst)
        if isinstance(self.average, int):
            if self.average == 1:
                input_img, target_img = self._process_data(*self.id_slice_time_lst[indx], num_slice=self.slice_num, num_time = self.time_frame)           

        input_img = np.nan_to_num(input_img)
        target_img = np.nan_to_num(target_img)
        
        if self.soft_label is None:
            return input_img, target_img
        else:
            # use the model pred, # will change to preloaded soft_labels in the future
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
    
    def get_slice_time_indicies(self, current_sub): ##sqy used for prediction
        # make into df and caluclate max index
        # make into df
        df = pd.DataFrame(self.id_slice_time_lst, columns=["key", "slice_indx","time"])
        df = df[df["key"] == current_sub]
        
        split_indices = []
        for t in range(len(df["time"].unique())):
            slice_indices = df.index[df['time'] == t].tolist()
            split_indices.append(slice_indices)

        return split_indices

#############################################################
#### SQY 2/11
# For 3D dataset of size 
class LOFT4DMRData_Multidelay(object):
    
    def __init__(self, h5_data, split_dict, prop_cull=None, slice_axis=-3, time_axis=-2, num_PLD = 1, remove_spike=True):
        
        self.data = h5_data
        self.split_dict = split_dict
        self.prop_cull = [0., 0.] if prop_cull is None else prop_cull
        self.slice_axis = slice_axis
        self.time_axis = time_axis
        self.remove_spike = remove_spike
        self.num_PLD = num_PLD
        
    def _get_id_slice_time_lst(self, id_lst):
        
        # make range list of time points 
        time_range_lst = [(0, np.array(self.data[x + '/input/data_input']).shape[self.time_axis]) for x in id_lst]         
        PLD_range_lst = [(0,self.num_PLD-1)]
        
        # make range list of slices
        slice_range_lst = [(0,np.array( self.data[x + '/input/data_input']).shape[self.slice_axis]) for x in id_lst] #-1
        
        # remove top and bottom slices
        slice_range_lst = [(
            np.floor(self.prop_cull[0] * x[1]).astype("int"), # start index
            np.floor(x[1] - self.prop_cull[1] * x[1]).astype("int"), # end index
        ) for x in slice_range_lst] # cull slices since they may be air

        # make slice lists and unlist nested list
        id_slice_lst = [[(x, y) for y in range(*y)] for x, y in zip(id_lst, slice_range_lst)]
        time_lst = [[(x, z) for z in range(*z)] for x,z in zip(id_slice_lst, time_range_lst)]
        #id_slice_time_lst = [x for y in id_slice_time_lst for x in y]

        id_slice_time_lst = []
        removed_index = []
        subject_id_global_mean = []
        
        # removing the spike time points
        for id in range(len(id_slice_lst)):
            curr_id = id_lst[id]
            #print(curr_id)
            for pld in range(self.num_PLD):
                curr_5Dvolume = np.array(self.data[curr_id + '/input/data_input'])
                curr_PLD_4Dvolume = curr_5Dvolume[:,:,:,:,pld]
                curr_mask_all = np.array(self.data[curr_id + '/mask/dset_mask'])
                curr_PLD_mask = curr_mask_all[:,:,:,pld]
                
                time_value_lst = []
                useful_timepoints = curr_5Dvolume.shape[3]
                for t in range(len(time_lst[id])):
                    tmp_volume = curr_PLD_4Dvolume[:,:,:,t].squeeze()
                    if not np.sum(tmp_volume) == 0:
                        time_value_lst.append(np.mean(tmp_volume[curr_PLD_mask>0]))
                    else:
                        useful_timepoints -= 1
                               
                mean_value = np.mean(time_value_lst)
                std_value = np.std(time_value_lst)
            
                for t in range(useful_timepoints):
                    if self.remove_spike:
                        if not time_value_lst[t]>mean_value + 2*std_value or time_value_lst[t]<mean_value-2*std_value:
                            for slice in range(len(time_lst[id][t][0])):
                                id_slice_time_lst.append(time_lst[id][t][0][slice] + (t,pld,))
                        else:
                            print(curr_id + str(t) + ' removed')
                            # record the removed timepoints and that PLD for future use
                            removed_index.append((curr_id,t,pld))
                    else:
                        for slice in range(len(time_lst[id][t][0])):
                            id_slice_time_lst.append(time_lst[id][t][0][slice] + (t,pld,))

        # remove input if necessary
        #id_slice_lst = [(x[0].replace("/input", ""), x[1]) for x in id_slice_lst]
        return id_slice_time_lst, removed_index
    
    # SQY 2/11
    def generate_dataset_MD(self, mode, soft_label = None, *args, **kwargs):
        
        assert mode in self.split_dict        
        id_lst = self.split_dict[mode]
        id_slice_time_lst, removed_index = self._get_id_slice_time_lst(id_lst)             
        return LOFT4DMRDataset_Multi_Delay(self.data, id_slice_time_lst, removed_index, soft_label = soft_label, slice_axis=self.slice_axis,  *args, **kwargs)
    
class LOFT4DMRDataset_Multi_Delay(Dataset):
    
    def __init__(self, h5_data, id_slice_time_lst, removed_index, input_name = "input/data_input", target_name = "target/data_target", 
                 mask_name = None, preproc_lst = None, ksp_sim_lst = None, slice_axis = -2, time_axis = -1, 
                 soft_label = None, average=1, fixed_average = False, slice_num=3, PLD_constraint = 1, global_norm = False, 
                 three_dim = False, use_M0 = False, PLD_array = [9,9,9,9,3], padding_weight=0.1):
        
        self.data = h5_data
        self.id_slice_time_lst = id_slice_time_lst
        shortened_lst = []
        for item in self.id_slice_time_lst:
            pld = item[-1]
            if not item[-2]>=PLD_array[pld]:
                shortened_lst.append(item)
        
        self.id_slice_time_lst = shortened_lst
        self.removed_index = removed_index
        self.input_name = input_name
        self.target_name = target_name
        self.mask_name = mask_name #
        self.preproc_lst = preproc_lst
        self.ksp_sim_lst = ksp_sim_lst if ksp_sim_lst is not None else []
        self.slice_axis = slice_axis
        self.time_axis = time_axis   
        self.soft_label = soft_label
        self.average = average 
        self.fixed_average = fixed_average 
        self.slice_num = slice_num
        self.PLD_constraint = PLD_constraint ## used as input channels
        self.global_norm = global_norm
        self.three_dim = three_dim
        self.use_M0 = use_M0
        self.PLD_array = PLD_array
        self.padding_weight = padding_weight

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

            # convert back to image space
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
     
    def _process_data(self, curr_id, slice_indx, time_indx, pld_indx, num_slice, num_pld, num_average = 1, fixed_average = False):
        """ method to process data
        :params: curr_id: the dataset id key (str) to use
        :params: slice_indx: the slice index (int) to use
        :params: f_conn: the hdf5 data connection
        :params: time_indx: can be a list or a single time point
        :returns: [0] input image
        :returns: [1] target image
        """
        # load initial data
        # make correct keys
        #mark1 = time.time()
                
        curr_input_key = curr_id + "/" + self.input_name
        curr_target_key = curr_id + "/" + self.target_name
        curr_M0_key    = curr_id + "/M0/dset_M0" # use the M0 included hdf5 data
        curr_mask_key = curr_id + "/mask/dset_mask"
        if self.mask_name is not None:
            curr_mask_key = curr_id + "/" + self.mask_name
             
        input_volume = self.data[curr_input_key][...] # 5D volume         
        # currently hard code here as the slice index is axis=-2
        (size_x, size_y, size_z, size_t, size_pld) = input_volume.shape
                   
        #target_volume = self.data[curr_target_key][...] # 5D volume
        mask_volume = self.data[curr_mask_key][...]
        #mask_volume[mask_volume>0]=1
              
        rand_t_list = []
        
        for p in range(size_pld):
            # random choose N timepoints
            flag = False     
            while not flag:
                count = 0
                t_list = list(np.random.choice(range(self.PLD_array[p]),num_average))
                for t in t_list:
                    if (curr_id, t, p) in self.removed_index: # detect whether it's in the spike, otherwise redo the sampling
                        count += 1                
                if count ==0:
                    flag = True
            
            if p == pld_indx:
                if not time_indx in t_list:
                    t_list[-1] = time_indx #must include the current time index
                
            rand_t_list.append(t_list)
       
        #mark3 = time.time()
        #print("block 2 time: ", mark3 - mark2)
        
        input_stacked_img = []
        # specify the slice indices of the neighbouring multislice
        slice_indices = [*range(slice_indx-int((num_slice-1)/2), slice_indx+int((num_slice-1)/2)+1, 1)]
        #print("slice_indices: ", slice_indices)
        t_list = rand_t_list[pld_indx]
        
        for slc in slice_indices:            
            if slc>=0 and slc<size_z:
                # average for the time frames for the case of random average
                if len(t_list) == 1:
                    #input_slice = input_volume[..., slc, t_list, pld_indx]
                    #input_slice = self.data[curr_input_key][..., slc, t_list, pld_indx]    
                    input_stacked_img.append(input_volume[..., slc, t_list, pld_indx]) 
                elif len(t_list) > 1:
                    input_slice = np.mean(self.data[curr_input_key][..., slc,t_list, pld_indx], axis=-1)                
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx, pld_indx]
                    input_slice = input_slice * mask_img                               
            elif slc<0:
                # currently use same padding
                if len(t_list) == 1:
                    #input_slice = self.data[curr_input_key][..., 0,t_list, pld_indx]
                    input_stacked_img.append(input_volume[..., 0,t_list, pld_indx]) 
                elif len(t_list) > 1:
                    input_slice = np.mean(self.data[curr_input_key][..., 0,t_list, pld_indx], axis=-1)
                if self.mask_name is not None:
                    mask_img = mask_volume[..., 0, pld_indx]
                    input_slice = input_slice * mask_img             
                #input_stacked_img.append(input_slice.squeeze())                                
            elif slc>=size_z:
                # currently use same padding
                if len(t_list) == 1:
                    #input_slice = self.data[curr_input_key][..., size_z-1,t_list, pld_indx]
                    input_stacked_img.append(input_volume[..., size_z-1,t_list, pld_indx])
                elif len(t_list) > 1:
                    input_slice = np.mean(self.data[curr_input_key][..., size_z-1,t_list, pld_indx], axis=-1)                   
                if self.mask_name is not None:
                    mask_img = mask_volume[..., size_z-1, pld_indx]
                    input_slice = input_slice * mask_img                    
                #input_stacked_img.append(input_slice.squeeze())
        
        #mark4 = time.time()
        #print("block 3 time: ", mark4 - mark3)
        # specify the time indices for temporal denoising
        
        pld_indices = [*range(pld_indx-int((num_pld-1)/2), pld_indx+int((num_pld-1)/2)+1, 1)]
        #print("time_indices: ", time_indices)
        # exclude current time index (duplicated)
        pld_indices.remove(pld_indx)
        
        for pld in pld_indices:            
            if pld>=0 and pld<size_pld:
                t = rand_t_list[pld]
                if(len(t)>1):
                    input_slice = np.mean(input_volume[..., slice_indx,t,pld], axis = -1)
                else:
                    #input_slice = self.data[curr_input_key][..., slice_indx, t, pld]
                    input_stacked_img.append(input_volume[..., slice_indx, t, pld])
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx,pld]
                    input_slice = input_slice * mask_img                    
                #input_stacked_img.append(input_slice.squeeze())            
            elif pld<0:
                # currently use same padding
                t = rand_t_list[0]
                if(len(t)>1):
                    input_slice = np.mean(input_volume[..., slice_indx,t,0], axis = -1)
                else:
                    #input_slice = self.data[curr_input_key][..., slice_indx, t, 0]
                    input_stacked_img.append(input_volume[..., slice_indx, t, 0] * self.padding_weight)
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx, 0]
                    input_slice = input_slice * mask_img             
                #input_stacked_img.append(input_slice.squeeze())                                
            elif pld>=size_pld-1:
                # currently use same padding
                t = rand_t_list[size_pld-1]
                if(len(t)>1):
                    input_slice = np.mean(input_volume[..., slice_indx,t,size_pld-1], axis = -1)
                else:
                    #input_slice = self.data[curr_input_key][..., slice_indx, t, size_pld-1]
                    input_stacked_img.append(input_volume[..., slice_indx, t, size_pld-1] * self.padding_weight)
                if self.mask_name is not None:
                    mask_img = mask_volume[..., slice_indx, size_pld-1]
                    input_slice = input_slice * mask_img                    
                #input_stacked_img.append(input_slice.squeeze())
        
        # combine all input channels
        input_img = np.stack(input_stacked_img)
        input_img = input_img.squeeze()        
        # this part may need to think more carefully if want to use other type of labels instead of average
        target_img = self.data[curr_target_key][..., slice_indx, time_indx, pld_indx]         
        
        if self.mask_name is not None:
            mask_img = mask_volume[..., slice_indx]
            target_img = target_img * mask_img            
                
        if self.global_norm:
            # global normalization, if used global norm, maybe not using the slice wise normalization again
            input_global_min  = np.min(input_volume[...,time_indx, pld_indx])
            input_global_max  = np.max(input_volume[...,time_indx, pld_indx])
            #target_global_min = np.min(self.data[curr_target_key][...,time_indx, pld_indx])
            #target_global_max = np.max(self.data[curr_target_key][...,time_indx, pld_indx])       
            input_img = (input_img - input_global_min) / (input_global_max - input_global_min)
            target_img = (target_img - input_global_min)/ (input_global_max - input_global_min)
               
        # include M0 as an additional channel
        if self.use_M0:                        
            M0 = np.array(self.data[curr_M0_key][..., slice_indx])
            M0_min = np.min(self.data[curr_M0_key])
            M0_max = np.max(self.data[curr_M0_key])
            M0_normed = (M0 - M0_min) / (M0_max - M0_min)
            M0_normed = np.expand_dims(M0_normed, axis=0)
            # input is num_slice by x by y, M0 is 1 by x by y
            input_img = np.concatenate((input_img, M0_normed), axis=0)
                        
        input_img, target_img = self._apply_preproc(input_img, target_img, apply_before_ksp=True)
        # apply ksp simulations
        input_img = self._apply_ksp_simulations(input_img)
        
        # apply preprocessing
        input_img, target_img = self._apply_preproc(input_img, target_img, apply_before_ksp=False)        
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
        # print(self.id_slice_time_lst)

        input_img, target_img = self._process_data(*self.id_slice_time_lst[indx], num_slice=self.slice_num, num_pld = self.PLD_constraint, num_average = self.average, fixed_average = self.fixed_average)   
        input_img = np.nan_to_num(input_img)
        target_img = np.nan_to_num(target_img)
        #print(indx)
        if self.soft_label is None:
            return input_img, target_img
        else:
            # use the model pred, # will change to preloaded soft_labels in the future
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
    
    def get_slice_time_indicies(self, current_sub): ##sqy used for prediction
        # make into df and caluclate max index
        # make into df
        df = pd.DataFrame(self.id_slice_time_lst, columns=["key", "slice_indx","time_indx",'pld_indx'])
        df = df[df["key"] == current_sub]
                
        split_indices_all = []
        for pld in range(len(df["pld_indx"].unique())):
            split_indices_PLD = []
            tmp = df[df['pld_indx']==pld]
            for t in range(len(df["time_indx"].unique())):
                slice_indices = tmp.index[tmp['time_indx'] == t].tolist()
                split_indices_PLD.append(slice_indices)
            
            split_indices_all.append(split_indices_PLD)

        return split_indices_all    
##########################################   #####################   #####################   #####################      
# class LOFT4DMRDataset_3D(Dataset):
# SQY to be implemented

class MultipleLOFT4DMRDataset(Dataset):
    """ class for containing multiple datasets
    :params: dataset_lst: list of datasets
    :probs: list of probabilities to set for data
    """
    def __init__(self, dataset_lst, probs=None):
        # save data
        self.dataset_lst = dataset_lst

        # set id_slice_lst, may need to check it
        self.id_slice_time_lst = [x.id_slice_time_lst for x in dataset_lst]
        self.id_slice_time_lst = [x for y in self.id_slice_time_lst for x in y]


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

    def get_slice_time_indicies(self, current_sub):
        """ method for using 'get_slice_indicies' of the child dataset_lst objects
        """
        # get list of slice inidicies
        slice_time_indicies_lst = [x.get_slice_time_indicies(current_sub) for x in self.dataset_lst]

        # initialize with first component
        rslt_slice_time_inidices = []
        cummulate_num = 0
        for i in range(len(slice_time_indicies_lst)):
            if slice_time_indicies_lst[i] == []:
                cummulate_num += len(self.dataset_lst[i])
                #print(cummulate_num)
            else:
                for sub_list in slice_time_indicies_lst[i]:
                    rslt_slice_time_inidices.append([x + cummulate_num for x in sub_list])
        # if we have more than one item, we can iterate by adding 2nd and up
        # with index with max number
        '''
        if len(slice_time_indicies_lst) > 1:
            
            for curr_slice_time_indicies in slice_time_indicies_lst[1:]:
                rslt_slice_time_inidices = np.concatenate([
                    rslt_slice_time_inidices,
                    curr_slice_time_indicies + rslt_slice_time_inidices[-1][-1]],
                )
        '''
        return rslt_slice_time_inidices

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
    
    
class LOFT4DMRPredDataset(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, preproc_lst = None):
        # save data
        self.preproc_lst = preproc_lst if preproc_lst is not None else []

        # save dcm info
        self.pxl_mtx = input_mtx
        # save pxl matrix info
        # move to numpy matrix and apply preprocessing

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
        # get index of image and then
        input_mtx = self.pxl_mtx[indx]

        # expand dims for channel
        input_mtx = np.expand_dims(input_mtx, 0)
        #print(input_mtx.shape)
        if input_mtx.ndim==4:
            input_mtx = np.squeeze(input_mtx, axis=0)
        # must return tuple
        return input_mtx,

#########################################################################################################################    
class LOFT4DMRPredDataset_multislice(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, M0_mtx = None, num_channels = 3, preproc_lst = None):
        # save data
        self.preproc_lst = preproc_lst if preproc_lst is not None else []
        self.num_channels = num_channels
        # save dcm info
        self.pxl_mtx = input_mtx
        self.M0_mtx = M0_mtx
        # first preprocess, then split channels
        self.pxl_mtx = self._preprocess_data(self.pxl_mtx)            
        self.pxl_mtx = self.pxl_mtx.astype('float32')
        
        #print("in dataset init")
        #print(np.min(self.pxl_mtx), np.max(self.pxl_mtx))
        # should arrange the matrix as num_slice * num_channels * img_x * img_y       
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
                    # currently use same padding
                    input_slice = self.pxl_mtx[0, :, :]                    
                    input_stacked_img.append(input_slice.squeeze())                              
                elif slc>=size_z:
                    # currently use same padding
                    input_slice = self.pxl_mtx[size_z-1, :,:]                    
                    input_stacked_img.append(input_slice.squeeze())                
            input_stack = np.stack(input_stacked_img) # size of num_channels * img_x * img_y
            
            if self.M0_mtx is not None:
                M0_slice = M0_mtx[slice_indx, :, :]
                M0_max, M0_min = np.max(M0_slice), np.min(M0_slice)
                M0_slice = (M0_slice - M0_min) / (M0_max - M0_min)
                M0_slice = np.expand_dims(M0_slice, axis=0)
                input_stack = np.concatenate((input_stack, M0_slice), axis=0)
                
            stack_mtx.append(input_stack)
        
        self.pxl_mtx = np.stack(stack_mtx) # size of num_slice * num_channels * img_x * img_y
        # move to numpy matrix and apply preprocessing
        


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

        # expand dims for channel
        #input_mtx = np.expand_dims(input_mtx, 0)

        # must return tuple
        #print(np.min(input_mtx), np.max(input_mtx))
        return input_mtx,

#########################################################################################################################
class LOFT4DMRPredDataset_spatiotemporal(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, M0_mtx = None, num_slice = 3, num_time =3, curr_time = 0, preproc_lst = None):
        # save data
        self.preproc_lst = preproc_lst if preproc_lst is not None else []
        self.num_slice = num_slice
        self.num_time  = num_time
        self.curr_time = curr_time
        # save dcm info
        self.pxl_mtx = input_mtx
        self.M0_mtx = M0_mtx
        # first preprocess, then split channels
        self.pxl_mtx = self._preprocess_data(self.pxl_mtx)            
        self.pxl_mtx = self.pxl_mtx.astype('float32')
        
        #print("in dataset init")
        #print(np.min(self.pxl_mtx), np.max(self.pxl_mtx))
        # should arrange the matrix as num_slice * num_channels * img_x * img_y       
        size_z = input_mtx.shape[-2]
        size_t = input_mtx.shape[-1]
        
        stack_mtx = []
        for slice_indx in range(size_z):
            slice_indices = [*range(slice_indx-int((self.num_slice-1)/2), slice_indx+int((self.num_slice-1)/2)+1, 1)]
            input_stacked_img = []
            for slc in slice_indices:
                
                if slc>=0 and slc<size_z:
                    input_slice = self.pxl_mtx[:,:,slc,curr_time]              
                    input_stacked_img.append(input_slice.squeeze())
            
                elif slc<0:
                    # currently use same padding
                    input_slice = self.pxl_mtx[:, :, 0, curr_time]                    
                    input_stacked_img.append(input_slice.squeeze())                              
                elif slc>=size_z:
                    # currently use same padding
                    input_slice = self.pxl_mtx[:,:,size_z-1,curr_time]                    
                    input_stacked_img.append(input_slice.squeeze())                
                       
            time_indices = [*range(curr_time-int((num_time-1)/2), curr_time+int((num_time-1)/2)+1, 1)]
            time_indices.remove(self.curr_time)
            for t in time_indices:            
                if t>=0 and t<size_t:
                    input_slice = self.pxl_mtx[..., slc,t]                             
                    input_stacked_img.append(input_slice.squeeze())            
                elif t<0:
                    # currently use same padding
                    input_slice = self.pxl_mtx[..., slc,0]           
                    input_stacked_img.append(input_slice.squeeze())                                
                elif t>=size_t:
                    # currently use same padding
                    input_slice = self.pxl_mtx[..., slc,size_t-1]                                  
                    input_stacked_img.append(input_slice.squeeze())
            
            input_stack = np.stack(input_stacked_img, axis=-1) # size of num_channels * img_x * img_y
            input_stack = np.transpose(input_stack,(2,0,1))
            #print(input_stack.shape)
            if self.M0_mtx is not None:
                M0_slice = M0_mtx[slice_indx, :, :]
                M0_max, M0_min = np.max(M0_slice), np.min(M0_slice)
                M0_slice = (M0_slice - M0_min) / (M0_max - M0_min)
                M0_slice = np.expand_dims(M0_slice, axis=0)
                input_stack = np.concatenate((input_stack, M0_slice), axis=0)
                
            stack_mtx.append(input_stack)
            
        
        self.pxl_mtx = np.stack(stack_mtx) # size of num_slice * num_channels * img_x * img_y
        # move to numpy matrix and apply preprocessing
        
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

        # expand dims for channel
        #input_mtx = np.expand_dims(input_mtx, 0)

        # must return tuple
        #print(np.min(input_mtx), np.max(input_mtx))
        return input_mtx,

######################################################################################################################
class LOFT4DMRPredDataset_Multidelay(Dataset):
    """ Dataset class for dicom (used primarly for inference)
    :params: dicom_path: directory which contains dicom files
    """
    def __init__(self, input_mtx, M0_mtx = None, num_slice = 3, num_PLD = 3, curr_PLD = 0, padding_weight=1, preproc_lst = None):
        # save data
        self.preproc_lst = preproc_lst if preproc_lst is not None else []
        self.num_slice = num_slice
        self.num_PLD  = num_PLD
        self.curr_PLD = curr_PLD
        # save dcm info
        self.pxl_mtx = input_mtx
        self.M0_mtx = M0_mtx
        # first preprocess, then split channels
        self.pxl_mtx = self._preprocess_data(self.pxl_mtx)            
        self.pxl_mtx = self.pxl_mtx.astype('float32')
        self.padding_weight = padding_weight

        # here the matrix size is x y z size_PLD    
        size_z = input_mtx.shape[-2]
        size_t = input_mtx.shape[-1]
        
        stack_mtx = []
        for slice_indx in range(size_z):
            slice_indices = [*range(slice_indx-int((self.num_slice-1)/2), slice_indx+int((self.num_slice-1)/2)+1, 1)]
            input_stacked_img = []
            for slc in slice_indices:
                
                if slc>=0 and slc<size_z:
                    input_slice = self.pxl_mtx[:,:,slc,curr_PLD]              
                    input_stacked_img.append(input_slice.squeeze())
            
                elif slc<0:
                    # currently use same padding
                    input_slice = self.pxl_mtx[:, :, 0, curr_PLD]                    
                    input_stacked_img.append(input_slice.squeeze())                              
                elif slc>=size_z:
                    # currently use same padding
                    input_slice = self.pxl_mtx[:,:,size_z-1,curr_PLD]                    
                    input_stacked_img.append(input_slice.squeeze())                
                       
            time_indices = [*range(curr_PLD-int((num_PLD-1)/2), curr_PLD+int((num_PLD-1)/2)+1, 1)]
            time_indices.remove(self.curr_PLD)
            for t in time_indices:            
                if t>=0 and t<size_t:
                    input_slice = self.pxl_mtx[..., slc,t]                             
                    input_stacked_img.append(input_slice.squeeze())            
                elif t<0:
                    # currently use same padding
                    input_slice = self.pxl_mtx[..., slc,0] * self.padding_weight           
                    input_stacked_img.append(input_slice.squeeze())                                
                elif t>=size_t:
                    # currently use same padding
                    input_slice = self.pxl_mtx[..., slc,size_t-1] * self.padding_weight                                  
                    input_stacked_img.append(input_slice.squeeze())
            
            input_stack = np.stack(input_stacked_img, axis=-1) # size of num_channels * img_x * img_y
            input_stack = np.transpose(input_stack,(2,0,1))
            #print(input_stack.shape)
            if self.M0_mtx is not None:
                M0_slice = M0_mtx[slice_indx, :, :]
                M0_max, M0_min = np.max(M0_slice), np.min(M0_slice)
                M0_slice = (M0_slice - M0_min) / (M0_max - M0_min)
                M0_slice = np.expand_dims(M0_slice, axis=0)
                input_stack = np.concatenate((input_stack, M0_slice), axis=0)
                
            stack_mtx.append(input_stack)
            
        
        self.pxl_mtx = np.stack(stack_mtx) # size of num_slice * num_channels * img_x * img_y
        # move to numpy matrix and apply preprocessing
        
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

        # expand dims for channel
        #input_mtx = np.expand_dims(input_mtx, 0)

        # must return tuple
        #print(np.min(input_mtx), np.max(input_mtx))
        return input_mtx,

def avg_for_each_PLD(input_mtx, PLD_array):
    
    # function for average of the input across time points for each PLD
    # input_mtx should be arranged in the shape of x y z t num_PLD
    num_PLD = input_mtx.shape[-1]
    output_mtx = np.zeros((input_mtx.shape[0], input_mtx.shape[1], input_mtx.shape[2], num_PLD))
    for p in range(num_PLD):
        print(PLD_array[p])
        output_mtx[:,:,:,p] = np.mean(input_mtx[:,:,:,0:PLD_array[p],p],axis=3)
    
    return output_mtx