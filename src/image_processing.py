import math
import finufft
import numpy as np

from skimage.transform import rotate
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, convolve
from scipy.signal import convolve2d
from sklearn.feature_extraction import image
import cv2

def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# fourier and inverse fourier transforms
def fft_im(image):
    """ Fourier transform of a 2/3D image to k-space
    :param image: 2/3D numpy array in image domain
    :return: output: 2/3D numpy array in k-space domain
    """
    dim = range(image.ndim)

    if len(image.shape) == 2:
        axes = (0, 1)
    elif len(image.shape) == 3:
        axes = (1, 2)

    output = (np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(image, axes=axes), axes=axes), axes=axes)).astype(np.complex128)
    output /= np.sqrt(np.prod(np.take(output.shape, dim)))

    return output

def ifft_im(ksp_dat):
    """ Fourier transform of a k-space data to image
    :param image: 2D numpy array in k-space domain
    :return: output: converted numpy image
    """
    output = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ksp_dat), norm="ortho"))

    return output

def sample_weight_ssim(img_input, img_target):
    
    # compare the ssim of the input and target and give a weight of the sampled patch from 0-1
    # maybe also try other weighting methods, need to check the qualities
    # assume the input image patches have been normaliz
    img_input = img_input * 255
    img_target = img_target * 255 
    SSIM = _ssim(img_input, img_target)
    print(SSIM)
    
    if SSIM > 0.95:
        patch_sample_weight = 1
    elif SSIM<0.3:
        patch_sample_weight = 0
    else:
        patch_sample_weight = (SSIM - 0.3) / (0.95-0.3)
        
    return patch_sample_weight

def sample_weight_foregroundmean(img_input, img_target, threshold_low=0.005, threshold_high=0.1):
    
    # compare the ssim of the input and target and give a weight of the sampled patch from 0-1
    # maybe also try other weighting methods, need to check the qualities
    # assume the input image patches have been normaliz
    FGmean_img1 = np.mean(img_input[img_input>0])
    FGmean_img2 = np.mean(img_target[img_target>0])
    dif = (FGmean_img1-FGmean_img2)/FGmean_img2
    
    if dif > threshold_high:
        patch_sample_weight = 0
    elif dif<threshold_low:
        patch_sample_weight = 1
    else:
        patch_sample_weight = (dif - threshold_high) / (threshold_low - threshold_high)
        
    return patch_sample_weight
class BasePreprocess(object):
    def __init__(self, apply_before_ksp=True, *args, **kwargs):
        # apply super method
        super(BasePreprocess, self).__init__(*args, **kwargs)

        # set apply
        self._apply_before_ksp = apply_before_ksp

class ClipData(BasePreprocess):
    """ classs to contain matrix preprocessing steps
    :params: clip_values: percentile of values to clip; setting to None skips clipping
    :params: max_normalization: normalize max values to this; setting to None skips normalization
    """
    def __init__(self, clip_values = 1., *args, **kwargs):
        # apply super method
        super(ClipData, self).__init__(*args, **kwargs)

        # save information
        self.clip_values = clip_values

    def _clip(self, input_mtx):
        """ clips data
        :params: input_mtx: input numpy matrix
        :returns: clipped input matrix
        """
        rslt_mtx = np.clip(
            input_mtx,
            a_min=np.percentile(input_mtx, self.clip_values),
            a_max=np.percentile(input_mtx, 100 - self.clip_values),
        )
        a_min=np.percentile(input_mtx, self.clip_values),
        a_max=np.percentile(input_mtx, 100 - self.clip_values),
        #print(a_min, a_max)
        return rslt_mtx

    def __call__(self, *args):
        """ applies preprocess steps
        :params: *args: list of numpy array dataset
        :returns: tuple of preprocessed images
        """
        # assert all are numpy matrixies
        assert all([type(x) == np.ndarray for x in args])

        # if all don't have values above zero, return
        if not all([x.max() > 0 for x in args]):
            return args

        # apply normalization if it's been defined
        if self.clip_values is not None:
            args = [self._clip(x) for x in args]

        return args

class NormalizeData(BasePreprocess):
    """ classs to contain matrix preprocessing steps
    :params: clip_values: percentile of values to clip; setting to None skips clipping
    :params: max_normalization: normalize max values to this; setting to None skips normalization
    """
    def __init__(self, max_normalization = 1., *args, **kwargs):
        # apply super method
        super(NormalizeData, self).__init__(*args, **kwargs)

        # save information
        self.max_normalization = max_normalization

    def normalize(self, input_mtx):
        """ normalizes data between 0 and 1
        :params: input_mtx: input numpy matrix
        :returns: normalized input matrix
        """
        rslt_mtx = (input_mtx - np.min(input_mtx)) / (np.max(input_mtx) - np.min(input_mtx))
        rslt_mtx *= self.max_normalization

        return rslt_mtx

    def __call__(self, *args):
        """ applies normalization to values
        :params: *args: list of numpy array dataset
        :returns: tuple of preprocessed images
        """
        # assert all are numpy matrixies
        assert all([type(x) == np.ndarray for x in args])

        # if all don't have values above zero, return
        if not all([x.max() > 0 for x in args]):
            return args

        # apply normalization if it's been defined
        if self.max_normalization is not None:
            args = [self.normalize(x) for x in args]

        return args

class Zscore_NormalizeData(BasePreprocess):
    """ classs to contain matrix preprocessing steps
    :params: clip_values: percentile of values to clip; setting to None skips clipping
    :params: max_normalization: normalize max values to this; setting to None skips normalization
    """
    def __init__(self, *args, **kwargs):
        # apply super method
        super(Zscore_NormalizeData, self).__init__(*args, **kwargs)

    def normalize(self, input_mtx):
        """ normalizes data between 0 and 1
        :params: input_mtx: input numpy matrix
        :returns: normalized input matrix
        """
        rslt_mtx = (input_mtx - np.mean(input_mtx)) / np.std(input_mtx)

        return rslt_mtx

    def __call__(self, *args):
        """ applies normalization to values
        :params: *args: list of numpy array dataset
        :returns: tuple of preprocessed images
        """
        # assert all are numpy matrixies
        assert all([type(x) == np.ndarray for x in args])

        # if all don't have values above zero, return
        if not all([x.max() > 0 for x in args]):
            return args

        args = [self.normalize(x) for x in args]
        return args  
    
class NormalizePostProcess(NormalizeData):
    """ inherients fro overwrites normalize method to save min and max values
    __call__ is the same to preserve functionality
    """
    def __init__(self, use_M0 = False, *args, **kwargs):
        super(NormalizePostProcess, self).__init__(*args, **kwargs)
        self.use_M0 = use_M0
        
    def normalize(self, input_mtx):
        """ normalizes data between 0 and 1
        :params: input_mtx: input numpy matrix
        :returns: normalized input matrix
        """
        # set min and max values
        if not self.use_M0:
            self.min_value = input_mtx.min()
            self.max_value = input_mtx.max()

            rslt_mtx = (input_mtx - self.min_value) / (self.max_value - self.min_value)
            rslt_mtx *= self.max_normalization
        else:
            self.min_value = input_mtx[0:-1,:,:].min() # M0 channel is always the last channel
            self.max_value = input_mtx[0:-1,:,:].max()
            
            rslt_mtx = np.zeros_like(input_mtx)
            rslt_mtx[0:-1,:,:] = (input_mtx[0:-1,:,:]- self.min_value) / (self.max_value - self.min_value)
            rslt_mtx[-1,:,:] = input_mtx[-1,:,:]
        
        #print(self.min_value, self.max_value)
        return rslt_mtx

    def post_process(self, input_mtx, *args, **kwargs):
        """ unnormalizes values
        :params: input_mtx: input numpy matrix
        :returns: matrix in data scale rage prior to normalization
        """
        # remove max_norm scaling
        rslt_mtx = input_mtx/self.max_normalization

        # get back range scaling
        rslt_mtx *= (self.max_value - self.min_value)
        
        # add back baseline
        rslt_mtx += self.min_value
        
        return rslt_mtx

class NormalizePostProcessChannel(NormalizeData): 
    # used for multi-delay spatiotemporal case
    # input_mtx should be in the shape of x y z PLD
    """ inherients fro overwrites normalize method to save min and max values
    __call__ is the same to preserve functionality
    """
    def __init__(self, use_M0 = False, main_channel=1,*args, **kwargs):
        super(NormalizePostProcessChannel, self).__init__(*args, **kwargs)
        self.main_channel = main_channel
        
    def normalize(self, input_mtx):
        """ normalizes data between 0 and 1
        :params: input_mtx: input numpy matrix
        :returns: normalized input matrix
        """
        # set min and max values

        self.min_value = input_mtx[:,:,:,self.main_channel].min()
        self.max_value = input_mtx[:,:,:,self.main_channel].max()

        rslt_mtx = (input_mtx - self.min_value) / (self.max_value - self.min_value)
        rslt_mtx *= self.max_normalization
        
        #print(self.min_value, self.max_value)
        return rslt_mtx

    def post_process(self, input_mtx, *args, **kwargs):
        """ unnormalizes values
        :params: input_mtx: input numpy matrix
        :returns: matrix in data scale rage prior to normalization
        """
        # remove max_norm scaling
        rslt_mtx = input_mtx/self.max_normalization

        # get back range scaling
        rslt_mtx *= (self.max_value - self.min_value)
        
        # add back baseline
        rslt_mtx += self.min_value
        
        return rslt_mtx
class ClipDataSlice(BasePreprocess):
    """ classs to contain matrix preprocessing steps
    :params: clip_values: percentile of values to clip; setting to None skips clipping
    :params: max_normalization: normalize max values to this; setting to None skips normalization
    """
    def __init__(self, clip_values = 1., *args, **kwargs):
        # apply super method
        super(ClipDataSlice, self).__init__(*args, **kwargs)

        # save information
        self.clip_values = clip_values

    def _clip(self, input_mtx_3d):
        """ clips data
        :params: input_mtx: input numpy matrix
        :returns: clipped input matrix
        """
        num_slice = input_mtx_3d.shape[0]
        rslt_mtx = input_mtx_3d
        
        for i in range(num_slice):
            input_mtx = input_mtx_3d[i,:,:]
            rslt_mtx[i,:,:] = np.clip(
                input_mtx,
                a_min=np.percentile(input_mtx, self.clip_values),
                a_max=np.percentile(input_mtx, 100 - self.clip_values),
            )

        return rslt_mtx

    def __call__(self, *args):
        """ applies preprocess steps
        :params: *args: list of numpy array dataset
        :returns: tuple of preprocessed images
        """
        # assert all are numpy matrixies
        assert all([type(x) == np.ndarray for x in args])

        # if all don't have values above zero, return
        if not all([x.max() > 0 for x in args]):
            return args

        # apply normalization if it's been defined
        if self.clip_values is not None:
            args = [self._clip(x) for x in args]

        return args

class NormalizePostProcessSlice(NormalizeData):
    """ inherients fro overwrites normalize method to save min and max values
    __call__ is the same to preserve functionality
    """
    def __init__(self, *args, **kwargs):
        super(NormalizePostProcessSlice, self).__init__(*args, **kwargs)

    def normalize(self, input_mtx):
        """ normalizes data between 0 and 1 for each slice and save the max, min values for each slice
        :params: input_mtx: input numpy matrix
        :returns: normalized input matrix
        """
        # set min and max values
        rslt_mtx = np.zeros_like(input_mtx)
        self.min_values = []
        self.max_values = []
        for i in range(input_mtx.shape[0]):
            
            min_value = input_mtx[i].min()
            max_value = input_mtx[i].max()
            
            self.min_values.append(min_value)
            self.max_values.append(max_value)

            rslt_mtx[i] = (input_mtx[i] - min_value) / (max_value - min_value)
            rslt_mtx[i] *= self.max_normalization
        
        return rslt_mtx

    def post_process(self, input_mtx, *args, **kwargs):
        """ unnormalizes values
        :params: input_mtx: input numpy matrix
        :returns: matrix in data scale rage prior to normalization
        """
        # remove max_norm scaling
        rslt_mtx = np.zeros_like(input_mtx)
        
        for i in range(input_mtx.shape[0]):
            rslt_mtx[i] = input_mtx[i]/self.max_normalization
            # get back range scaling
            rslt_mtx[i] *= (self.max_values[i] - self.min_values[i])
            # add back baseline
            rslt_mtx[i] += self.min_values[i]
        
        return rslt_mtx    
    
class PatchPairedImage(BasePreprocess):
    """ extracts a (paired) patch from both input and target images
    :params: patch_size: tuple of image size (ints) to extract
    :params: max_attempts: int of number of max attempts before giving up and returning a random patch
    :params: min_patch_value: min signal intensity threshold
    """
    def __init__(self, patch_size, max_attempts=80, min_patch_value=0.05, apply_before_ksp=False, *args, **kwargs):
        # apply super method
        super(PatchPairedImage, self).__init__(apply_before_ksp=apply_before_ksp, *args, **kwargs)

        # save data
        self.patch_size = patch_size
        self.max_attempts = max_attempts
        self.min_patch_value = min_patch_value
        
    def __call__(self, input_img, target_img):
        """ method to patch images
        :params: input_img: the input image
        :params: target_img: the target image
        :returns: [0] input image
        :returns: [1] target image
        """
        # loop until we fine valid patch
        found_patch = False
        curr_attempt = 0
        while not found_patch:
            # increment our attempts
            curr_attempt += 1

            # get a random int
            rand_int = np.random.randint(9999)
            
            if(input_img.ndim==2):
                # extract patches
                input_patch = image.extract_patches_2d(input_img, patch_size=self.patch_size, random_state=rand_int, max_patches=1)[0]
                target_patch = image.extract_patches_2d(target_img, patch_size=self.patch_size, random_state=rand_int, max_patches=1)[0]

                # get patch mean
                patch_mean = (input_patch.mean() + target_patch.mean())/2

                # determine if we need to escape
                if patch_mean > self.min_patch_value:
                    found_patch = True
                elif curr_attempt > self.max_attempts:
                    found_patch = True
            
            elif(input_img.ndim==3):
                stack_patch = []
                for i in range(input_img.shape[0]):
                    input_patch_slc = image.extract_patches_2d(input_img[i,:,:], patch_size=self.patch_size, random_state=rand_int, max_patches=1)[0]
                    stack_patch.append(input_patch_slc)
                    
                input_patch = np.stack(stack_patch)
                target_patch = image.extract_patches_2d(target_img, patch_size=self.patch_size, random_state=rand_int, max_patches=1)[0]
                # get patch mean
                center = int((input_patch.shape[0]-1)/2)
                patch_mean = (input_patch[center,:,:].mean() + target_patch.mean())/2

                # determine if we need to escape
                if patch_mean > self.min_patch_value:
                    found_patch = True
                elif curr_attempt > self.max_attempts:
                    found_patch = True
        
        #print("patch found")
        return input_patch, target_patch

class ScaledImage(BasePreprocess):
    """ classs to contain matrix preprocessing steps
    :params: clip_values: percentile of values to clip; setting to None skips clipping
    :params: max_normalization: normalize max values to this; setting to None skips normalization
    """
    def __init__(self, *args, **kwargs):
        # apply super method
        super(ScaledImage, self).__init__(*args, **kwargs)

    def scaling(self, input_mtx, gt_mtx):
        """ scale the input_mtx to the gt_mtx
        :params: input_mtx: input numpy matrix, gt_mtx: reference numpy matrix
        :returns: scaled input matrix and the gt_mtx
        """
        if input_mtx.ndim == 3:
            center = int((input_mtx.shape[0]-1)/2) 
            input_center = input_mtx[center,:,:]
            
            b = gt_mtx.reshape(-1,1)
            A = np.hstack((input_center.reshape(-1,1), np.ones_like(b)))
            fctr = np.linalg.lstsq(A, b, rcond=None)[0]
        
        else:
            b = gt_mtx.reshape(-1,1)
            A = np.hstack((input_mtx.reshape(-1,1), np.ones_like(b)))
            fctr = np.linalg.lstsq(A, b, rcond=None)[0]
            
        return (input_mtx * fctr[0] + fctr[1]), gt_mtx


    def __call__(self, input_mtx, gt_mtx):
        """ applies scaling
        :returns: tuple of preprocessed images
        """
        return self.scaling(input_mtx, gt_mtx)

class PaddingImage(BasePreprocess):
    """ extracts a (paired) patch from both input and target images
    :params: patch_size: tuple of image size (ints) to extract
    :params: max_attempts: int of number of max attempts before giving up and returning a random patch
    :params: min_patch_value: min signal intensity threshold
    """
    def __init__(self, pad_target_size, pad_value = 0, apply_before_ksp=False, *args, **kwargs):
        # apply super method
        super(PaddingImage, self).__init__(apply_before_ksp=apply_before_ksp, *args, **kwargs)

        # save data
        self.pad_target_size = pad_target_size
        self.pad_value = pad_value

    def __call__(self, input_img, target_img):
        """ method to patch images
        :params: input_img: the input image
        :params: target_img: the target image
        :returns: [0] input image
        :returns: [1] target image
        """
        # loop until we fine valid patch
        if not input_img.shape==self.pad_target_size:
            pad_width0 = (self.pad_target_size[0]-input_img.shape[0])//2
            pad_width1 = (self.pad_target_size[1]-input_img.shape[1])//2
            padded_input = np.pad(input_img,pad_width=((pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values=self.pad_value)    
            padded_target = np.pad(target_img,pad_width=((pad_width0,pad_width0),(pad_width1,pad_width1)),mode='constant',constant_values = self.pad_value)
        
            return padded_input, padded_target
        else:
            return input_img, target_img
    
class Augmentation(object):
    """ base class for augmentation
    :params: prob_apply: probablity of applying augmentation
    """
    def __init__(self, prob_apply=0.5, *args, **kwargs):
        super(Augmentation, self).__init__(*args, **kwargs)
        self.prob_apply = prob_apply

    def _determine_apply_augmentation(self):
        """ determine if we randomly apply augmentation
        :returns: bool if we apply augmentation
        """
        
        return bool(np.random.binomial(1, self.prob_apply))

class AugmentRandomRotation(Augmentation, BasePreprocess):
    """ randomly rotates image
    :params: augmentation_range: tuple of negative and positive floats to randomly sample from
    """
    def __init__(self, augmentation_range, *args, **kwargs):
        # super method
        # prob_apply is used through here
        super(AugmentRandomRotation, self).__init__(*args, **kwargs)

        # save data
        self.augmentation_range = augmentation_range

    def __call__(self, input_x, input_y):
        """
        :params: input_x: x 2D numpy array dataset
        :params: input_y: y 2D numpy array dataset
        :returns: randomly rotated images (if we apply)
        """
        # determine if we randomly rotate
        if self._determine_apply_augmentation():
            # randomly create rotation angle within augmentation range
            random_degree = np.random.uniform(*self.augmentation_range)

            # rotate current images
            if input_x.ndim==2 and input_y.ndim==2:
                input_x = rotate(input_x, random_degree)
                input_y = rotate(input_y, random_degree)
            elif input_x.ndim==3 and input_y.ndim==2: # pseudo3d case is arranged in the shape of slice_channels, x, y
                stack_x = [rotate(input_x[s],random_degree) for s in range(input_x.shape[0])]
                input_x = np.stack(stack_x)
                input_y = rotate(input_y, random_degree)
            elif input_x.ndim==3 and input_y.ndim==3: #3D case is arranged in the shape of x,y,z
                stack_x = [rotate(input_x[:,:,s],random_degree) for s in range(input_x.shape[2])]
                input_x = np.stack(stack_x, axis=-1)
                stack_y = [rotate(input_y[:,:,s],random_degree) for s in range(input_y.shape[2])]
                input_y = np.stack(stack_y, axis=-1)


        return input_x, input_y


class AugmentHorizonalFlip(Augmentation, BasePreprocess):
    """ randomly flips image hornizonally
    :params: probablity of applying augmentation
    """
    def __init__(self, *args, **kwargs):
        # super method
        # prob_apply is used through here
        super(AugmentHorizonalFlip, self).__init__(*args, **kwargs)

    def __call__(self, input_x, input_y):
        """
        :params: input_x: x 2D numpy array dataset
        :params: input_y: y 2D numpy array dataset
        :returns: randomly horizontally flipped images (if we apply)
        """
        if self._determine_apply_augmentation():
            # should be the same for both 2D and 3D
            if input_x.ndim==2 and input_y.ndim==2:
                input_x = input_x[:, ::-1]
                input_y = input_y[:, ::-1]
            elif input_x.ndim==3 and input_y.ndim==2:
                input_x = input_x[:,:,::-1]
                input_y = input_y[:,  ::-1]
            else: # both input and target are 3d
                if bool(np.random.binomial(1, self.prob_apply)):
                    input_x = input_x[:,::-1,:]
                    input_y = input_y[:,::-1,:]
                else:
                    input_x = input_x[::-1,:,:]
                    input_y = input_y[::-1,:,:]


        return input_x, input_y

# k-space augmentations
class AddGaussianKspNoise(Augmentation):
    """ class to add Gaussian K-space Noise
    :params: target_img_snr_range: the tuple target image SNRs we sample from
    :param: noise_mu: mean of Gaussian normal distribution
    :param: noise_sigma: std of Gaussian normal distribution
    """
    def __init__(self, target_img_snr_range, noise_mu=0, noise_sigma=1, *args, **kwargs):
        # super method
        # prob_apply is used through here
        super(AddGaussianKspNoise, self).__init__(*args, **kwargs)

        # save data
        self.target_img_snr_range = target_img_snr_range
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma

    def _make_noise(self, ksp_data, snr):
        """ applies applies random k-space noise
        :params: ksp_data: input k-space image (either 2D or 3D)
        :params: snr: target signal-to-noise ratio, if none, skips simulation
        :returns: noised k-space data
        """
        # add dimension if necessary
        if len(ksp_data.shape) == 2:
            ksp_data = np.expand_dims(ksp_data, 0)

        # initialize k-space data
        noise_simu_ksp = np.zeros(ksp_data.shape, dtype=np.complex128)

        # iterate over slices
        for i, ksp in enumerate(ksp_data):
            # calculate standard deviation of the noise matrix
            n_ksp_sigma = np.linalg.norm(ksp) / np.sqrt(ksp.shape[0] * ksp.shape[1]) / snr / np.sqrt(2)

            # generate random noise matrix follow standard normal distribution
            noise_r = np.random.normal(self.noise_mu, self.noise_sigma, (ksp.shape[0] * ksp.shape[1]))
            noise_r = noise_r.reshape(ksp.shape[0], ksp.shape[1])
            noise_c = np.random.normal(self.noise_mu, self.noise_sigma, (ksp.shape[0] * ksp.shape[1]))
            noise_c = noise_c.reshape(ksp.shape[0], ksp.shape[1])

            # generate noise simulated k-space
            simulated_ksp = ksp + n_ksp_sigma * (noise_r + noise_c * 1j)

            # save to matrix
            noise_simu_ksp[i, :, :] = simulated_ksp

        # squeeze if necessary
        noise_simu_ksp = np.squeeze(noise_simu_ksp)

        return noise_simu_ksp

    def __call__(self, ksp_data):
        """ call hook
        :params: ksp_data: input k-space data (either 2D or 3D)
        :returns: noised k-space data (if we apply)
        """
        if self._determine_apply_augmentation():
            # sample range
            target_snr = np.random.uniform(*self.target_img_snr_range)

            # make new data
            ksp_data = self._make_noise(ksp_data, target_snr)

        return ksp_data

# sqy define for histogram based normalization
def HistogramNormalization(input_img, target_img, mask=None,
                          i_min = 1, i_max=99, l_percentile = 10, u_percentile=90, step=10):
    
    # match the histogram of the input image to the target to remove the bias
    percs = np.concatenate(([i_min], np.arange(l_percentile, u_percentile, step),[i_max]))
    # [1,10,20,30,40,50,60,70,80,90,99]
    input_img_masked = input_img[mask>0]
    target_img_masked = target_img[mask>0]
    
    target_landmarks = [np.amin(target_img)] + list(np.percentile(target_img_masked, percs)) + [np.amax(target_img)]
    input_landmarks  = [np.amin(input_img)] + list(np.percentile(input_img_masked, percs))+  [np.amax(input_img)]
    f = interp1d(input_landmarks, target_landmarks)
    input_img = np.clip(input_img,a_min = np.percentile(input_img,i_min), a_max = np.percentile(input_img,i_max))
    # apply transformation to the input_img
    return f(input_img)