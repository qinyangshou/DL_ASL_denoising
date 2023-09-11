# import libraries
import types
import torch
import torchvision

import numpy as np

from torch.nn import functional as F
from pytorch_msssim import MS_SSIM, SSIM, ssim, ms_ssim
from scipy.signal import convolve2d
from torch.nn.functional import l1_loss
from scipy.signal import convolve2d
import cv2

fft  = lambda x, ax : np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=ax), axes=ax, norm='ortho'), axes=ax)
ifft = lambda X, ax : np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(X, axes=ax), axes=ax, norm='ortho'), axes=ax)

class SSIM_Loss(SSIM):
    def forward(self, input, target):
        """ forward hook
        :params: input: torch tensor
        :params: target: torch tensor
        :returns: calculated SSIM loss
        """
        if target.ndim==4:
            return 1 - super(SSIM_Loss, self).forward(input, target)
        elif target.ndim == 5:
            ssim_3d = SSIM(spatial_dims=3, channel=1)
            return 1 - ssim_3d.forward(input, target)
       
class SSIM_Loss_range(SSIM):
    def __init__(self,data_range=1,*args, **kwargs):
        super(SSIM_Loss_range, self).__init__()
        #define Sobel operator
        self.data_range=data_range
    def forward(self, input, target):
        """ forward hook
        :params: input: torch tensor
        :params: target: torch tensor
        :returns: calculated SSIM loss
        """
        return 1 - ssim(input, target, data_range=self.data_range)
    
class Sobel_Loss(torch.nn.Module):
    def __init__(self,*args, **kwargs):
        super(Sobel_Loss, self).__init__()
        #define Sobel operator
        self.Gx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        self.Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    def forward(self, input, target):
        
        loss_x = 0
        loss_y = 0
        for i in range(input.shape[0]):
            loss_x += np.mean(abs((convolve2d(input[i,:,:,:].cpu().detach().numpy().squeeze(), self.Gx, 'same')- convolve2d(target[i,:,:,:].cpu().detach().numpy().squeeze(), self.Gx, 'same'))))
            loss_y += np.mean(abs((convolve2d(input[i,:,:,:].cpu().detach().numpy().squeeze(), self.Gy, 'same')- convolve2d(target[i,:,:,:].cpu().detach().numpy().squeeze(), self.Gy, 'same'))))
        
        loss_x /= input.shape[0]
        loss_y /= input.shape[0]
        return (loss_x + loss_y) / 2
    
class CombinedLoss(torch.nn.Module):
    """ combined losses
    :params: loss_lst: list of losses (each must take x and y arguements)
    :params: weight_lst: list of (normalized) weights for each loss
    """
    def __init__(self, loss_lst, weight_lst=None):
        # initialize super
        super(CombinedLoss, self).__init__()

        # save losses
        self.loss_lst = loss_lst

        # save weights
        if weight_lst == None:
            self.weights = np.repeat(1/len(loss_lst), len(loss_lst))
        elif len(loss_lst) == len(weight_lst):
            self.weights = np.array(weight_lst) / np.array(weight_lst).sum()
        else:
            raise AssertionError("length of dataset_lst does not equal length of probs.")

    def forward(self, input, target):
        """ forward hook
        :params: input: torch tensor
        :params: target: torch tensor
        :returns: weighted loss
        """
        # multiply
        loss_lst = [loss(input, target) * wght for loss, wght in zip(self.loss_lst, self.weights)]

        # return sum
        return sum(loss_lst)

    def get_losses_and_weights(self):
        """ method to return infromation about losses and associated weights
        """
        rslt_dict = {}
        for curr_loss, curr_weight in zip(self.loss_lst, self.weights):
            # if function, get name directly
            if isinstance(curr_loss, types.FunctionType):
                name = curr_loss.__name__

            # otherwise get name through class
            else:
                name = curr_loss.__class__.__name__

            # add to dict
            rslt_dict[name] = curr_weight

        return rslt_dict

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

def sample_weight_ssim(img_input, img_target):
    
    # compare the ssim of the input and target and give a weight of the sampled patch from 0-1
    # maybe also try other weighting methods, need to check the qualities
    # assume the input image patches have been normalized
    img_input = img_input.squeeze()
    img_target = img_target.squeeze()
    img_input = img_input * 255
    img_target = img_target * 255 
    SSIM = _ssim(img_input, img_target)
    
    if SSIM > 0.95:
        patch_sample_weight = 1
    elif SSIM<0.3:
        patch_sample_weight = 0
    else:
        patch_sample_weight = (SSIM - 0.3) / (0.95-0.3)
        
    return patch_sample_weight

def sample_weight_foregroundmean(img_input, img_target, threshold_low=0.005, threshold_high=0.2):
    
    # compare the ssim of the input and target and give a weight of the sampled patch from 0-1
    # maybe also try other weighting methods, need to check the qualities
    # assume the input image patches have been normaliz
    img_input = img_input.squeeze()
    img_target = img_target.squeeze()
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

def combined_sample_weight(img_input, img_target):

    W_ssim = sample_weight_ssim(img_input,img_target)
    W_FGmean = sample_weight_foregroundmean(img_input, img_target)
    combined_weight = np.sqrt(W_ssim * W_FGmean)
    
    return combined_weight

class WeightedCombinedLoss(torch.nn.Module):
    """ combined losses
    :params: loss_lst: list of losses (each must take x and y arguements)
    :params: weight_lst: list of (normalized) weights for each loss
    """
    def __init__(self, loss_lst, weight_lst=None):
        # initialize super
        super(WeightedCombinedLoss, self).__init__()

        # save losses
        self.loss_lst = loss_lst

        # save weights
        if weight_lst == None:
            self.weights = np.repeat(1/len(loss_lst), len(loss_lst))
        elif len(loss_lst) == len(weight_lst):
            self.weights = np.array(weight_lst) / np.array(weight_lst).sum()
        else:
            raise AssertionError("length of dataset_lst does not equal length of probs.")

    def forward(self, input, target, weight=None):
        """ forward hook
        :params: input: torch tensor
        :params: target: torch tensor
        :returns: weighted loss
        """
        if weight==None:
            loss_lst = [loss(input, target) * wght for loss, wght in zip(self.loss_lst, self.weights)]
            # return sum
            return sum(loss_lst)
        else:
            batch_loss_array = [torch.zeros((1,1)).to('cuda')]*len(self.loss_lst)
            # multiply
            for i in range(input.shape[0]): # batch dimension
                sample_input  = torch.unsqueeze(input[i,:,:,:],dim=0)
                sample_target = torch.unsqueeze(target[i,:,:,:],dim=0)
                patch_weight = weight[i]
            
                loss_array = [loss(sample_input, sample_target) * wght * patch_weight for loss, wght in zip(self.loss_lst, self.weights)] 
                batch_loss_array = [sum(value) for value in zip(loss_array, batch_loss_array)]
            # return sum
            return sum(loss_array)

    def get_losses_and_weights(self):
        """ method to return infromation about losses and associated weights
        """
        rslt_dict = {}
        for curr_loss, curr_weight in zip(self.loss_lst, self.weights):
            # if function, get name directly
            if isinstance(curr_loss, types.FunctionType):
                name = curr_loss.__name__

            # otherwise get name through class
            else:
                name = curr_loss.__class__.__name__

            # add to dict
            rslt_dict[name] = curr_weight

        return rslt_dict
    
# perceptual loss
class LPIPSvgg(torch.nn.Module):
    def __init__(self, weight_path):
        # initialize
        super(LPIPSvgg, self).__init__()

        # get
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features

        # define VGG stages
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()

        # set features
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        # turn off requirement for parameters
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [64, 128, 256, 512, 512]
        self.weights = torch.load(weight_path)
        self.weights = list(self.weights.items())

    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        for k in range(len(outs)):
            outs[k] = F.normalize(outs[k])
        return outs

    def forward(self, input, target, as_loss=True):
        """ combined losses
        :params: x: input: torch tensor
        :params: y: target: torch tensor
        :params: as_loss:
        """
        assert x.shape == y.shape
        if as_loss:
            feats0 = self.forward_once(input)
            feats1 = self.forward_once(target)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(input)
                feats1 = self.forward_once(target)
        score = 0.
        for k in range(len(self.chns)):
            score = score + (self.weights[k][1]*(feats0[k]-feats1[k])**2).mean([2,3]).sum(1)
        if as_loss:
            return score.mean()
        else:
            return score
        
class L2pooling(torch.nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTSLoss(torch.nn.Module):
    def __init__(self, weight_path):
        super(DISTSLoss, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", torch.nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", torch.nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if weight_path:
            #weights = torch.load(os.path.join(sys.prefix,'weights.pt'))
            weights = torch.load(weight_path)
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']

    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    #def forward(self, x, y, require_grad=False, batch_average=False):
    def forward(self, x, y, require_grad=True, batch_average=True):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score