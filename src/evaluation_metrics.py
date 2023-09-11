import numpy as np
import cv2
import torch

# make the img_in the same scale as img_ref
def im_scaling(img_in, img_ref, step=4):
    ndim = img_in.ndim
    ishape = img_in.shape
    idtype = img_in.dtype
    
    img_in = img_in.astype(np.float32)
    img_ref = img_ref.astype(np.float32)
    step = round(step) if step >= 1 else 1
    
    idx = np.ones(ishape, dtype=bool)
    for ax in range(ndim):
        tmp = np.zeros((ishape[ax],), dtype=bool)
        tmp[((min(step, ishape[ax])-1)//2)::min(step, ishape[ax])] = True
        idx = idx & np.moveaxis(tmp.reshape([-1]+(ndim-1)*[1]), 0, ax)
    
    b = img_ref[idx].reshape(-1,1)
    A = np.hstack((img_in[idx].reshape(-1,1), np.ones_like(b)))
    fctr = np.linalg.lstsq(A, b, rcond=None)[0]
    return (img_in * fctr[0] + fctr[1]).astype(idtype)

def im_scaling_slice(img_in, img_ref, mask=None):
    
    if mask is None:
        mask = np.ones(img_in.shape)
        
    num_slice = img_in.shape[0]
    scaled_img_in = np.zeros_like(img_in)
    
    for slc in range(num_slice):
        img_ref_slc = img_ref[slc,:,:]
        img_in_slc  = img_in[slc,:,:]
        mask_slc    = mask[slc,:,:]
        
        img_in_slc_vec  = img_in_slc[mask_slc>0]
        img_ref_slc_vec = img_ref_slc[mask_slc>0]
                
        b = img_ref_slc_vec.reshape(-1,1)
        A = np.hstack((img_in_slc_vec.reshape(-1,1), np.ones_like(b)))
        fctr = np.linalg.lstsq(A, b, rcond=None)[0]
        scaled_img_in[slc,:,:] = img_in_slc * fctr[0] + fctr[1]

    return scaled_img_in

# calculate the PSNR of the two images 
def calculate_psnr(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)

    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))

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

# calculate the ssim of two images
def calculate_ssim(img1, img2, crop_border, input_order='HWC', test_y_channel=False):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []

    if img1.ndim==3:
        for i in range(img1.shape[2]):
            ssims.append(_ssim(img1[..., i], img2[..., i]))
    elif img1.ndim==2:
        ssims.append(_ssim(img1, img2))
    return np.array(ssims).mean()

def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

# calculate the ssim for the 3d image slice by slice
def cal_ssim_cases(pred_dset, gt_dset, slice_axis=-1):
    
    pred_data = pred_dset
    gt_data   = gt_dset
        
    #pred_data   = im_scaling(pred_data, gt_data)
    
    min_gt, max_gt = np.amin(gt_data), np.amax(gt_data)
    gt_data   = (gt_data - min_gt) / (max_gt - min_gt)
    pred_data = (pred_data - min_gt) / (max_gt - min_gt)
    pred_data = np.clip(pred_data, 0., 1.)
    
    pred_data = pred_data*255
    gt_data   = gt_data * 255
    
    SSIM_pred_lst = []    

    for i in range(pred_data.shape[slice_axis]):
        if slice_axis==0:
            SSIM_pred_lst.append(calculate_ssim(pred_data[i].squeeze(),gt_data[i].squeeze(),crop_border=0))
        elif slice_axis==-1:
            SSIM_pred_lst.append(calculate_ssim(pred_data[:,:,i].squeeze(),gt_data[:,:,i].squeeze(),crop_border=0))
        
    return np.mean(SSIM_pred_lst)

def cal_psnr_cases(pred_dset, gt_dset, slice_axis=-1):
    
    pred_data = pred_dset
    gt_data   = gt_dset
        
    #pred_data   = im_scaling(pred_data, gt_data)
    
    min_gt, max_gt = np.amin(gt_data), np.amax(gt_data)
    gt_data   = (gt_data - min_gt) / (max_gt - min_gt)
    pred_data = (pred_data - min_gt) / (max_gt - min_gt)
    pred_data = np.clip(pred_data, 0., 1.)
    
    pred_data = pred_data*255
    gt_data   = gt_data * 255
    
    PSNR_pred_lst = []    
    for i in range(pred_data.shape[slice_axis]):
        if slice_axis==0:
            psnr_slice = calculate_psnr(pred_data[i].squeeze(),gt_data[i].squeeze(),crop_border=0)
            if not np.isinf(psnr_slice): 
                PSNR_pred_lst.append(calculate_psnr(pred_data[i].squeeze(),gt_data[i].squeeze(),crop_border=0))
        elif slice_axis==-1:
            psnr_slice = calculate_psnr(pred_data[:,:,i].squeeze(),gt_data[:,:,i].squeeze(),crop_border=0)
            if not np.isinf(psnr_slice):
                PSNR_pred_lst.append(calculate_psnr(pred_data[:,:,i].squeeze(),gt_data[:,:,i].squeeze(),crop_border=0))
    return np.mean(PSNR_pred_lst)