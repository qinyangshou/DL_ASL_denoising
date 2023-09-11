import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

input_data = nib.load('/ifs/loni/groups/loft/qinyang/KWIA_DL_project/Models/test_resize/PWI_01_original.nii').get_fdata()
print(input_data.shape)

target_shape = (96,96,40)
zoomed_img = zoom(input_data,(target_shape[0]/input_data.shape[0], target_shape[1]/input_data.shape[1], target_shape[2]/input_data.shape[2]),mode='nearest')

print(zoomed_img.shape)
img = nib.Nifti1Image(zoomed_img, np.eye(4))
nib.save(img, '/ifs/loni/groups/loft/qinyang/KWIA_DL_project/Models/test_resize/PWI_01_resized_2.nii')

zoomed_img2 = zoom(zoomed_img,(1/(target_shape[0]/input_data.shape[0]), 1/(target_shape[1]/input_data.shape[1]), 1/(target_shape[2]/input_data.shape[2])),mode='nearest')
img = nib.Nifti1Image(zoomed_img2, np.eye(4))
nib.save(img, '/ifs/loni/groups/loft/qinyang/KWIA_DL_project/Models/test_resize/PWI_01_zoomback.nii')

img = nib.Nifti1Image(input_data, np.eye(4))
nib.save(img, '/ifs/loni/groups/loft/qinyang/KWIA_DL_project/Models/test_resize/PWI_01_originsize.nii')