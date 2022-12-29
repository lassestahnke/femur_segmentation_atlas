import SimpleITK as sitk
import numpy as np
import os
import matplotlib.pyplot as plt
from registration import est_lin_transf, est_nl_transf

atlas = os.path.join("data", "COMMON_images_masks")
files = os.listdir(atlas)
masks = [fil for fil in files if "mask" in fil]
images = [fil for fil in files if "image" in fil]

# load atlas
fixed_image = images[0]
fixed_mask = masks[0]

atlas_images = images[1:]
atlas_mask = masks[1:]

# read images
fix_img = sitk.ReadImage(os.path.join(atlas, fixed_image), sitk.sitkFloat32, imageIO="NiftiImageIO")
moving_img = sitk.ReadImage(os.path.join(atlas, atlas_images[0]), sitk.sitkFloat32, imageIO="NiftiImageIO")

print("Estimate linear transformation")
OutTx = est_lin_transf(fix_img, moving_img)

print("Resampling images")
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fix_img)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(OutTx)
out = resampler.Execute(moving_img)

print("estimate non-linear transformation")
OutTx_2 = est_nl_transf(fix_img, out)

print("Resampling images")
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fix_img)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(100)
resampler.SetTransform(OutTx_2)
out_2 = resampler.Execute(out)

print("Visualizing registration")
simg1 = sitk.Cast(sitk.RescaleIntensity(fix_img), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out_2), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
sitk.Show(cimg, "ImageRegistration1 Composition")

print('finished')
