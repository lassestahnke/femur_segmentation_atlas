import SimpleITK as sitk
import os
from segmentation import seg_atlas

atlas_masks_base = os.path.join("data", "GROUP_masks")
atlas_images_base = os.path.join("data", "GROUP_images")

files_masks = os.listdir(atlas_masks_base)
files_images = os.listdir(atlas_images_base)
atlas_masks = [os.path.join(atlas_masks_base, fil) for fil in files_masks if ".nii" in fil]
atlas_masks.sort()
atlas_images = [os.path.join(atlas_images_base, fil) for fil in files_images if ".nii" in fil]
atlas_images.sort()

# load image to segment:
im_base_dir = os.path.join("data", "COMMON_images_masks")
image = sitk.ReadImage(os.path.join(im_base_dir, "common_40_image.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")

# segment image using the atlas
segmentation = seg_atlas(image, atlas_images, atlas_masks)

# load ground truth segmentation
#gt_seg = sitk.ReadImage(os.path.join(im_base_dir, "common_40_mask.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")

print("Visualizing Segmentation")
simg1 = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(segmentation), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
sitk.Show(segmentation, "final segmentation")

print('finished')
