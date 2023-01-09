import SimpleITK as sitk
import os
from assessment import compute_hausdorff, compute_dice
from segmentation import seg_atlas

atlas_masks_base = os.path.join("data", "GROUP_masks")
atlas_images_base = os.path.join("data", "GROUP_images")
save_base_path = os.path.join("data", "atlas_based_segmentations")
im_base_dir = os.path.join("data", "COMMON_images_masks")

files_masks = os.listdir(atlas_masks_base)
files_images = os.listdir(atlas_images_base)
atlas_masks = [os.path.join(atlas_masks_base, fil) for fil in files_masks if ".nii" in fil]
atlas_masks.sort()
atlas_images = [os.path.join(atlas_images_base, fil) for fil in files_images if ".nii" in fil]
atlas_images.sort()

# load image 40 to segment:
image_40 = sitk.ReadImage(os.path.join(im_base_dir, "common_40_image.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")
gt_seg_40 = sitk.ReadImage(os.path.join(im_base_dir, "common_40_mask.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")

# segment image using the atlas
segmentation_40 = seg_atlas(image_40, atlas_images, atlas_masks)
sitk.WriteImage(segmentation_40, os.path.join(save_base_path, "common_40_image.nii.gz"))

# load image 41 to segment:
image_41 = sitk.ReadImage(os.path.join(im_base_dir, "common_41_image.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")
gt_seg_41 = sitk.ReadImage(os.path.join(im_base_dir, "common_41_mask.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")
# segment image
segmentation_41 = seg_atlas(image_41, atlas_images, atlas_masks)
sitk.WriteImage(segmentation_41, os.path.join(save_base_path, "common_41_image.nii.gz"))

# load image 42 to segment:
image_42 = sitk.ReadImage(os.path.join(im_base_dir, "common_42_image.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")
gt_seg_42 = sitk.ReadImage(os.path.join(im_base_dir, "common_42_mask.nii.gz"), sitk.sitkFloat32, imageIO="NiftiImageIO")
# segment image
segmentation_42 = seg_atlas(image_42, atlas_images, atlas_masks)
sitk.WriteImage(segmentation_42, os.path.join(save_base_path, "common_42_image.nii.gz"))
print('finished Registration')

print("start of quality asessment")
# load manually segmented images from Common directory and ground truths
im_base_dir = os.path.join("data", "COMMON_images_masks")
manual_seg_base_dir = os.path.join("data", "manual_segmentation")
common40_gt = sitk.ReadImage(os.path.join(im_base_dir, "common_40_mask.nii.gz"),
                             sitk.sitkFloat32, imageIO="NiftiImageIO")
common41_gt = sitk.ReadImage(os.path.join(im_base_dir, "common_41_mask.nii.gz"),
                             sitk.sitkFloat32, imageIO="NiftiImageIO")
common42_gt = sitk.ReadImage(os.path.join(im_base_dir, "common_42_mask.nii.gz"),
                             sitk.sitkFloat32, imageIO="NiftiImageIO")

# load manual segmentations
common40_man = sitk.ReadImage(os.path.join(manual_seg_base_dir, "common_mask", "common_40_mask.nii.gz"),
                              sitk.sitkFloat32, imageIO="NiftiImageIO")
common41_man = sitk.ReadImage(os.path.join(manual_seg_base_dir, "common_mask", "common_41_mask.nii.gz"),
                              sitk.sitkFloat32, imageIO="NiftiImageIO")
common42_man = sitk.ReadImage(os.path.join(manual_seg_base_dir, "common_mask", "common_42_mask.nii.gz"),
                              sitk.sitkFloat32, imageIO="NiftiImageIO")

# comput metrics
common40_dice_man = compute_dice(common40_gt, common40_man)
common40_hd_man = compute_hausdorff(common40_gt, common40_man)
print("Manual Dice common 40: ", common40_dice_man)
print("Manual HD common 40:", common40_hd_man)

common41_dice_man = compute_dice(common41_gt, common41_man)
common41_hd_man = compute_hausdorff(common41_gt, common41_man)
print("Manual Dice common 41: ", common41_dice_man)
print("Manual HD common 41:", common41_hd_man)

common42_dice_man = compute_dice(common42_gt, common42_man)
common42_hd_man = compute_hausdorff(common42_gt, common42_man)
print("Manual Dice common 42: ", common42_dice_man)
print("Manual HD common 42:", common42_hd_man)

# evaluate atlas based method
common40_dice = compute_dice(segmentation_40, common40_gt)
common40_hd = compute_hausdorff(segmentation_40, common40_gt)
print("common 40: dice of segmentation", common40_dice, "HD:", common40_hd)

common41_dice = compute_dice(segmentation_41, common41_gt)
common41_hd = compute_hausdorff(segmentation_41, common41_gt)
print("common 41: dice of segmentation", common41_dice, "HD:", common41_hd)

common42_dice = compute_dice(segmentation_42, common42_gt)
common42_hd = compute_hausdorff(segmentation_42, common42_gt)
print("common 42: dice of segmentation", common42_dice, "HD:", common42_hd)

# optional visualization
# print("Visualizing Segmentation")
# simg1 = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
# simg2 = sitk.Cast(sitk.RescaleIntensity(segmentation), sitk.sitkUInt8)
# cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
# sitk.Show(cimg, "final segmentation")
