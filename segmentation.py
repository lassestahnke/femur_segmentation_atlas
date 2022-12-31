import SimpleITK as sitk
from registration import est_lin_transf, est_nl_transf
from transformation import apply_transf

def seg_atlas(im, atlas_ct_list, atlas_seg_list):
    """
    Apply atlas-based segmentation of `im` using the list of CT images in `atlas_ct_list` and the corresponding
    segmentation masks in `atlas_seg_list`. Return the resulting segmentation mask after majority voting.
    """

    atlas_segmentations = [] # list to store segmentations

    # register all images in atlas to im and transform their segmentation masks
    print(len(atlas_ct_list), "atlas images found")

    for i in range(len(atlas_ct_list)):
        print("Register image", i+1, " out of ", len(atlas_ct_list))
        # read image and mask
        im_atlas = sitk.ReadImage(atlas_ct_list[i], sitk.sitkFloat32, imageIO="NiftiImageIO")
        mask_atlas = sitk.ReadImage(atlas_seg_list[i], sitk.sitkFloat32, imageIO="NiftiImageIO")

        # estimate linear transform
        print("Register image", i + 1, " out of ", len(atlas_ct_list), "... Estimate linear registration")
        transf_lin = est_lin_transf(im, im_atlas)

        # apply estimate on moving image
        im_atlas_lin = apply_transf(im, im_atlas, transf_lin)

        # estimate non-linear transform
        print("Register image", i + 1, " out of ", len(atlas_ct_list), "... Estimate non-linear Registration")
        transf_nl = est_nl_transf(im, im_atlas_lin)

        # apply linear and non-linear transformation to atlas mask
        segmentation = apply_transf(im, mask_atlas, transf_lin)
        segmentation = apply_transf(im, segmentation, transf_nl)
        segmentation = sitk.Cast(segmentation, sitk.sitkUInt16)

        # add to atlas segmenation for majority voting
        atlas_segmentations.append(segmentation)

        #todo: below are some commands to vidualize intermediate results for debugging
        #registered_image = apply_transf(im, im_atlas_lin, transf_nl)
        #simg1 = sitk.Cast(sitk.RescaleIntensity(segmentation), sitk.sitkUInt8)
        #simg2 = sitk.Cast(sitk.RescaleIntensity(im), sitk.sitkUInt8)
        #cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
        #sitk.Show(cimg, "segmentation of ith atlas registered")

        #simg1 = sitk.Cast(sitk.RescaleIntensity(registered_image), sitk.sitkUInt8)
        #simg2 = sitk.Cast(sitk.RescaleIntensity(im), sitk.sitkUInt8)
        #cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
        #sitk.Show(cimg, "registered image of ith atlas")

        #simg1 = sitk.Cast(sitk.RescaleIntensity(im_atlas), sitk.sitkUInt8)
        #simg2 = sitk.Cast(sitk.RescaleIntensity(mask_atlas), sitk.sitkUInt8)
        #cimg = sitk.Compose(simg1, simg2, simg1 // 2.0 + simg2 // 2.0)
        #sitk.Show(cimg, "atlas image + mask")


        #sitk.Show(mask_atlas, "gt segmentation")

    # return majority voting of mask
    return sitk.LabelVoting(atlas_segmentations)