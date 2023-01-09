import SimpleITK as sitk


def apply_transf(im_ref, im_mov, xfm):
    """
    Apply given transform `xfm` to `im_mov` and return the transformed image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(im_ref)  # reference image for size, origin and spacing
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) 
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(xfm)
    return resampler.Execute(im_mov)
