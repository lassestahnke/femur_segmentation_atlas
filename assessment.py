"""
This script provides functions to evaluate segmentation results and compare them with the ground truth.
"""
import SimpleITK as sitk


def compute_dice(im1, im2):
    """
    function to compute the dice-score between two images.
    :param im1: First image
    :param im2: Second image
    :return: dice score
    """
    im1 = im1 == 2 + im1 == 4
    im2 = im2 == 2 + im2 == 4

    filter = sitk.LabelOverlapMeasuresImageFilter()
    filter.Execute(sitk.Cast(im1, sitk.sitkUInt8), sitk.Cast(im2, sitk.sitkUInt8))

    return filter.GetDiceCoefficient()


def compute_hausdorff(im1, im2):
    """
    Computes the hausdorff distance between two label maps.
    :param im1: label map 1
    :param im2: label map 2
    :return: hausdorff distance
    """
    im1 = im1 == 2 + im1 == 4
    im2 = im2 == 2 + im2 == 4

    filter = sitk.HausdorffDistanceImageFilter()
    filter.Execute(sitk.Cast(im1, sitk.sitkUInt8), sitk.Cast(im2, sitk.sitkUInt8))

    return filter.GetHausdorffDistance()
