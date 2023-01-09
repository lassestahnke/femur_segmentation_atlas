"""
This file contains all methods used for estimating the transform for registration of a moving image to a reference
image. This file was written by https://github.com/lassestahnke and https://github.com/RebeccaBonato with the help of
sitk tutorials.
"""

import SimpleITK as sitk


def command_iteration(method):
    """
    Function to print current number of iterations and current metric value.
    :param method:
    :return:
    """
    print(
        f"{method.GetOptimizerIteration():3} "
        + f"= {method.GetMetricValue():10.5f} "
        + f": {method.GetOptimizerPosition()}"
    )


def est_lin_transf(im_ref, im_mov):
    """
    Estimate linear transform to align `im_mov` to `im_ref` and return the transform parameters.
    :param  im_ref: [sitk image] reference image
            im_mov: [sitk image] moving image
    return: transformation
    """
    im_ref_mask = im_ref > 150
    im_ref_mask = sitk.Cast(im_ref_mask, sitk.sitkInt16)

    im_mov_mask = im_mov > 150  # only look at strong signals
    im_mov_mask = sitk.Cast(im_mov_mask, sitk.sitkInt16)

    initial_transform = sitk.CenteredTransformInitializer(im_ref_mask,
                                                          im_mov_mask,
                                                          sitk.Similarity3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.MOMENTS
                                                          )

    # Set methods for registration; Start with Linear
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsRegularStepGradientDescent(1.2, 0.01, 20)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetMetricSamplingPercentage(0.10)
    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    return R.Execute(im_ref, im_mov)


def est_nl_transf(im_ref, im_mov):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and return the transform parameters.
    :param  im_ref: [sitk image] reference image
            im_mov: [sitk image] moving image
    return: transformation
    """

    im_ref_mask = im_ref > 150
    im_ref = sitk.Normalize(im_ref)
    im_ref = sitk.DiscreteGaussian(im_ref, 3)

    im_mov_mask = im_mov > 150  # only look at strong signals
    im_mov = sitk.Normalize(im_mov)
    im_mov = sitk.DiscreteGaussian(im_mov, 3)

    transformDomainMeshSize = [8] * im_mov.GetDimension()
    tx = sitk.BSplineTransformInitializer(im_ref, transformDomainMeshSize)

    print("Initial Parameters:")
    print(tx.GetParameters())

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(50)
    R.SetOptimizerAsGradientDescentLineSearch(
        5.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5
    )
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)
    R.SetMetricMovingMask(im_mov_mask)
    R.SetMetricFixedMask(im_ref_mask)

    R.SetInitialTransform(tx, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)
    # R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    return R.Execute(im_ref, im_mov)
