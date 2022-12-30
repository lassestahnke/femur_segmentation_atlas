"""
This file contains all methods used for estimating the transform for registration of a moving image to a reference
image. This file was written by https://github.com/lassestahnke and *insert Rebis Github account* with the help of
sitk turorials.
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
    """
    im_ref_mask = im_ref > 250
    im_ref_mask = sitk.Cast(im_ref_mask, sitk.sitkInt8)

    im_mov_mask = im_mov > 250  # only look at strong signals
    im_mov_mask = sitk.Cast(im_mov_mask, sitk.sitkInt8)

    initial_transform = sitk.CenteredTransformInitializer(im_ref,
                                                          im_mov,
                                                          sitk.Similarity3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY
                                                          )

    # Set methods for registration; Start with Linear
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsRegularStepGradientDescent(1, 0.01, 300)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetMetricSamplingStrategy(R.NONE)
    R.SetMetricMovingMask(im_mov_mask)
    R.SetMetricFixedMask(im_ref_mask)
    R.SetInitialTransform(initial_transform, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    return R.Execute(im_ref, im_mov)

def est_nl_transf(im_ref, im_mov):
    """
    Estimate non-linear transform to align `im_mov` to `im_ref` and return the transform parameters.
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
    R.SetMetricAsJointHistogramMutualInformation()
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05)
    R.SetMetricMovingMask(im_mov_mask)
    R.SetMetricFixedMask(im_ref_mask)
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=200,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )

    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx, inPlace=False)
    R.SetInterpolator(sitk.sitkLinear)
    R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))

    return R.Execute(im_ref, im_mov)
