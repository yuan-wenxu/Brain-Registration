"""Reusable 2-D registration stages shared by Step2 and Step3."""

import SimpleITK as sitk


def _set_sampling(registration, strategy, percentage, seed=None):
    strategy = str(strategy).lower()
    if strategy == 'random':
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        if seed is None:
            registration.SetMetricSamplingPercentage(float(percentage))
        else:
            registration.SetMetricSamplingPercentage(float(percentage), seed=int(seed))
    elif strategy == 'regular':
        registration.SetMetricSamplingStrategy(registration.REGULAR)
        registration.SetMetricSamplingPercentage(float(percentage))
    elif strategy == 'none':
        registration.SetMetricSamplingStrategy(registration.NONE)
    else:
        raise ValueError(f'Unsupported metric sampling strategy: {strategy}')


def _configure_registration(
    registration,
    histogram_bins,
    sampling_strategy,
    sampling_percentage,
    sampling_seed,
    learning_rate,
    number_of_iterations,
    shrink_factors,
    smoothing_sigmas,
):
    registration.SetMetricAsMattesMutualInformation(int(histogram_bins))
    _set_sampling(registration, sampling_strategy, sampling_percentage, sampling_seed)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=float(learning_rate),
        numberOfIterations=int(number_of_iterations),
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([int(x) for x in shrink_factors])
    registration.SetSmoothingSigmasPerLevel([float(x) for x in smoothing_sigmas])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()


def _stage_result(name, registration, transform, image):
    return {
        'name': name,
        'transform': transform,
        'image': image,
        'metric': float(registration.GetMetricValue()),
        'iterations': int(registration.GetOptimizerIteration()),
        'stop_reason': registration.GetOptimizerStopConditionDescription(),
    }


def rigid_register(
    fixed,
    moving,
    *,
    histogram_bins=50,
    sampling_strategy='random',
    sampling_percentage=0.25,
    sampling_seed=None,
    learning_rate=0.1,
    number_of_iterations=140,
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0),
    metric_history=None,
    stage_name='rigid',
):
    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration = sitk.ImageRegistrationMethod()
    _configure_registration(
        registration, histogram_bins, sampling_strategy, sampling_percentage,
        sampling_seed, learning_rate, number_of_iterations,
        shrink_factors, smoothing_sigmas,
    )
    registration.SetInitialTransform(initial, inPlace=False)
    if metric_history is not None:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: metric_history.append((
                registration.GetOptimizerIteration(),
                float(registration.GetMetricValue()),
                stage_name,
            )),
        )
    transform = registration.Execute(fixed, moving)
    transform = sitk.Euler2DTransform(transform.GetNthTransform(0))
    image = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return _stage_result(stage_name, registration, transform, image)


def affine_register(
    fixed,
    moving,
    *,
    center=None,
    histogram_bins=64,
    sampling_strategy='random',
    sampling_percentage=0.25,
    sampling_seed=None,
    learning_rate=0.15,
    number_of_iterations=220,
    shrink_factors=(4, 2, 1),
    smoothing_sigmas=(2, 1, 0),
    metric_history=None,
    stage_name='affine',
):
    initial = sitk.AffineTransform(2)
    if center is not None:
        initial.SetCenter(center)
    registration = sitk.ImageRegistrationMethod()
    _configure_registration(
        registration, histogram_bins, sampling_strategy, sampling_percentage,
        sampling_seed, learning_rate, number_of_iterations,
        shrink_factors, smoothing_sigmas,
    )
    registration.SetInitialTransform(initial, inPlace=False)
    if metric_history is not None:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: metric_history.append((
                registration.GetOptimizerIteration(),
                float(registration.GetMetricValue()),
                stage_name,
            )),
        )
    transform = registration.Execute(fixed, moving)
    image = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return _stage_result(stage_name, registration, transform, image)


def bspline_register(
    fixed,
    moving,
    *,
    mesh_size,
    initial_transform=None,
    stage_name='bspline',
    metric_history=None,
    histogram_bins=64,
    sampling_strategy='regular',
    sampling_percentage=0.25,
    sampling_seed=None,
    learning_rate=0.08,
    number_of_iterations=260,
    shrink_factors=(8, 4, 2, 1),
    smoothing_sigmas=(4, 2, 1, 0),
):
    if initial_transform is None:
        initial = sitk.BSplineTransformInitializer(
            fixed, [int(mesh_size), int(mesh_size)], order=3,
        )
    else:
        initial = sitk.BSplineTransform(initial_transform)

    registration = sitk.ImageRegistrationMethod()
    _configure_registration(
        registration, histogram_bins, sampling_strategy, sampling_percentage,
        sampling_seed, learning_rate, number_of_iterations,
        shrink_factors, smoothing_sigmas,
    )
    registration.SetInitialTransform(initial, inPlace=True)
    if metric_history is not None:
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: metric_history.append((
                registration.GetOptimizerIteration(),
                float(registration.GetMetricValue()),
                stage_name,
            )),
        )
    transform = registration.Execute(fixed, moving)
    image = sitk.Resample(
        moving, fixed, transform, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )
    return _stage_result(stage_name, registration, transform, image)


def compose_transforms(*transforms):
    composite = sitk.CompositeTransform(2)
    for transform in transforms:
        composite.AddTransform(transform)
    if hasattr(composite, 'FlattenTransform'):
        composite.FlattenTransform()
    return composite
