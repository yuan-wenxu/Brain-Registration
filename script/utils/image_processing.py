"""Image-processing helpers shared by registration pipeline stages."""

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks


def _normalize_01(image):
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.min(arr[finite]))
    high = float(np.max(arr[finite]))
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - low) / (high - low), 0.0, 1.0)


def _axis_grid_peaks(log_power, axis, min_period, max_period, max_peaks):
    """Detect periodic peaks on one centered Fourier axis."""
    h, w = log_power.shape
    cy, cx = h // 2, w // 2
    band = 2
    if axis == 'x':
        spectrum = np.mean(log_power[cy - band:cy + band + 1, :], axis=0)
        center, size = cx, w
    elif axis == 'y':
        spectrum = np.mean(log_power[:, cx - band:cx + band + 1], axis=1)
        center, size = cy, h
    else:
        raise ValueError("axis must be 'x' or 'y'")

    positive = spectrum[center + 1:]
    offsets = np.arange(1, positive.size + 1)
    min_offset = max(2, int(np.floor(size / float(max_period))))
    max_offset = min(positive.size - 1, int(np.ceil(size / float(min_period))))
    valid = (offsets >= min_offset) & (offsets <= max_offset)
    if not np.any(valid):
        return []

    baseline = gaussian_filter1d(positive, sigma=max(2.0, size / 100.0))
    residual = positive - baseline
    valid_values = residual[valid]
    median = float(np.median(valid_values))
    mad = float(np.median(np.abs(valid_values - median)))
    prominence = max(0.12, 4.0 * 1.4826 * mad)
    peaks, properties = find_peaks(
        residual,
        distance=max(2, int(round(size / float(max_period)))),
        prominence=prominence,
    )

    candidates = []
    for peak, peak_prominence in zip(peaks, properties['prominences']):
        offset = int(peak + 1)
        if min_offset <= offset <= max_offset:
            candidates.append((offset, float(peak_prominence)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[:max_peaks]


def _masked_profile(image, mask, axis):
    values = np.where(mask > 0.5, image, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        profile = np.nanmedian(values, axis=axis)
    valid = np.isfinite(profile)
    if np.count_nonzero(valid) < 2:
        return np.zeros_like(profile, dtype=np.float32)
    coordinates = np.arange(profile.size)
    return np.interp(coordinates, coordinates[valid], profile[valid]).astype(np.float32)


def _remove_axis_stripes(image, mask, strength=0.9, profile_sigma=8.0):
    """Remove non-stationary row/column offsets before spectral filtering."""
    corrected = np.asarray(image, dtype=np.float32).copy()
    column_profile = _masked_profile(corrected, mask, axis=0)
    column_artifact = column_profile - gaussian_filter1d(column_profile, profile_sigma)
    corrected -= float(strength) * column_artifact[None, :] * mask

    row_profile = _masked_profile(corrected, mask, axis=1)
    row_artifact = row_profile - gaussian_filter1d(row_profile, profile_sigma)
    corrected -= float(strength) * row_artifact[:, None] * mask
    return _normalize_01(corrected), column_artifact, row_artifact


def remove_periodic_grid(
    image,
    tissue_mask=None,
    *,
    notch_strength=0.85,
    notch_sigma=2.0,
    min_period=6.0,
    max_period=100.0,
    max_peaks_per_axis=8,
    stripe_profile_strength=0.9,
    stripe_profile_sigma=8.0,
):
    """Suppress an axis-aligned periodic grid with soft Fourier notches.

    This creates a registration/scoring proxy while preserving the original
    source image separately. Returns the filtered image and JSON-serializable
    diagnostics.
    """
    arr = _normalize_01(image)
    if arr.ndim != 2:
        raise ValueError('remove_periodic_grid expects a 2-D image')
    if not 0.0 <= float(notch_strength) <= 1.0:
        raise ValueError('notch_strength must be between 0 and 1')
    if float(notch_sigma) <= 0:
        raise ValueError('notch_sigma must be positive')

    if tissue_mask is None:
        mask = np.ones(arr.shape, dtype=np.float32)
    else:
        mask = np.asarray(tissue_mask, dtype=np.float32)
        if mask.shape != arr.shape:
            raise ValueError('tissue_mask shape must match image shape')
        mask = (mask > 0.5).astype(np.float32)

    profile_corrected, column_artifact, row_artifact = _remove_axis_stripes(
        arr,
        mask,
        strength=stripe_profile_strength,
        profile_sigma=stripe_profile_sigma,
    )

    background_sigma = max(3.0, min(arr.shape) / 40.0)
    detection = (
        profile_corrected - gaussian_filter(profile_corrected, sigma=background_sigma)
    ) * mask
    window = np.outer(np.hanning(arr.shape[0]), np.hanning(arr.shape[1]))
    detection_fft = np.fft.fftshift(np.fft.fft2(detection * window))
    log_power = np.log1p(np.abs(detection_fft))
    x_peaks = _axis_grid_peaks(
        log_power, 'x', min_period, max_period, max_peaks_per_axis,
    )
    y_peaks = _axis_grid_peaks(
        log_power, 'y', min_period, max_period, max_peaks_per_axis,
    )

    pad_y = max(8, arr.shape[0] // 8)
    pad_x = max(8, arr.shape[1] // 8)
    padded = np.pad(
        profile_corrected, ((pad_y, pad_y), (pad_x, pad_x)), mode='reflect',
    )
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    ph, pw = padded.shape
    cy, cx = ph // 2, pw // 2
    yy, xx = np.ogrid[:ph, :pw]
    notch = np.ones((ph, pw), dtype=np.float32)

    for axis, peaks, scale in (
        ('x', x_peaks, pw / float(arr.shape[1])),
        ('y', y_peaks, ph / float(arr.shape[0])),
    ):
        for offset, _ in peaks:
            padded_offset = float(offset) * scale
            centers = (
                ((cy, cx - padded_offset), (cy, cx + padded_offset))
                if axis == 'x'
                else ((cy - padded_offset, cx), (cy + padded_offset, cx))
            )
            for peak_y, peak_x in centers:
                distance_sq = np.square(yy - peak_y) + np.square(xx - peak_x)
                attenuation = 1.0 - float(notch_strength) * np.exp(
                    -distance_sq / (2.0 * float(notch_sigma) ** 2)
                )
                notch *= attenuation.astype(np.float32)

    filtered_padded = np.fft.ifft2(np.fft.ifftshift(spectrum * notch)).real
    filtered = filtered_padded[pad_y:pad_y + arr.shape[0], pad_x:pad_x + arr.shape[1]]
    filtered = _normalize_01(filtered)
    filtered = filtered * mask + arr * (1.0 - mask)

    diagnostics = {
        'applied': bool(x_peaks or y_peaks),
        'notch_strength': float(notch_strength),
        'notch_sigma': float(notch_sigma),
        'stripe_profile_strength': float(stripe_profile_strength),
        'stripe_profile_sigma': float(stripe_profile_sigma),
        'column_artifact_rms': float(np.sqrt(np.mean(np.square(column_artifact)))),
        'row_artifact_rms': float(np.sqrt(np.mean(np.square(row_artifact)))),
        'min_period_px': float(min_period),
        'max_period_px': float(max_period),
        'x_frequency_peaks': [
            {'offset': int(offset), 'period_px': float(arr.shape[1] / offset), 'prominence': prominence}
            for offset, prominence in x_peaks
        ],
        'y_frequency_peaks': [
            {'offset': int(offset), 'period_px': float(arr.shape[0] / offset), 'prominence': prominence}
            for offset, prominence in y_peaks
        ],
    }
    return filtered.astype(np.float32), diagnostics
