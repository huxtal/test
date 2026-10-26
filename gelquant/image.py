"""Loading and preparing SDS-PAGE gel images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage import io as skio
from skimage.transform import rotate


@dataclass
class GelImage:
    path: Path
    raw: np.ndarray            # original grayscale pixel values (float)
    signal: np.ndarray         # bands positive, background ~0..low, scaled 0..1
    inverted: bool             # True if the raw image had dark bands on light background
    bit_depth: int
    channel: str = "gray"      # plane analysed ("gray", or "r"/"g"/"b" of a colour image)
    crop: tuple[int, int, int, int] | None = None
    angle: float = 0.0         # deskew rotation applied (degrees, counter-clockwise)
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def shape(self) -> tuple[int, int]:
        return self.signal.shape


CHANNELS = {"r": 0, "g": 1, "b": 2}


def _to_gray(img: np.ndarray, channel: str = "auto") -> tuple[np.ndarray, str]:
    """Grayscale plane to analyse, and which one was used.

    For colour images ``auto`` picks the channel with the widest intensity
    range: for Coomassie blue that is red (blue dye absorbs red light), while
    the blue channel is nearly flat.
    """
    if img.ndim == 2:
        return img.astype(float), "gray"
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = img[..., :3].astype(float)
        if channel == "gray":
            return rgb @ np.array([0.2126, 0.7152, 0.0722]), "gray"
        if channel in CHANNELS:
            return rgb[..., CHANNELS[channel]], channel
        spread = [np.percentile(rgb[..., c], 99) - np.percentile(rgb[..., c], 1) for c in range(3)]
        c = int(np.argmax(spread))
        return rgb[..., c], "rgb"[c]
    if img.ndim == 3:   # multi-page TIFF: take the first plane
        return img[0].astype(float), "page 1"
    raise ValueError(f"unsupported image shape {img.shape}")


def load_gel(path: str | Path, invert: bool | None = None, channel: str = "auto",
             crop: tuple[int, int, int, int] | None = None) -> GelImage:
    """Read a gel image and return it with bands as positive signal.

    ``crop`` = (x0, y0, x1, y1) keeps only the resolving gel (exclude wells,
    stacking gel, dye front and gel edges, which would otherwise read as
    bands).  ``invert=None`` decides automatically: images whose background (median)
    is brighter than mid-range are treated as dark-band-on-light (Coomassie,
    silver) and inverted; fluorescent / stain-free images are kept as is.
    """
    path = Path(path)
    img = skio.imread(str(path))
    warnings: list[str] = []
    dtype = img.dtype
    bit_depth = 16 if dtype == np.uint16 else 8 if dtype == np.uint8 else 32
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        warnings.append("colour image (photo/scan): band volumes are only semi-quantitative; "
                        "a raw grayscale TIFF from a gel imager is preferable")
    if path.suffix.lower() in (".jpg", ".jpeg"):
        warnings.append("JPEG is lossy and often contrast-adjusted; band volumes may not "
                        "be linear (use the imager's raw 16-bit TIFF if possible)")

    gray, used = _to_gray(img, channel)
    if crop is not None:
        x0, y0, x1, y1 = crop
        gray = gray[y0:y1, x0:x1]
    full = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else float(gray.max())
    sat_hi = np.mean(gray >= full * 0.999)
    sat_lo = np.mean(gray <= 0)

    lo, hi = np.percentile(gray, [0.1, 99.9])
    if invert is None:
        invert = np.median(gray) > (lo + hi) / 2
    # Saturation that matters is at the band end of the range.
    sat = sat_lo if invert else sat_hi
    if sat > 1e-4:
        warnings.append(f"{sat * 100:.2f}% of pixels are saturated at the band end of the "
                        "range: the strongest bands are underestimated (shorter exposure)")

    sig = (full - gray) if invert else gray.copy()
    sig -= np.percentile(sig, 0.5)
    sig /= max(np.percentile(sig, 99.99), 1e-12)
    return GelImage(path, gray, np.clip(sig, 0, None), bool(invert), bit_depth,
                    channel=used, crop=crop, warnings=warnings)


def deskew_angle(sig: np.ndarray, max_angle: float = 3.0, step: float = 0.25) -> float:
    """Rotation (degrees) that makes lanes most vertical.

    Lanes are vertical when the column-sum profile has the sharpest contrast,
    so the angle maximising its variance is chosen (on a downsampled copy).
    """
    f = max(1, int(round(max(sig.shape) / 600)))
    small = sig[::f, ::f]
    best, best_score = 0.0, -np.inf
    for a in np.arange(-max_angle, max_angle + 1e-9, step):
        r = rotate(small, a, resize=False, mode="constant", cval=0.0, order=1)
        prof = r.mean(axis=0)
        score = np.var(np.diff(prof))
        if score > best_score:
            best, best_score = float(a), score
    return best


def apply_rotation(gel: GelImage, angle: float) -> GelImage:
    if abs(angle) < 1e-9:
        return gel
    gel.signal = rotate(gel.signal, angle, resize=False, mode="edge", order=1)
    gel.raw = rotate(gel.raw, angle, resize=False, mode="edge", order=1, preserve_range=True)
    gel.angle = angle
    return gel

