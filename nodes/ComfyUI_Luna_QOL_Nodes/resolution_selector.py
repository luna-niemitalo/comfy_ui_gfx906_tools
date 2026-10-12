"""
resolution_selector.py  –  ComfyUI custom node
================================================
Outputs width & height for a target pixel budget and aspect ratio.

Pixel budget is specified as the **square root of the total pixel count**
(e.g. enter 1024 → 1024 × 1024 = 1 048 576 total pixels).

Ratio is entered as "W:H" (e.g. "16:9", "1:1", "4:3").

Optional chunk-align mode snaps both dimensions to the nearest multiple
of CHUNK_SIZE (default 64), so every image is composed of N×M tiles of
exactly CHUNK_SIZE² pixels – useful for diffusion model tile attention.

Limits
------
* Maximum ratio component value  : MAX_RATIO_VALUE  (1 000)
* Minimum ratio component value  : 1
* Maximum pixel-root value       : MAX_PIXEL_ROOT   (16 384, i.e. 16 K²)
* Minimum pixel-root value       : MIN_PIXEL_ROOT   (8)
* Absolute maximum dimension     : MAX_DIMENSION    (16 384)
* Default chunk size             : CHUNK_SIZE        (64)

Extension points
----------------
* _parse_ratio()          – tokenises / validates the "W:H" string
* _validate_pixel_root()  – validates the pixel-root integer
* _compute_dimensions()   – core maths; override for exotic rounding modes
* _snap_to_chunk()        – rounds a single value to the nearest chunk multiple
* _clamp_dimension()      – final safety clamp to MAX_DIMENSION
"""

import math
from typing import Tuple
import comfy.utils


# ---------------------------------------------------------------------------
# Module-level constants – change these to tune global limits.
# ---------------------------------------------------------------------------

MAX_RATIO_VALUE: int = 1_000        # largest allowed number in either ratio part
MIN_RATIO_VALUE: int = 1            # smallest (must be ≥ 1)
MAX_PIXEL_ROOT: int = 16_384        # sqrt(pixels); 16 384² = 268 M ≈ 16 K image
MIN_PIXEL_ROOT: int = 8             # tiny lower bound for sanity
MAX_DIMENSION: int = 16_384         # absolute per-axis safety cap (16 K)
CHUNK_SIZE: int = 64                # default tile size for snap-to-chunk mode


# ---------------------------------------------------------------------------
# Pure helper functions  (no class state – easy to unit-test / reuse)
# ---------------------------------------------------------------------------

def _parse_ratio(ratio_str: str) -> Tuple[int, int]:
    """
    Parse and validate a "W:H" ratio string.

    Parameters
    ----------
    ratio_str : str
        User-supplied string, e.g. "16:9", "1 : 1", "5:8".

    Returns
    -------
    (ratio_w, ratio_h) : Tuple[int, int]
        Validated, reduced (GCD-simplified) ratio components.

    Raises
    ------
    ValueError
        On any malformed or out-of-range input.

    Notes
    -----
    The ratio is GCD-reduced (e.g. "4:2" → (2, 1)) before returning so
    that downstream maths stays in the smallest integer domain.
    """
    if not isinstance(ratio_str, str) or ratio_str.strip() == "":
        raise ValueError("Ratio must be a non-empty string such as '16:9'.")

    parts = ratio_str.strip().split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Ratio '{ratio_str}' is invalid – expected exactly one ':' separator, "
            f"e.g. '16:9'."
        )

    try:
        rw = int(parts[0].strip())
        rh = int(parts[1].strip())
    except ValueError:
        raise ValueError(
            f"Both parts of the ratio must be integers, got '{parts[0]}' and '{parts[1]}'."
        )

    for name, val in (("width component", rw), ("height component", rh)):
        if val < MIN_RATIO_VALUE:
            raise ValueError(
                f"Ratio {name} must be ≥ {MIN_RATIO_VALUE}, got {val}."
            )
        if val > MAX_RATIO_VALUE:
            raise ValueError(
                f"Ratio {name} must be ≤ {MAX_RATIO_VALUE} (got {val}). "
                f"Ratios like '{val}:…' are unreasonably large."
            )

    # Reduce to lowest terms so (4, 2) → (2, 1)
    divisor = math.gcd(rw, rh)
    return rw // divisor, rh // divisor


def _validate_pixel_root(pixel_root: int) -> int:
    """
    Validate and return the pixel-root value (√total_pixels).

    Parameters
    ----------
    pixel_root : int
        Square root of the desired total pixel count.

    Returns
    -------
    int
        The same value if valid.

    Raises
    ------
    ValueError
        If the value is out of the allowed range.
    """
    if not isinstance(pixel_root, int) or isinstance(pixel_root, bool):
        try:
            pixel_root = int(pixel_root)
        except (TypeError, ValueError):
            raise ValueError(f"Pixel root must be an integer, got {pixel_root!r}.")

    if pixel_root < MIN_PIXEL_ROOT:
        raise ValueError(
            f"Pixel root must be ≥ {MIN_PIXEL_ROOT}, got {pixel_root}."
        )
    if pixel_root > MAX_PIXEL_ROOT:
        raise ValueError(
            f"Pixel root must be ≤ {MAX_PIXEL_ROOT} (≈ {MAX_PIXEL_ROOT}² = "
            f"{MAX_PIXEL_ROOT**2:,} pixels), got {pixel_root}."
        )
    return pixel_root


def _snap_to_chunk(value: float, chunk: int = CHUNK_SIZE) -> int:
    """
    Round *value* to the nearest positive multiple of *chunk*.

    A value of 0 is snapped up to *chunk* so we never produce a zero
    dimension.

    Parameters
    ----------
    value : float
        The raw (possibly fractional) dimension.
    chunk : int
        Tile size to snap to. Defaults to the module-level CHUNK_SIZE.

    Returns
    -------
    int
        Nearest multiple of *chunk* ≥ chunk.
    """
    snapped = max(chunk, round(value / chunk) * chunk)
    return int(snapped)


def _clamp_dimension(value: int, axis: str = "dimension") -> int:
    """
    Clamp a dimension to [1, MAX_DIMENSION] and warn if clamped.

    Parameters
    ----------
    value : int
        Computed dimension (pixels).
    axis : str
        Human-readable label ("width" or "height") for log messages.

    Returns
    -------
    int
        Clamped value.
    """
    if value > MAX_DIMENSION:
        print(
            f"[ResolutionSelector] Warning: computed {axis} {value} exceeds "
            f"MAX_DIMENSION {MAX_DIMENSION} – clamping."
        )
        return MAX_DIMENSION
    if value < 1:
        print(
            f"[ResolutionSelector] Warning: computed {axis} {value} < 1 – clamping to 1."
        )
        return 1
    return value


def _compute_dimensions(
    ratio_w: int,
    ratio_h: int,
    pixel_root: int,
    snap_to_chunk: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[int, int]:
    """
    Core maths: derive (width, height) from ratio and pixel-root.

    Formula
    -------
    total_pixels = pixel_root²
    width  = sqrt(total_pixels × ratio_w / ratio_h)
    height = total_pixels / width

    In snap mode both axes are individually rounded to the nearest
    multiple of *chunk_size*, so the total pixel count may differ
    slightly from the target – but the ratio of chunks is preserved.

    Parameters
    ----------
    ratio_w, ratio_h : int
        Already-validated, GCD-reduced ratio components.
    pixel_root : int
        Square root of the desired pixel count.
    snap_to_chunk : bool
        If True, snap both dimensions to the nearest chunk multiple.
    chunk_size : int
        Tile size used in snap mode.

    Returns
    -------
    (width, height) : Tuple[int, int]
        Final clamped integer dimensions.
    """
    total_pixels: float = float(pixel_root) ** 2

    # Exact floating-point dimensions
    raw_w: float = math.sqrt(total_pixels * ratio_w / ratio_h)
    raw_h: float = total_pixels / raw_w   # keeps product = total_pixels

    if snap_to_chunk:
        # In chunk mode we count chunks per axis and snap independently.
        # chunks_w / chunks_h ≈ ratio_w / ratio_h (may round independently).
        width  = _snap_to_chunk(raw_w, chunk_size)
        height = _snap_to_chunk(raw_h, chunk_size)
    else:
        width  = max(1, round(raw_w))
        height = max(1, round(raw_h))

    width  = _clamp_dimension(width,  "width")
    height = _clamp_dimension(height, "height")

    return width, height


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------

class LunaResolutionSelector:
    INPUT_NODE = True
    """
    ComfyUI node: Resolution Selector
    ----------------------------------
    Converts a human-friendly pixel budget + aspect ratio into concrete
    width & height integer outputs.

    Inputs
    ------
    ratio        (STRING)  – Aspect ratio, e.g. "16:9", "1:1", "5:8"
    pixel_root   (INT)     – √(total pixels), e.g. 1024 → 1 048 576 px
    snap_to_chunk (BOOL)   – Snap dimensions to nearest CHUNK_SIZE multiple

    Outputs
    -------
    width  (INT)
    height (INT)
    ratio        (STRING)  – Aspect ratio, e.g. "16:9", "1:1", "5:8"
    pixel_root   (INT)     – √(total pixels), e.g. 1024 → 1 048 576 px
    snap_to_chunk (BOOL)   – Snap dimensions to nearest CHUNK_SIZE multiple
    """

    CATEGORY = "utils/resolution"
    FUNCTION = "select_resolution"
    RETURN_TYPES = ("INT", "INT", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "ratio", "pixel_root", "snap_to_chunk")

    # ------------------------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ratio": (
                    "STRING",
                    {
                        "default": "1:1",
                        "tooltip": (
                            "Aspect ratio as W:H.  Examples: 1:1, 16:9, 5:8.  "
                            f"Each component must be an integer between "
                            f"{MIN_RATIO_VALUE} and {MAX_RATIO_VALUE}."
                        ),
                    },
                ),
                "pixel_root": (
                    "INT",
                    {
                        "default": 1024,
                        "min": MIN_PIXEL_ROOT,
                        "max": MAX_PIXEL_ROOT,
                        "step": 1,
                        "tooltip": (
                            "Square root of the target pixel count.  "
                            "1024 → 1024×1024 = 1 M pixels (for a 1:1 ratio).  "
                            f"Range: {MIN_PIXEL_ROOT}–{MAX_PIXEL_ROOT}."
                        ),
                    },
                ),
            },
            "optional": {
                "snap_to_chunk": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            f"Round both dimensions to the nearest multiple of "
                            f"{CHUNK_SIZE}.  Useful for tiled diffusion attention.  "
                            f"A 1:1 image at pixel_root=1024 is unchanged; a "
                            f"non-square ratio may shift slightly to preserve tiling."
                        ),
                    },
                ),
            },
        }

    # ------------------------------------------------------------------
    def select_resolution(
        self,
        ratio: str,
        pixel_root: int,
        snap_to_chunk: bool = False,
    ) -> Tuple[int, int, str, int, bool]:
        """
        Main execution method called by ComfyUI.

        Validates all inputs, delegates to helper functions, and returns
        (width, height) as a tuple of ints.

        Parameters
        ----------
        ratio : str
            Raw user string, e.g. "16:9".
        pixel_root : int
            Square root of the desired total pixel count.
        snap_to_chunk : bool
            Whether to snap dimensions to CHUNK_SIZE multiples.

        Returns
        -------
        Tuple[int, int, str, int, bool]
            (width, height) in pixels.
            ratio : str
                Ratio string, e.g. "16:9".
            pixel_root : int
                Square root of the desired total pixel count.
            snap_to_chunk : bool
                Whether dimensions were snapped to CHUNK_SIZE multiples.
        """
        # --- Validate ---
        try:
            ratio_w, ratio_h = _parse_ratio(ratio)
        except ValueError as exc:
            raise ValueError(f"[ResolutionSelector] Invalid ratio – {exc}") from exc

        try:
            pixel_root = _validate_pixel_root(pixel_root)
        except ValueError as exc:
            raise ValueError(f"[ResolutionSelector] Invalid pixel root – {exc}") from exc

        # --- Compute ---
        width, height = _compute_dimensions(
            ratio_w, ratio_h, pixel_root,
            snap_to_chunk=snap_to_chunk,
            chunk_size=CHUNK_SIZE,
        )

        # --- Log ---
        total = width * height
        chunk_note = f"  [chunk-aligned to ×{CHUNK_SIZE}]" if snap_to_chunk else ""
        print(
            f"[ResolutionSelector] ratio={ratio_w}:{ratio_h}  "
            f"pixel_root={pixel_root}  →  {width}×{height}  "
            f"({total:,} px){chunk_note}"
        )

        return (width, height, ratio, pixel_root, snap_to_chunk)
