# xdsp_filters
```python
#!/usr/bin/env python3
"""
xdsp_filters.py

RBJ biquads + cascades + Butterworth + Linkwitz–Riley
All in one file.

Requires:
    numpy
    matplotlib
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lp2bp_zpk
from scipy.signal import cheb1ap, cheb2ap, ellipap,besselap
from math import pi, sin, cos, sqrt
from typing import Dict, List, Tuple, Literal, TypedDict

try:
    from scipy.signal import butterap
except ImportError:
    import numpy as np

    def butterap(N):
        """Local fallback: analog Butterworth prototype (ωc=1)."""
        if N < 1:
            raise ValueError("Order must be >= 1.")
        k = 1.0
        poles = []
        for m in range(N):
            theta = np.pi * (2*m + 1 + N) / (2*N)
            p = -np.sin(theta) + 1j*np.cos(theta)
            poles.append(p)
        return [], np.array(poles), k


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

class BiquadCoeffs(TypedDict):
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float


class BiquadState(TypedDict):
    b0: float
    b1: float
    b2: float
    a1: float
    a2: float
    mode: str
    fs: float
    f0: float
    Q: float
    gain_db: float
    slope: float
    x1: float
    x2: float
    y1: float
    y2: float


# ---------------------------------------------------------------------
# RBJ Biquad Designer
# ---------------------------------------------------------------------

def rbj_biquad_design(
    mode: str,
    f0: float,
    fs: float,
    Q: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
) -> BiquadCoeffs:
    """
    Compute RBJ-style biquad filter coefficients (Audio EQ Cookbook).

    mode:
        "lowpass", "highpass",
        "bandpass", "notch",
        "peak",
        "lowshelf", "highshelf"

    All coefficients are returned normalized (a0 = 1).
    """
    if fs <= 0:
        raise ValueError("Sampling rate fs must be > 0.")
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist (0 < f0 < fs/2).")
    if Q <= 0:
        raise ValueError("Q must be > 0.")
    if slope <= 0:
        raise ValueError("slope must be > 0.")

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * pi * (f0 / fs)
    cosw = cos(w0)
    sinw = sin(w0)

    # Default alpha (non-shelving)
    alpha = sinw / (2.0 * Q)

    # Shelf slope correction (RBJ: case S)
    if mode in ("lowshelf", "highshelf"):
        alpha = sinw / 2.0 * sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)

    # ---------------------------------------------------------------
    if mode == "lowpass":
        b0 = (1 - cosw) / 2.0
        b1 = 1 - cosw
        b2 = (1 - cosw) / 2.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "highpass":
        b0 = (1 + cosw) / 2.0
        b1 = -(1 + cosw)
        b2 = (1 + cosw) / 2.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "bandpass":
        # RBJ "constant 0 dB peak gain" variant
        b0 = alpha
        b1 = 0.0
        b2 = -alpha
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "notch":
        b0 = 1.0
        b1 = -2 * cosw
        b2 = 1.0
        a0 = 1 + alpha
        a1 = -2 * cosw
        a2 = 1 - alpha

    elif mode == "peak":
        b0 = 1 + alpha * A
        b1 = -2 * cosw
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw
        a2 = 1 - alpha / A

    elif mode == "lowshelf":
        sqrtA = sqrt(A)
        b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw)
        b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha)
        a0 = (A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cosw)
        a2 = (A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha

    elif mode == "highshelf":
        sqrtA = sqrt(A)
        b0 = A * ((A + 1) + (A - 1) * cosw + 2 * sqrtA * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw)
        b2 = A * ((A + 1) + (A - 1) * cosw - 2 * sqrtA * alpha)
        a0 = (A + 1) - (A - 1) * cosw + 2 * sqrtA * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cosw)
        a2 = (A + 1) - (A - 1) * cosw - 2 * sqrtA * alpha

    else:
        raise ValueError(f"Invalid rbj mode: '{mode}'")

    # Normalize to a0 = 1.0
    b0 /= a0
    b1 /= a0
    b2 /= a0
    a1 /= a0
    a2 /= a0

    return {"b0": b0, "b1": b1, "b2": b2, "a1": a1, "a2": a2}


# ---------------------------------------------------------------------
# Biquad State Init / Processing
# ---------------------------------------------------------------------

def biquad_init(
    mode: str,
    f0: float,
    fs: float,
    Q: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
) -> BiquadState:
    """Initialize a biquad filter with given design and zeroed state."""
    coeffs = rbj_biquad_design(mode, f0, fs, Q, gain_db, slope)
    return BiquadState(
        b0=coeffs["b0"],
        b1=coeffs["b1"],
        b2=coeffs["b2"],
        a1=coeffs["a1"],
        a2=coeffs["a2"],
        mode=mode,
        fs=fs,
        f0=f0,
        Q=Q,
        gain_db=gain_db,
        slope=slope,
        x1=0.0,
        x2=0.0,
        y1=0.0,
        y2=0.0,
    )


def biquad_tick(state: BiquadState, x: float) -> Tuple[float, BiquadState]:
    """Process one sample through a RBJ biquad."""
    b0, b1, b2 = state["b0"], state["b1"], state["b2"]
    a1, a2 = state["a1"], state["a2"]
    x1, x2 = state["x1"], state["x2"]
    y1, y2 = state["y1"], state["y2"]

    y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2

    state["x2"] = x1
    state["x1"] = x
    state["y2"] = y1
    state["y1"] = y

    return y, state


def biquad_block(state: BiquadState, x: np.ndarray) -> Tuple[np.ndarray, BiquadState]:
    """Process a block through a RBJ biquad."""
    b0, b1, b2 = state["b0"], state["b1"], state["b2"]
    a1, a2 = state["a1"], state["a2"]
    x1, x2 = state["x1"], state["x2"]
    y1, y2 = state["y1"], state["y2"]

    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x, dtype=np.float64)

    for n, xn in enumerate(x):
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        y[n] = yn
        x2, x1 = x1, xn
        y2, y1 = y1, yn

    state["x1"], state["x2"] = x1, x2
    state["y1"], state["y2"] = y1, y2

    return y, state


# ---------------------------------------------------------------------
# Cascades
# ---------------------------------------------------------------------

def cascade_init(stage_params: List[dict]) -> List[BiquadState]:
    """
    Initialize a cascade of RBJ biquads.

    stage_params: list of dicts, each accepted by biquad_init(...)
    """
    return [biquad_init(**p) for p in stage_params]


def cascade_block(states: List[BiquadState], x: np.ndarray) -> Tuple[np.ndarray, List[BiquadState]]:
    """
    Process a block through a cascade of biquads.
    Mutates the states list in-place.
    """
    y = np.asarray(x, dtype=np.float64)
    for i, st in enumerate(states):
        y, states[i] = biquad_block(st, y)
    return y, states


# ---------------------------------------------------------------------
# Butterworth Utilities
# ---------------------------------------------------------------------

def butterworth_Qs(order: int) -> List[float]:
    """
    Return Q values for an even-order Butterworth low-pass.

    order:
        Even integer: 2, 4, 6, 8, ...

    These Qs correspond to the pole pairs of the analog prototype.
    """
    if order % 2 != 0 or order < 2:
        raise ValueError("Butterworth order must be an even integer >= 2.")

    n_sections = order // 2
    Qs: List[float] = []
    for k in range(1, n_sections + 1):
        theta = np.pi * (2 * k - 1) / (2 * order)
        Q = 1.0 / (2.0 * np.cos(theta))
        Qs.append(Q)
    return Qs


def butterworth_lowpass_stages(order: int, f0: float, fs: float) -> List[BiquadState]:
    """Create a cascade of RBJ low-pass sections matching an Nth-order Butterworth."""
    Qs = butterworth_Qs(order)
    params = [{"mode": "lowpass", "f0": f0, "fs": fs, "Q": q} for q in Qs]
    return cascade_init(params)

def design_hpeq_butterworth_rbj(order, fs, f0, gain_db):
    if order < 2 or order % 2 != 0:
        raise ValueError("Order must be even and ≥ 2.")
    n_sections = order // 2
    Qs = butterworth_Qs(order)
    per_section_gain_db = gain_db / n_sections
    return [
        biquad_init("peak", f0=f0, fs=fs, Q=Q, gain_db=per_section_gain_db)
        for Q in Qs
    ]

def design_rbj_cascade(
    mode: str,
    order: int,
    fs: float,
    f0: float,
    Q_base: float = 0.707,
    gain_db: float = 0.0,
    slope: float = 1.0,
) -> list[BiquadState]:
    """
    Create a cascade of RBJ biquads of the given type (mode),
    scaled to emulate an Nth-order filter (Butterworth-like by default).

    Parameters
    ----------
    mode : str
        RBJ mode: 'lowpass', 'highpass', 'bandpass', 'notch',
                  'peak', 'lowshelf', 'highshelf'.
    order : int
        Even integer >= 2. Each biquad = 2nd order.
    fs : float
        Sampling frequency (Hz).
    f0 : float
        Center/cutoff frequency (Hz).
    Q_base : float
        Base Q value for single-section mode (used if no Q array).
        Default = 0.707.
    gain_db : float
        Total desired gain (dB). Distributed evenly across sections
        for gain-sensitive modes (peak, shelves).
    slope : float
        Shelf slope for shelving filters (default = 1.0).

    Returns
    -------
    stages : list[BiquadState]
        List of initialized RBJ biquads forming the cascade.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("Order must be even and ≥ 2.")
    if mode not in (
        "lowpass",
        "highpass",
        "bandpass",
        "notch",
        "peak",
        "lowshelf",
        "highshelf",
    ):
        raise ValueError(f"Unsupported RBJ mode '{mode}'")

    n_sections = order // 2

    # Butterworth-derived Qs for natural rolloff; otherwise uniform Q_base
    try:
        Qs = butterworth_Qs(order)
    except Exception:
        Qs = [Q_base] * n_sections

    # Divide total gain evenly across sections when gain applies
    per_section_gain_db = (
        gain_db / n_sections if mode in ("peak", "lowshelf", "highshelf") else 0.0
    )

    return [
        biquad_init(
            mode=mode,
            f0=f0,
            fs=fs,
            Q=Q,
            gain_db=per_section_gain_db,
            slope=slope,
        )
        for Q in Qs
    ]

# ---------------------------------------------------------------------
# Generic ZPK → SOS → BiquadState helpers
# ---------------------------------------------------------------------

def _pair_conjugates(roots: List[complex]) -> List[Tuple[complex, complex]]:
    """Pair complex-conjugate (or nearly equal real) roots into biquad pairs."""
    unused = list(roots)
    pairs = []

    while unused:
        r = unused.pop(0)
        # If essentially real:
        if abs(r.imag) < 1e-8:
            # find matching real or use itself twice
            match_idx = None
            for i, r2 in enumerate(unused):
                if abs(r2.imag) < 1e-8 and abs(r2.real - r.real) < 1e-6:
                    match_idx = i
                    break
            if match_idx is not None:
                r2 = unused.pop(match_idx)
            else:
                r2 = r
            pairs.append((r, r2))
        else:
            # complex: pair with its conjugate
            conj_idx = None
            for i, r2 in enumerate(unused):
                if abs(r2 - np.conj(r)) < 1e-6:
                    conj_idx = i
                    break
            if conj_idx is not None:
                r2 = unused.pop(conj_idx)
            else:
                # fallback: use conjugate directly
                r2 = np.conj(r)
            pairs.append((r, r2))

    return pairs


def zpk_to_biquad_states(
    z: List[complex],
    p: List[complex],
    k: float,
    fs: float,
    mode_label: str,
) -> List[BiquadState]:
    """
    Convert zeros/poles/gain to normalized biquad states (Direct Form I/II compatible).
    Assumes all roots come in real or complex-conjugate pairs.
    """
    # Pair poles; zeros may be fewer → pad with z = -1 (high-freq zeros).
    pole_pairs = _pair_conjugates(p)
    zero_pairs = _pair_conjugates(z) if z else []

    # Pad zeros if needed (zeros at z = -1 act like HF transmission zeros)
    while len(zero_pairs) < len(pole_pairs):
        zero_pairs.append((-1.0 + 0j, -1.0 + 0j))

    biquads: List[BiquadState] = []

    for i, (pp, zp) in enumerate(zip(pole_pairs, zero_pairs)):
        p1, p2 = pp
        z1, z2 = zp

        a = np.poly([p1, p2]).real  # [a0, a1, a2]
        b = np.poly([z1, z2]).real  # [b0, b1, b2]

        # Apply overall gain k to the FIRST section's numerator
        if i == 0:
            b *= k

        a0 = a[0]
        b0, b1, b2 = b / a0
        _, a1, a2 = a / a0

        biquads.append(
            BiquadState(
                b0=float(b0),
                b1=float(b1),
                b2=float(b2),
                a1=float(a1),
                a2=float(a2),
                mode=mode_label,
                fs=float(fs),
                f0=0.0,
                Q=0.0,
                gain_db=0.0,
                slope=1.0,
                x1=0.0,
                x2=0.0,
                y1=0.0,
                y2=0.0,
            )
        )

    return biquads





# ---------------------------------------------------------------------
# Butterworth (ZPK version)
# ---------------------------------------------------------------------

def butterworth_analog_lowpass_zpk(order: int):
    """Analog Butterworth prototype (ωc = 1 rad/s)."""
    if order < 1:
        raise ValueError("Butterworth order must be >= 1.")
    z, p, k = butterap(order)
    return z, p, k


def butterworth_lowpass_stages_zpk(order: int, f0: float, fs: float) -> List[BiquadState]:
    """
    Digital Butterworth low-pass via analog prototype + bilinear transform.
    More rigorous version than the Q-based one.
    """
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist.")
    z_a, p_a, k_a = butterworth_analog_lowpass_zpk(order)

    # Prewarp to desired cutoff
    w_pre = 2.0 * fs * np.tan(np.pi * f0 / fs)

    # Scale poles/zeros to frequency
    z_scaled = [w_pre * zz for zz in z_a]
    p_scaled = [w_pre * pp for pp in p_a]

    # Bilinear transform
    z_d = [(2.0 * fs + zz) / (2.0 * fs - zz) for zz in z_scaled]
    p_d = [(2.0 * fs + pp) / (2.0 * fs - pp) for pp in p_scaled]

    # DC gain normalization
    num_dc = np.prod([1.0 - zz for zz in z_d]) if z_d else 1.0
    den_dc = np.prod([1.0 - pp for pp in p_d])
    k_d = (den_dc / num_dc).real * k_a

    return zpk_to_biquad_states(z_d, p_d, k_d, fs, mode_label="butterworth_lowpass")

# ---------------------------------------------------------------------
# Chebyshev Type I Utilities
# ---------------------------------------------------------------------

def chebyshev1_poles(order: int, ripple_db: float):
    """Compute normalized analog poles of a Chebyshev I lowpass."""
    if order % 2 != 0 or order < 2:
        raise ValueError("Chebyshev I order must be even and >=2.")
    eps = np.sqrt(10**(ripple_db / 10.0) - 1.0)
    sinh_asinh = np.sinh((1.0 / order) * np.arcsinh(1.0 / eps))
    cosh_asinh = np.cosh((1.0 / order) * np.arcsinh(1.0 / eps))

    poles = []
    for k in range(1, order + 1):
        theta = np.pi * (2 * k - 1) / (2 * order)
        sigma = sinh_asinh * np.sin(theta)
        omega = cosh_asinh * np.cos(theta)
        if sigma > 0:
            poles.append(-sigma + 1j * omega)
    return poles


def chebyshev1_Qs(order: int, ripple_db: float):
    """Return list of (Q, ω0) for each biquad of a Chebyshev I LPF."""
    poles = chebyshev1_poles(order, ripple_db)
    Qs = []
    for p in poles:
        sigma = -p.real
        omega = abs(p.imag)
        Q = 1.0 / (2.0 * sigma)
        w0 = np.sqrt(sigma**2 + omega**2)
        Qs.append((Q, w0))
    return Qs


def chebyshev1_lowpass_stages(order: int, ripple_db: float, f0: float, fs: float):
    """Create a Chebyshev I low-pass RBJ cascade."""
    Qw = chebyshev1_Qs(order, ripple_db)
    stages = []
    for Q, _ in Qw:
        stages.append(biquad_init("lowpass", f0, fs, Q=Q))
    return stages

# ---------------------------------------------------------------------
# Chebyshev Type I Low-Pass (rigorous ZPK → BLT)
# ---------------------------------------------------------------------

def cheby1_analog_lowpass_zpk(order: int, ripple_db: float):
    """
    Normalized analog Chebyshev-I low-pass prototype (ωc = 1 rad/s).

    Returns (z, p, k) in s-plane:
      - z: []
      - p: poles in LHP
      - k: gain for |H(j0)| = 1 (unity at DC).
    """
    if order < 1:
        raise ValueError("Chebyshev I order must be >= 1.")
    if ripple_db <= 0:
        raise ValueError("Passband ripple (dB) must be > 0.")

    N = order
    eps = np.sqrt(10.0**(ripple_db / 10.0) - 1.0)

    mu = np.arcsinh(1.0 / eps) / N
    sinh_mu = np.sinh(mu)
    cosh_mu = np.cosh(mu)

    poles: List[complex] = []
    for k in range(1, N + 1):
        theta = np.pi * (2 * k - 1) / (2 * N)
        sigma = -sinh_mu * np.sin(theta)
        omega = cosh_mu * np.cos(theta)
        p = sigma + 1j * omega
        if p.real < 0:
            poles.append(p)

    # Analog gain for unity at ω=0:
    # H(s) = k / Π (s - p_k), so H(0) = k / Π (-p_k) = 1 → k = Π (-p_k)
    den0 = np.prod([-p for p in poles])
    k = den0.real

    return [], poles, k


def cheby1_lowpass_stages(order: int, ripple_db: float, f0: float, fs: float) -> List[BiquadState]:
    """
    Design digital Chebyshev-I low-pass:
      1) analog prototype (ωc=1)
      2) prewarp to f0
      3) bilinear transform
      4) normalize gain at DC
      5) factor to biquads
    """
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist.")
    if ripple_db <= 0:
        raise ValueError("Passband ripple must be > 0 dB.")

    z_a, p_a, _k_a = cheby1_analog_lowpass_zpk(order, ripple_db)

    # Prewarp
    w_pre = 2.0 * fs * np.tan(np.pi * f0 / fs)

    # Frequency scale analog poles
    z_scaled = [w_pre * zz for zz in z_a]
    p_scaled = [w_pre * pp for pp in p_a]

    # Bilinear transform: s = 2fs (1 - z^-1)/(1 + z^-1)
    # Inverse mapping for roots: z = (2fs + s) / (2fs - s)
    z_d = [(2.0 * fs + zz) / (2.0 * fs - zz) for zz in z_scaled]
    p_d = [(2.0 * fs + pp) / (2.0 * fs - pp) for pp in p_scaled]

    # Gain normalization in digital domain: enforce H(z=1) = 1
    num_dc = np.prod([1.0 - zz for zz in z_d]) if z_d else 1.0
    den_dc = np.prod([1.0 - pp for pp in p_d])
    k_d = (den_dc / num_dc).real

    return zpk_to_biquad_states(z_d, p_d, k_d, fs, mode_label="cheby1_lowpass")

# ---------------------------------------------------------------------
# Chebyshev Type II (Inverse Chebyshev) Low-Pass
# ---------------------------------------------------------------------
# Based on standard analog prototype formulas:
# - Equiripple stopband
# - Monotone passband
# - ε from stopband attenuation
#
# Reference formulas: poles/zeros relations and epsilon from stopband
# attenuation as in standard Chebyshev II definitions. :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------------------

def cheby2_analog_lowpass_zpk(order: int, stopband_db: float):
    """
    Normalized analog Type-II Chebyshev low-pass prototype (ω0 = 1).

    Returns (z, p, k) in s-plane.
    """
    if order % 2 != 0 or order < 2:
        raise ValueError("Chebyshev II order must be even and >= 2.")
    if stopband_db <= 0:
        raise ValueError("stopband_db must be > 0.")

    N = order
    # ε from minimum stopband attenuation (γ in dB)
    # ε = 1 / sqrt(10^{γ/10} - 1)
    eps = 1.0 / np.sqrt(10.0 ** (stopband_db / 10.0) - 1.0)

    alpha = (1.0 / N) * np.arcsinh(1.0 / eps)
    sinh_a = np.sinh(alpha)
    cosh_a = np.cosh(alpha)

    zeros: List[complex] = []
    poles: List[complex] = []

    # Zeros on jΩ axis: Ω_zm = 1 / cos((2m-1)π/2N)
    for m in range(1, N + 1):
        theta_m = np.pi * (2 * m - 1) / (2 * N)
        cz = np.cos(theta_m)
        if abs(cz) < 1e-12:
            continue
        wz = 1.0 / cz
        zeros.append(1j * wz)

    # Poles: from inverse of Chebyshev-I-like structure (per standard formula)
    for m in range(1, N + 1):
        theta_m = np.pi * (2 * m - 1) / (2 * N)
        sigma = sinh_a * np.sin(theta_m)
        omega = cosh_a * np.cos(theta_m)

        # 1/s_pm^± = ±sigma + j*omega
        for sign in (+1, -1):
            u = sign * sigma + 1j * omega
            s = 1.0 / u
            if s.real < 0:
                poles.append(s)

    # We only want N left-half-plane poles
    # (The loop can generate duplicates numerically, so trim if needed.)
    poles = poles[:N]

    # Gain k chosen so that |H(j0)| = 1 (unity DC gain)
    # For low-pass, DC corresponds to s -> 0 → in z-domain z -> 1.
    # But here still analog prototype (ω0=1). DC gain = k * Π(-z_i) / Π(-p_i).
    num_dc = np.prod([-z for z in zeros]) if zeros else 1.0
    den_dc = np.prod([-p for p in poles])
    k = (den_dc / num_dc).real

    return zeros, poles, k


def cheby2_lowpass_stages(order: int, stopband_db: float, f0: float, fs: float) -> List[BiquadState]:
    """
    Design digital Chebyshev-II low-pass via:
      1. Analog prototype (ω0 = 1)
      2. Frequency prewarp to desired f0
      3. Bilinear transform
      4. ZPK → SOS biquad states

    Returns list of BiquadState suitable for cascade_block().
    """
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist.")
    z_a, p_a, k_a = cheby2_analog_lowpass_zpk(order, stopband_db)

    # Prewarp desired cutoff
    w_pre = 2.0 * fs * np.tan(np.pi * f0 / fs)

    # Scale analog prototype to desired cutoff
    z_scaled = [w_pre * zz for zz in z_a]
    p_scaled = [w_pre * pp for pp in p_a]

    # Bilinear transform: s = 2fs * (1 - z^-1)/(1 + z^-1)
    # Inverse mapping for each analog root:
    # z = (2fs + s) / (2fs - s)
    z_d = [(2.0 * fs + zz) / (2.0 * fs - zz) for zz in z_scaled]
    p_d = [(2.0 * fs + pp) / (2.0 * fs - pp) for pp in p_scaled]

    # Digital gain: enforce H(z=1) = 1 (unity DC)
    num = np.prod([1.0 - zz for zz in z_d]) if z_d else 1.0
    den = np.prod([1.0 - pp for pp in p_d])
    k_d = (den / num).real * k_a

    # Convert to RBJ-compatible biquad states
    return zpk_to_biquad_states(z_d, p_d, k_d, fs, mode_label="cheby2_lowpass")

# ---------------------------------------------------------------------
# Elliptic (Cauer) Low-Pass Filter
# ---------------------------------------------------------------------
# Uses scipy.signal.ellipap for analog prototype, then prewarps,
# bilinear-transforms, and normalizes DC gain.
# ---------------------------------------------------------------------


def elliptic_analog_lowpass_zpk(order: int, ripple_db: float, stopband_db: float):
    """
    Get normalized analog elliptic low-pass prototype poles/zeros/gain.

    Parameters
    ----------
    order : int
        Filter order (number of poles).
    ripple_db : float
        Maximum ripple in the passband (dB).
    stopband_db : float
        Minimum attenuation in the stopband (dB).

    Returns
    -------
    z, p, k : analog zeros, poles, and gain (ωc = 1 rad/s)
    """
    if order < 1:
        raise ValueError("Elliptic filter order must be >= 1.")
    if ripple_db <= 0 or stopband_db <= 0:
        raise ValueError("Ripple and stopband attenuation must be > 0 dB.")
    z, p, k = ellipap(order, ripple_db, stopband_db)
    return z, p, k


def elliptic_lowpass_stages(
    order: int, ripple_db: float, stopband_db: float, f0: float, fs: float
) -> List[BiquadState]:
    """
    Design digital elliptic (Cauer) low-pass filter via:
      1. analog prototype (ω0 = 1)
      2. frequency prewarp to desired f0
      3. bilinear transform
      4. DC gain normalization
      5. ZPK → biquad states
    """
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist.")

    z_a, p_a, k_a = elliptic_analog_lowpass_zpk(order, ripple_db, stopband_db)

    # Prewarp
    w_pre = 2.0 * fs * np.tan(np.pi * f0 / fs)

    # Scale analog roots
    z_scaled = [w_pre * zz for zz in z_a]
    p_scaled = [w_pre * pp for pp in p_a]

    # Bilinear transform
    z_d = [(2.0 * fs + zz) / (2.0 * fs - zz) for zz in z_scaled]
    p_d = [(2.0 * fs + pp) / (2.0 * fs - pp) for pp in p_scaled]

    # Normalize DC gain (unity at z=1)
    num_dc = np.prod([1.0 - zz for zz in z_d]) if z_d else 1.0
    den_dc = np.prod([1.0 - pp for pp in p_d])
    k_d = (den_dc / num_dc).real * k_a

    return zpk_to_biquad_states(z_d, p_d, k_d, fs, mode_label="elliptic_lowpass")

# ---------------------------------------------------------------------
# Bessel (Thomson) Low-Pass Filter
# ---------------------------------------------------------------------
# Uses scipy.signal.besselap for analog prototype, then prewarps,
# bilinear-transforms, and normalizes DC gain.
# ---------------------------------------------------------------------


def bessel_analog_lowpass_zpk(order: int):
    """
    Get normalized analog Bessel (Thomson) low-pass prototype poles/zeros/gain.

    Parameters
    ----------
    order : int
        Filter order (number of poles).

    Returns
    -------
    z, p, k : analog zeros, poles, and gain (ωc = 1 rad/s)
    """
    if order < 1:
        raise ValueError("Bessel filter order must be >= 1.")
    z, p, k = besselap(order)
    return z, p, k


def bessel_lowpass_stages(order: int, f0: float, fs: float) -> List[BiquadState]:
    """
    Design digital Bessel (Thomson) low-pass filter via:
      1. analog prototype (ωc = 1)
      2. frequency prewarp to desired f0
      3. bilinear transform
      4. DC gain normalization
      5. ZPK → biquad states
    """
    if not (0 < f0 < fs / 2):
        raise ValueError("f0 must be between 0 and Nyquist.")

    z_a, p_a, k_a = bessel_analog_lowpass_zpk(order)

    # Prewarp
    w_pre = 2.0 * fs * np.tan(np.pi * f0 / fs)

    # Scale analog roots
    z_scaled = [w_pre * zz for zz in z_a]
    p_scaled = [w_pre * pp for pp in p_a]

    # Bilinear transform: s → (2fs)(1 − z⁻¹)/(1 + z⁻¹)
    z_d = [(2.0 * fs + zz) / (2.0 * fs - zz) for zz in z_scaled]
    p_d = [(2.0 * fs + pp) / (2.0 * fs - pp) for pp in p_scaled]

    # Normalize DC gain
    num_dc = np.prod([1.0 - zz for zz in z_d]) if z_d else 1.0
    den_dc = np.prod([1.0 - pp for pp in p_d])
    k_d = (den_dc / num_dc).real * k_a

    return zpk_to_biquad_states(z_d, p_d, k_d, fs, mode_label="bessel_lowpass")


# ---------------------------------------------------------------------
# Linkwitz–Riley Utilities
# ---------------------------------------------------------------------

def linkwitz_riley_stages(
    mode: Literal["lowpass", "highpass"],
    butter_order: int,
    f0: float,
    fs: float,
) -> List[BiquadState]:
    """
    Create Linkwitz–Riley stages for the given mode ("lowpass" or "highpass").

    butter_order:
        The underlying Butterworth order (even).
        The final LR acoustic slope is 2 * butter_order.

    Example:
        butter_order=2 -> LR2 (12 dB/oct per branch)
        butter_order=4 -> LR4 (24 dB/oct per branch), etc.
    """
    if mode not in ("lowpass", "highpass"):
        raise ValueError("Linkwitz–Riley mode must be 'lowpass' or 'highpass'.")

    Qs = butterworth_Qs(butter_order)

    stages: List[BiquadState] = []

    # LR = cascade of two identical Butterworth filters
    for _ in range(2):
        for q in Qs:
            stages.append(biquad_init(mode, f0, fs, Q=q))

    return stages


def linkwitz_riley_block(states: List[BiquadState], x: np.ndarray) -> Tuple[np.ndarray, List[BiquadState]]:
    """Process a signal through a Linkwitz–Riley cascade."""
    return cascade_block(states, x)


# ---------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------

def compute_freq_response(b0, b1, b2, a1, a2, fs: float, n_fft: int = 2048):
    w = np.linspace(0, np.pi, n_fft)
    z = np.exp(1j * w)
    H = (b0 + b1 / z + b2 / (z ** 2)) / (1 + a1 / z + a2 / (z ** 2))
    f = w * fs / (2 * np.pi)
    return f, H


def plot_single_biquad_response(state: BiquadState, title: str):
    b0, b1, b2, a1, a2 = [state[k] for k in ("b0", "b1", "b2", "a1", "a2")]
    fs = state["fs"]
    f, H = compute_freq_response(b0, b1, b2, a1, a2, fs, 4096)
    mag_db = 20 * np.log10(np.maximum(np.abs(H), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.semilogx(f, mag_db)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cascade_response(states: List[BiquadState], title: str):
    fs = states[0]["fs"]
    w = np.linspace(0, np.pi, 4096)
    z = np.exp(1j * w)
    H_total = np.ones_like(w, dtype=complex)

    for s in states:
        b0, b1, b2, a1, a2 = [s[k] for k in ("b0", "b1", "b2", "a1", "a2")]
        H = (b0 + b1 / z + b2 / (z ** 2)) / (1 + a1 / z + a2 / (z ** 2))
        H_total *= H

    f = w * fs / (2 * np.pi)
    mag_db = 20 * np.log10(np.maximum(np.abs(H_total), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.semilogx(f, mag_db)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Example Demos
# ---------------------------------------------------------------------

def example_lowpass():
    fs = 48000
    f0 = 2000.0
    st = biquad_init("lowpass", f0, fs, Q=0.707)
    plot_single_biquad_response(st, f"RBJ Lowpass (f0={f0} Hz, Q=0.707)")


def example_highpass():
    fs = 48000
    f0 = 2000.0
    st = biquad_init("highpass", f0, fs, Q=0.707)
    plot_single_biquad_response(st, f"RBJ Highpass (f0={f0} Hz, Q=0.707)")


def example_bandpass():
    fs = 48000
    f0 = 1000.0
    st = biquad_init("bandpass", f0, fs, Q=5.0)
    plot_single_biquad_response(st, f"RBJ Bandpass (f0={f0} Hz, Q=5)")


def example_notch():
    fs = 48000
    f0 = 1000.0
    st = biquad_init("notch", f0, fs, Q=10.0)
    plot_single_biquad_response(st, f"RBJ Notch (f0={f0} Hz, Q=10)")


def example_peak():
    fs = 48000
    f0 = 1000.0
    st = biquad_init("peak", f0, fs, Q=2.0, gain_db=6.0)
    plot_single_biquad_response(st, f"RBJ Peaking EQ (+6 dB @ {f0} Hz, Q=2)")


def example_lowshelf():
    fs = 48000
    f0 = 500.0
    st = biquad_init("lowshelf", f0, fs, gain_db=6.0)
    plot_single_biquad_response(st, f"RBJ Low Shelf (+6 dB below {f0} Hz)")


def example_highshelf():
    fs = 48000
    f0 = 5000.0
    st = biquad_init("highshelf", f0, fs, gain_db=-6.0)
    plot_single_biquad_response(st, f"RBJ High Shelf (−6 dB above {f0} Hz)")


def example_filter_noise():
    """Apply an RBJ lowpass to white noise and show spectra."""
    fs = 48000
    f0 = 2000.0
    N = fs // 2

    x = np.random.randn(N)
    st = biquad_init("lowpass", f0, fs, Q=0.707)
    y, _ = biquad_block(st, x)

    win = np.hanning(N)

    f = np.fft.rfftfreq(N, 1.0 / fs)
    mag_x = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(x * win)), 1e-9))
    mag_y = 20 * np.log10(np.maximum(np.abs(np.fft.rfft(y * win)), 1e-9))

    plt.figure(figsize=(8, 3))
    plt.semilogx(f, mag_x, label="Input Noise")
    plt.semilogx(f, mag_y, label="Filtered Output")
    plt.title("RBJ Lowpass on White Noise")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def example_6th_order_butterworth():
    """6th-order Butterworth via RBJ cascade."""
    fs = 48000
    f0 = 2000.0
    order = 6
    stages = butterworth_lowpass_stages(order, f0, fs)
    plot_cascade_response(stages, f"{order}th-Order Butterworth Low-Pass (RBJ cascade)")


def example_nth_order_butterworth():
    """Generic Nth-order Butterworth using auto Qs."""
    fs = 48000
    f0 = 2000.0
    order = int(input("Enter even Butterworth order (e.g., 4, 6, 8): ").strip())
    stages = butterworth_lowpass_stages(order, f0, fs)
    plot_cascade_response(stages, f"{order}th-Order Butterworth Low-Pass (RBJ cascade)")


def example_linkwitz_riley_crossover():
    """LR crossover magnitude + phase + sum check."""
    from numpy import unwrap, angle

    fs = 48000
    f0 = 2000.0
    butter_order = int(input("Enter base Butterworth order for LR (e.g., 2 or 4): ").strip())

    lp_stages = linkwitz_riley_stages("lowpass", butter_order, f0, fs)
    hp_stages = linkwitz_riley_stages("highpass", butter_order, f0, fs)

    w = np.linspace(0, np.pi, 4096)
    z = np.exp(1j * w)
    H_lp = np.ones_like(w, dtype=complex)
    H_hp = np.ones_like(w, dtype=complex)

    for s in lp_stages:
        b0, b1, b2, a1, a2 = [s[k] for k in ("b0", "b1", "b2", "a1", "a2")]
        H_lp *= (b0 + b1 / z + b2 / (z ** 2)) / (1 + a1 / z + a2 / (z ** 2))

    for s in hp_stages:
        b0, b1, b2, a1, a2 = [s[k] for k in ("b0", "b1", "b2", "a1", "a2")]
        H_hp *= (b0 + b1 / z + b2 / (z ** 2)) / (1 + a1 / z + a2 / (z ** 2))

    f = w * fs / (2 * np.pi)
    mag_lp = 20 * np.log10(np.maximum(np.abs(H_lp), 1e-9))
    mag_hp = 20 * np.log10(np.maximum(np.abs(H_hp), 1e-9))
    mag_sum = 20 * np.log10(np.maximum(np.abs(H_lp + H_hp), 1e-9))

    phase_lp = unwrap(angle(H_lp)) * 180.0 / np.pi
    phase_hp = unwrap(angle(H_hp)) * 180.0 / np.pi
    phase_diff = phase_lp - phase_hp

    plt.figure(figsize=(8, 7))

    # Magnitude
    plt.subplot(2, 1, 1)
    plt.semilogx(f, mag_lp, label="LP")
    plt.semilogx(f, mag_hp, label="HP")
    plt.semilogx(f, mag_sum, "--", label="Sum")
    plt.title(f"Linkwitz–Riley (Butterworth {butter_order}) @ {f0:.0f} Hz")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    # Phase
    plt.subplot(2, 1, 2)
    plt.semilogx(f, phase_lp, label="LP phase")
    plt.semilogx(f, phase_hp, label="HP phase")
    plt.semilogx(f, phase_diff, "--", label="Phase diff")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (deg)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print check at crossover
    idx_cross = np.argmin(np.abs(f - f0))
    print(f"\nAt {f[idx_cross]:.2f} Hz:")
    print(f"  LP phase     = {phase_lp[idx_cross]:.2f}°")
    print(f"  HP phase     = {phase_hp[idx_cross]:.2f}°")
    print(f"  Phase diff   = {phase_diff[idx_cross]:.2f}°")
    print(f"  Sum magnitude= {mag_sum[idx_cross]:.2f} dB (target ≈ 0 dB)")


def example_lr_time_domain():
    """Time-domain LR test: LP + HP should reconstruct input."""
    fs = 48000
    f0 = 2000.0
    butter_order = 4  # decent slope
    N = fs  # 1 second of noise

    x = np.random.randn(N)

    lp_stages = linkwitz_riley_stages("lowpass", butter_order, f0, fs)
    hp_stages = linkwitz_riley_stages("highpass", butter_order, f0, fs)

    y_lp, _ = linkwitz_riley_block(lp_stages, x)
    y_hp, _ = linkwitz_riley_block(hp_stages, x)
    y_sum = y_lp + y_hp

    err = y_sum - x
    rms_in = np.sqrt(np.mean(x ** 2))
    rms_err = np.sqrt(np.mean(err ** 2))
    rel_db = 20 * np.log10(max(rms_err / (rms_in + 1e-15), 1e-15))

    print(f"LR time-domain reconstruction error: {rms_err:.3e} (relative {rel_db:.2f} dB)")

    t = np.arange(0, min(N, 2000)) / fs
    plt.figure(figsize=(8, 3))
    plt.plot(t, x[: len(t)], label="Input", alpha=0.5)
    plt.plot(t, y_sum[: len(t)], label="LP+HP", alpha=0.7)
    plt.title("Linkwitz–Riley Time-Domain Reconstruction (zoom)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def example_chebyshev1_lowpass():
    fs = 48000.0
    f0 = 2000.0
    order = int(input("Enter Chebyshev-I order (e.g., 4, 6, 8): ").strip())
    ripple_db = float(input("Enter passband ripple in dB (e.g., 1.0): ").strip())

    stages = cheby1_lowpass_stages(order, ripple_db, f0, fs)
    plot_cascade_response(
        stages,
        f"Chebyshev-I {order}-Order Low-Pass ({ripple_db} dB ripple) @ {int(f0)} Hz",
    )

def example_chebyshev2_lowpass():
    fs = 48000.0
    f0 = 2000.0
    order = int(input("Enter even Chebyshev-II order (e.g., 4, 6, 8): ").strip())
    stop_db = float(input("Enter min stopband attenuation in dB (e.g., 40): ").strip())

    stages = cheby2_lowpass_stages(order, stop_db, f0, fs)
    plot_cascade_response(
        stages,
        f"Chebyshev-II {order}-Order Low-Pass ({stop_db} dB stopband) @ {int(f0)} Hz",
    )

def example_elliptic_lowpass():
    fs = 48000.0
    f0 = 2000.0
    order = int(input("Enter Elliptic order (e.g., 4, 6, 8): ").strip())
    ripple_db = float(input("Enter passband ripple (dB, e.g., 1.0): ").strip())
    stopband_db = float(input("Enter min stopband attenuation (dB, e.g., 60): ").strip())

    stages = elliptic_lowpass_stages(order, ripple_db, stopband_db, f0, fs)
    plot_cascade_response(
        stages,
        f"Elliptic {order}-Order Low-Pass ({ripple_db} dB ripple, {stopband_db} dB stopband) @ {int(f0)} Hz",
    )

def example_bessel_lowpass():
    fs = 48000.0
    f0 = 2000.0
    order = int(input("Enter Bessel order (e.g., 2, 4, 6, 8): ").strip())

    stages = bessel_lowpass_stages(order, f0, fs)
    plot_cascade_response(stages, f"Bessel {order}-Order Low-Pass @ {int(f0)} Hz")

def example_butterworth_lowpass_zpk():
    fs = 48000.0
    f0 = 2000.0
    order = int(input("Enter Butterworth order (e.g., 4, 6, 8): ").strip())

    stages = butterworth_lowpass_stages_zpk(order, f0, fs)
    plot_cascade_response(stages, f"Butterworth {order}-Order Low-Pass (ZPK, precise) @ {int(f0)} Hz")

def example_hpeq_butterworth_rbj():
    fs = 48000.0
    f0 = 2000.0
    order = 6
    gain_db = 6.0

    # Use the new, robust higher-order RBJ EQ designer
    stages = design_hpeq_butterworth_rbj(order, fs, f0, gain_db)

    plot_cascade_response(
        stages,
        f"RBJ High-Order Butterworth-like PEQ: N={order}, +{gain_db} dB @ {int(f0)} Hz"
    )

# ---------------------------------------------------------------------
# Menu
# ---------------------------------------------------------------------

def main():
    print("\nRBJ + Butterworth + Linkwitz–Riley Demos")
    print("----------------------------------------")
    print(" 1: RBJ Lowpass")
    print(" 2: RBJ Highpass")
    print(" 3: RBJ Bandpass")
    print(" 4: RBJ Notch")
    print(" 5: RBJ Peaking EQ")
    print(" 6: RBJ Low Shelf")
    print(" 7: RBJ High Shelf")
    print(" 8: RBJ Lowpass on White Noise")
    print(" 9: 6th-Order Butterworth Lowpass (RBJ cascade)")
    print("10: Nth-Order Butterworth Lowpass (auto-Q)")
    print("11: Linkwitz–Riley Crossover (mag + phase)")
    print("12: Linkwitz–Riley Time-Domain Test")
    print("13: Chebyshev-I Lowpass")
    print("14: Chebyshev-II Lowpass")
    print("15: Elliptic Lowpass")
    print("16: Bessel Lowpass")
    print("17: Butterworth Lowpass (ZPK precise)")
    print("18: High-Order Parametric EQ (RBJ-based)")



    choice = input("Select example (1–17): ").strip()

    if choice == "1":
        example_lowpass()
    elif choice == "2":
        example_highpass()
    elif choice == "3":
        example_bandpass()
    elif choice == "4":
        example_notch()
    elif choice == "5":
        example_peak()
    elif choice == "6":
        example_lowshelf()
    elif choice == "7":
        example_highshelf()
    elif choice == "8":
        example_filter_noise()
    elif choice == "9":
        example_6th_order_butterworth()
    elif choice == "10":
        example_nth_order_butterworth()
    elif choice == "11":
        example_linkwitz_riley_crossover()
    elif choice == "12":
        example_lr_time_domain()
    elif choice == "13":
        example_chebyshev1_lowpass()
    elif choice == "14":
        example_chebyshev2_lowpass()
    elif choice == "15":
        example_elliptic_lowpass()
    elif choice == "16":
        example_bessel_lowpass()
    elif choice == "17":
        example_butterworth_lowpass_zpk()
    elif choice == "18":
        example_hpeq_butterworth_rbj()
    else:
        print("Invalid choice.")
        sys.exit(1)


if __name__ == "__main__":
    main()

```
