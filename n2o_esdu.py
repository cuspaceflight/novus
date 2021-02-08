"""N2O data."""

import numpy as np


def thermophys(temp):
    """Get N2O data at a given temperature.

    Returns:
        - N2O liquid density
        - vapour density
        - latent heat of vaporization
        - dynamic viscosity
        - vapour pressure for input temperature.
    Uses polynomials from ESDU sheet 91022. All units SI.
    """
    # Some handy definitions
    # ...I don't know what to call these properly
    T0 = temp / 309.57
    T0_RECIP = 1 / T0
    T0_INV = 1 - T0

    lden = 452 * np.exp(+ 1.72328 * pow(T_INV, 1/3)
                        - 0.83950 * pow(T_INV, 2/3)
                        + 0.51060 * T_INV
                        - 0.10412 * pow(T_INV, 4/3))

    vden = 452 * np.exp(- 1.009000 * pow(T_RECIP - 1, 1/3)
                        - 6.287920 * pow(T_RECIP - 1, 2/3)
                        + 7.503320 * (T_RECIP - 1)
                        - 7.904630 * pow(T_RECIP - 1, 4/3)
                        + 0.629427 * pow(T_RECIP - 1, 5/3))

    hl = 1000 * (- 200
                 + 116.043 * pow(T_INV, 1/3)
                 - 917.225 * pow(T_INV, 2/3)
                 + 794.779 * T_INV
                 - 589.587 * pow(T_INV, 4/3))

    hg = 1000 * (- 200
                 + 440.055 * pow(T_INV, 1/3)
                 - 459.701 * pow(T_INV, 2/3)
                 + 434.081 * T_INV
                 - 485.338 * pow(T_INV, 4/3))

    hv = hg - hl

    c = 1000 * 2.49973 * (1
                          + 0.023454 / T_INV
                          - 3.801360 * T_INV
                          + 13.09450 * pow(T_INV, 2)
                          - 14.51800 * pow(T_INV, 3))

    vpres = 7251000 * np.exp(T_RECIP * (- 6.71893 * T_INV
                                        + 1.35966 * pow(T_INV, 3/2)
                                        - 1.37790 * pow(T_INV, 5/2)
                                        - 4.05100 * pow(T_INV, 5)))

    return lden, vden, hv, c, vpres
