"""Adapters for class-based eurocodepy APIs.

eurocodepy exposes some data through classes (``Steel``, ``Bolt``,
``ProfilesI``, etc.) rather than plain functions.  These adapters wrap
them into simple ``(**kwargs) -> dict`` callables so the dispatcher can
treat everything uniformly.
"""

from __future__ import annotations


# ── EC3 Profile lookups ───────────────────────────────────────────────


def _section_to_dict(section) -> dict:
    """Extract all numeric/string properties from a SteelSection object."""
    out: dict = {}
    for attr in (
        "Section", "type", "h", "b", "tw", "tf", "r", "m", "P",
        "A", "Av_z", "Av_y",
        "Iy", "iy", "Wel_y", "Wpl_y",
        "Iz", "iz", "Wel_z", "Wpl_z",
        "IT", "WT", "Iw", "Ww",
        "fyd",
        "Npl_Rd", "Vpl_Rd_z", "Vpl_Rd_y",
        "Mel_Rd_y", "Mpl_Rd_y", "Mel_Rd_z", "Mpl_Rd_z",
        "CurveA", "CurveB",
    ):
        val = getattr(section, attr, None)
        if val is not None:
            out[attr] = val
    return out


def lookup_i_profile(*, profile_name: str) -> dict:
    """Look up an I-section profile (IPE/HEA/HEB/HEM)."""
    from eurocodepy.ec3 import ProfilesI

    key = profile_name.upper().replace(" ", "")
    if key not in ProfilesI:
        available = sorted(ProfilesI.keys())
        raise ValueError(
            f"Profile '{profile_name}' not found. "
            f"Available I-sections ({len(available)}): {', '.join(available[:20])}..."
        )
    return _section_to_dict(ProfilesI[key])


def lookup_chs_profile(*, profile_name: str) -> dict:
    """Look up a CHS profile from the eurocodepy CHS database."""
    import eurocodepy as ec

    key = profile_name.upper().replace(" ", "").replace(".", "_")
    for prof in ec.SteelCHSProfiles:
        if prof["Section"].upper() == key:
            return dict(prof)

    names = [p["Section"] for p in ec.SteelCHSProfiles]
    raise ValueError(
        f"CHS profile '{profile_name}' not found. "
        f"Available ({len(names)}): {', '.join(names[:15])}..."
    )


def lookup_rhs_profile(*, profile_name: str) -> dict:
    """Look up an RHS profile."""
    from eurocodepy.ec3 import ProfilesRHS

    key = profile_name.upper().replace(" ", "").replace(".", "_")
    if key not in ProfilesRHS:
        available = sorted(ProfilesRHS.keys())
        raise ValueError(
            f"RHS profile '{profile_name}' not found. "
            f"Available ({len(available)}): {', '.join(available[:15])}..."
        )
    return _section_to_dict(ProfilesRHS[key])


def lookup_shs_profile(*, profile_name: str) -> dict:
    """Look up an SHS profile."""
    from eurocodepy.ec3 import ProfilesSHS

    key = profile_name.upper().replace(" ", "").replace(".", "_")
    if key not in ProfilesSHS:
        available = sorted(ProfilesSHS.keys())
        raise ValueError(
            f"SHS profile '{profile_name}' not found. "
            f"Available ({len(available)}): {', '.join(available[:15])}..."
        )
    return _section_to_dict(ProfilesSHS[key])


# ── EC3 Steel grade ──────────────────────────────────────────────────


def lookup_steel_grade(*, grade: str) -> dict:
    """Look up structural steel properties by grade name."""
    from eurocodepy.ec3 import Steel

    try:
        s = Steel(grade)
    except Exception as e:
        raise ValueError(
            f"Steel grade '{grade}' not recognised: {e}. "
            "Try: S235, S275, S355, S420, S460."
        ) from e

    return {
        "grade": str(s.ClassType),
        "fy_MPa": s.fyk,
        "fy_40mm_MPa": s.fyk40,
        "fu_MPa": s.fuk,
        "fu_40mm_MPa": s.fuk40,
        "E_MPa": s.Es,
        "gamma_M0": s.gamma_M0,
        "gamma_M1": s.gamma_M1,
        "gamma_M2": s.gamma_M2,
    }


# ── EC3 Bolt ─────────────────────────────────────────────────────────


def lookup_bolt(*, diameter: str, grade: str) -> dict:
    """Look up bolt properties."""
    from eurocodepy.ec3 import Bolt

    try:
        b = Bolt(diameter, grade)
    except Exception as e:
        raise ValueError(
            f"Bolt '{diameter}' grade '{grade}' not found: {e}. "
            "Diameters: M12-M36. Grades: 4.6, 4.8, 5.6, 5.8, 6.8, 8.8, 10.9."
        ) from e

    return {
        "name": b.name,
        "grade": b.steel,
        "d_mm": b.d,
        "d0_mm": b.d0,
        "dnut_mm": b.dnut,
        "A_cm2": b.A,
        "Athread_cm2": b.Athread,
        "fub_MPa": b.fub,
        "fyb_MPa": b.fyb,
    }


# ── EC3 Flexural buckling (wraps BucklingParameters + check) ────────


def ec3_flexural_buckling(
    *,
    N_Ed: float,
    A: float,
    fy: float,
    L_cr: float,
    i: float,
    buckling_curve: str = "b",
    gamma_M1: float = 1.0,
) -> dict:
    """Run EC3 flexural buckling check via eurocodepy."""
    from eurocodepy.ec3.uls import BucklingParameters, eurocode3_buckling_check

    params = BucklingParameters(A=A, fy=fy, L_cr=L_cr, i=i)
    result = eurocode3_buckling_check(
        N_Ed=N_Ed,
        params=params,
        buckling_curve=buckling_curve,
        gamma_M1=gamma_M1,
    )
    return dict(result)
