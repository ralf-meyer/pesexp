from .calculators import (TeraChem, OpenbabelFF, CerjanMillerSurface,
                          AdamsSurface, MuellerBrownSurface)

_xtb_methods = ['xtb']
_openbabel_methods = ['uff', 'mmff94', 'gaff']
_available_methods = _xtb_methods + _openbabel_methods


def get_calculator(method):
    if method.lower() in _xtb_methods:
        from xtb.ase.calculator import XTB
        methods = dict(xtb=('GFN2-xTB', 300.0),
                       gfnff=('GFNFF', 0.0))
        # TODO: can't get GFNFF to work!
        return XTB(method=methods[method][0],
                   electronic_temperature=methods[method][1])
    elif method.lower() in _openbabel_methods:
        return OpenbabelFF(ff=method.upper())


__all__ = ['TeraChem', 'OpenbabelFF', 'CerjanMillerSurface', 'AdamsSurface',
           'MuellerBrownSurface']
