import numpy as np
from pycalphad.core.utils import generate_dof, unpack_components
from pycalphad import calculate, equilibrium
from scheil.utils import local_sample


def sample_phase_points(dbf, comps, phase_name, conditions, calc_pdens, pdens):
    """Sample new points from a phase around the single phase equilibrium site fractions at the given conditions.
    Parameters
    ----------
    dbf :
    comps :
    phase_name :
    conditions :
    calc_pdens :
        The point density passed to calculate for the nominal points added.
    pdens : int
        The number of points to add in the local sampling at each set of equilibrium site fractions.
    Returns
    -------
    np.ndarray[:,:]
    """
    _, subl_dof = generate_dof(dbf.phases[phase_name], unpack_components(dbf, comps))
    # subl_dof is number of species in each sublattice, e.g. (FE,NI,TI)(FE,NI)(FE,NI,TI) is [3, 2, 3]
    eqgrid = equilibrium(dbf, comps, [phase_name], conditions)
    all_eq_pts = eqgrid.Y.values[eqgrid.Phase.values == phase_name]
    # sample points locally
    additional_points = local_sample(all_eq_pts, subl_dof, pdens)
    # get the grid between endmembers and random point sampling from calculate
    pts_calc = calculate(dbf, comps, phase_name, pdens=calc_pdens, P=101325, T=300, N=1).Y.values.squeeze()
    return np.concatenate([additional_points, np.atleast_2d(pts_calc)], axis=0)
