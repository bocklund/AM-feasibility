import os
import time
from enum import Enum
from copy import deepcopy
from typing import Protocol, Dict

import numpy as np
from numpy.typing import ArrayLike
import matplotlib as mpl
import matplotlib.pyplot as plt

from pycalphad import Database, equilibrium, Model, variables as v
from pycalphad.core.calculate import _sample_phase_constitution
from pycalphad.core.utils import point_sample, filter_phases, unpack_components
from pycalphad.core.errors import DofError
from scheil import simulate_scheil_solidification

from ordering_helpers import create_ordering_records, rename_disordered_phases
from feasibility_helpers import _build_composition_list, _build_mass_balanced_grid,  get_amount_disallowed_phases, get_amount_disallowed_phases_scheil
from sampling_helpers import sample_phase_points

import warnings
warnings.filterwarnings('ignore', 'generate_dof')

def _get_indep_comps(comps, num_indep_comps=-1):
    # assumes comps are all string pure elements
    return sorted(set(comps)- {'VA'})[:num_indep_comps]


class PointsGenerator(Protocol):
    def generate_points(self) -> Dict[str, ArrayLike]:
        "Return a dictionary of points"

class DefaultPointsGenerator(PointsGenerator):
    """Create empty points (let pycalphad/scheil do what they do)"""
    def generate_points(self) -> Dict[str, ArrayLike]:
        return {}


class PhaseConstitutionPointsGenerator(PointsGenerator):
    """Use pycalphad's _sample_phase_constitution to create points dictionaries"""
    def __init__(self, dbf, comps, phases, model=None, pdens=50):
        self.dbf = dbf
        self.comps = comps
        self.phases = filter_phases(dbf, unpack_components(dbf, comps), phases)
        self.model_cls = model or Model
        self.pdens = pdens

    def generate_points(self) -> Dict[str, ArrayLike]:
        "Return a dictionary of points"
        points_dict = {}
        for phase_name in self.phases:
            mod = self.model_cls(self.dbf, self.comps, phase_name)
            # TODO: is this needed?
            num_nonvacant_sublattices = len([subl for subl in mod.constituents if len(subl - {v.Species('VA')}) > 0])
            if num_nonvacant_sublattices > 1:
                points_dict[phase_name] = _sample_phase_constitution(mod, point_sample, True, self.pdens)
        return points_dict


class EquilibriumSamplingPointsGenerator(PointsGenerator):
    """Sample from single phase equilibria"""
    def __init__(self, dbf, comps, phases, conditions_override=None, indep_comps=None, calc_pdens=200, local_pdens=2):
        self.dbf = dbf
        self.comps = comps
        if indep_comps is None:
            indep_comps = _get_indep_comps(comps)
        self.indep_comps = indep_comps
        self.phases = filter_phases(dbf, unpack_components(dbf, comps), phases)
        self.conditions_override = conditions_override or {}
        self.calc_pdens = calc_pdens
        self.local_pdens = local_pdens

    def generate_points(self) -> Dict[str, ArrayLike]:
        "Return a dictionary of points"
        points_dict = {}
        composition = {v.X(ic): (0, 1, 0.01) for ic in self.indep_comps}
        conds = {v.P: 101325, v.N: 1, v.T: 1000.0, **composition}
        conds.update(self.conditions_override)
        for phase_name in self.phases:
            points_dict[phase_name] = sample_phase_points(self.dbf, self.comps, phase_name, conds, self.calc_pdens, self.local_pdens)
        return points_dict


def run_simulation(dbf, comps, phases, eq_statevars, ngridpts, T_liquid, indep_comps=None, points_generator=None, show_progress=True, scheil_kwargs=None, equilibrium_kwargs=None):
    if indep_comps is None:
        indep_comps = _get_indep_comps(comps)
    equilibrium_kwargs = deepcopy(equilibrium_kwargs) or {}
    scheil_kwargs = deepcopy(scheil_kwargs) or {}
    scheil_kwargs.setdefault('adaptive', False)

    # Setup simulation:
    #   setup: sampled points
    if points_generator is None:
        points_generator = DefaultPointsGenerator()
    if show_progress:
        print("Generating points... ", end='')
    points_dict = points_generator.generate_points()
    if show_progress:
        print("Done.")
    #   setup: composition grid as a list of point composition conditions
    grid_comps = _build_mass_balanced_grid(len(indep_comps), ngridpts)
    compositions_list = _build_composition_list(indep_comps, grid_comps)
    #   setup: ordering_records to rename ordered partitioned phases to disordered
    ordering_records = create_ordering_records(dbf, comps, phases)

    # Run simulation:
    eq_results = []  # List[xarray.Dataset]
    scheil_results = []  # List[Optional[scheil.SolidificationResult]]
    for num, composition in enumerate(compositions_list):
        if show_progress:
            printable_composition = {kk: round(vv, 3) for kk, vv in composition.items()}
            print(f"({num+1}/{len(compositions_list)}) - {printable_composition} - ", end='')

        # Equilibrium calculation
        conds = {**eq_statevars, **composition}
        tick = time.time()
        eq_res = equilibrium(dbf, comps, phases, conds, calc_opts={'points': points_dict}, **equilibrium_kwargs)
        tock = time.time()
        rename_disordered_phases(eq_res, ordering_records)
        eq_results.append(eq_res)
        # eq_is_feasible = get_amount_disallowed_phases(eq_res, allowed_phases).max() < tolerance_deleterious_phases
        # eq_feas_str = f'({"feas" if eq_is_feasible else "infeas"})'
        eq_feas_str = ''
        if show_progress:
            print(f'Equilibrium time = {tock - tick: 0.2f} s {eq_feas_str}- ', end='')

        # Scheil simulation
        tick = time.time()
        sol_res = simulate_scheil_solidification(dbf, comps, phases, composition, T_liquid, eq_kwargs={'calc_opts': {'points': points_dict}}, **scheil_kwargs)
        tock = time.time()
        scheil_results.append(sol_res)
        # scheil_is_feasible = get_amount_disallowed_phases_scheil(sol_res, allowed_phases) < tolerance_deleterious_phases
        # scheil_feas_str = f'({"feas" if scheil_is_feasible else "infeas"})'
        scheil_feas_str = ''
        if show_progress:
            print(f'Scheil time = {tock - tick: 0.2f} s {scheil_feas_str}')

    if show_progress:
        print('Done simulating')
    return compositions_list, eq_results, scheil_results



def plot_figure(comps, compositions_list, eq_results, scheil_results, allowed_phases, tolerance_deleterious_phases, indep_comps=None, ax=None, scattersize=20, scattermarker='h'):
    # Plot feasibility on a ternary triangular diagram
    assert len(compositions_list) == len(eq_results) == len(scheil_results), f"Lengths must be the same. Got len(compositions_list)={len(compositions_list)}, len(eq_results)={len(eq_results)}, and len(scheil_results)={len(scheil_results)}."
    if indep_comps is None:
        indep_comps = sorted([str(ic)[2:] for ic in compositions_list[0].keys()])
    indep_comp_vars = [v.X(ic) for ic in indep_comps]

    handles = [
        mpl.patches.Patch(facecolor='green'),
        mpl.patches.Patch(facecolor='red'),
        mpl.patches.Patch(facecolor='darkorange'),
        mpl.patches.Patch(facecolor='darkred'),
    ]
    labels = [
        'Feasible',
        'Equilibrium infeasible',
        'Scheil infeasible',
        'Equilibrium and Scheil infeasible',
    ]

    if ax is None:
        ax = plt.figure().add_subplot(projection='triangular')

    for composition, eq_result, scheil_result in zip(compositions_list, eq_results, scheil_results):
        x_plot, y_plot = composition[indep_comp_vars[0]], composition[indep_comp_vars[1]]

        eq_is_feasible = get_amount_disallowed_phases(eq_result, allowed_phases).max() < tolerance_deleterious_phases
        scheil_is_feasible = get_amount_disallowed_phases_scheil(scheil_result, allowed_phases) < tolerance_deleterious_phases

        if eq_is_feasible and scheil_is_feasible:
            ax.scatter(x_plot, y_plot, c='green', s=scattersize, marker=scattermarker)
        elif not eq_is_feasible and scheil_is_feasible:
            ax.scatter(x_plot, y_plot, c='red', s=scattersize, marker=scattermarker)
        elif eq_is_feasible and not scheil_is_feasible:
            ax.scatter(x_plot, y_plot, c='darkorange', s=scattersize, marker=scattermarker)
        else:
            ax.scatter(x_plot, y_plot, c='darkred', s=scattersize, marker=scattermarker)


    fmtted_comps = '-'.join(sorted(set(comps) - {'VA'}))
    ax.set_title(f"{fmtted_comps}\nTolerance: {tolerance_deleterious_phases}")
    ax.set_xlabel(f'X({indep_comps[0]})')
    ax.set_ylabel(f'X({indep_comps[1]})', labelpad=-50)
    ax.figure.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(0.45, 0.5))
    ax.tick_params(labelsize=8)
    return ax
