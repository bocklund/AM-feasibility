"""Run feasibility tests on a ternary system with a plot"""

import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

from pycalphad import Database, equilibrium, Model, variables as v
from pycalphad.core.calculate import _sample_phase_constitution
from pycalphad.core.errors import DofError
from pycalphad.core.utils import point_sample
from scheil import simulate_scheil_solidification
from .feasibility_helpers import _build_composition_list, _build_mass_balanced_grid,  get_amount_disallowed_phases, get_amount_disallowed_phases_scheil



# User settings (edit these):
# Be careful not to edit the variable names, because they are used below.
DATABASE_DIR = '../databases'
dbf = Database(os.path.join(DATABASE_DIR, 'Cr-Ti-V_ghosh2002.tdb'))
comps = ['CR', 'TI', 'V', 'VA']
phases = list(dbf.phases.keys())

T_liquid = 2200  # temperature where everything is liquid
potentials = {v.N: 1, v.T: (1000, T_liquid, 10), v.P: 101325}  # for equilibrium calculations

# indep_comps = [comps[1], comps[2]]  # choose them automatically
indep_comps = ['V', 'TI']
ngridpts = 11  # number of points along each dimension of the composition grid

allowed_phases = ['LIQUID', 'BCC_A2', 'FCC_A1', 'HCP_A3']  # phases that are okay to have (non-deleterious)
tolerance_deleterious_phases = 0.10  # Maximum tolerance for deleterious phases

SAVE_TIME = True  # If true, don't perform more expensive feasibility tests (e.g. Scheil) if the composition is already known to be infeasible (e.g. from equilibrium)
OUTDIR = 'figures'

# Script (should not need editing)
if __name__ == '__main__':
    try:
        os.mkdir(OUTDIR)
    except FileExistsError:
        pass
    plt.style.use('papers.mplstyle')

    # Generate points for adaptive Scheil starting points (performance)
    points_dict = {}
    for phase_name in phases:
        try:
            mod = Model(dbf, comps, phase_name)
            points_dict[phase_name] = _sample_phase_constitution(mod, point_sample, True, 50)
        except DofError:
            pass

    # Build compositions to simulate over
    grid_comps = _build_mass_balanced_grid(len(indep_comps), ngridpts)
    compositions_list = _build_composition_list(indep_comps, grid_comps)

    # Run simulations
    eq_results = []  # List[xarray.Dataset]
    scheil_results = []  # List[Optional[scheil.SolidificationResult]]
    for num, composition in enumerate(compositions_list):
        print(f"{composition} ({num+1}/{len(compositions_list)})")
        # Equilibrium calculation for feasibility
        conds = {v.P: 101325, v.N: 1, v.T: (1000, 2200, 10), **composition}
        tick = time.time()
        eq_res = equilibrium(dbf, comps, phases, conds)
        tock = time.time()
        eq_results.append(eq_res)
        eq_is_feasible = get_amount_disallowed_phases(eq_res, allowed_phases).max() < tolerance_deleterious_phases
        print(f'Equilibrium time: {tock - tick: 0.2f} s - {"feasible" if eq_is_feasible else "infeasible"}')

        if SAVE_TIME and not eq_is_feasible:
            scheil_results.append(None)  # ensure that the shapes of the results line up, even if we don't do this calculation
            continue
        # Scheil
        tick = time.time()
        sol_res = simulate_scheil_solidification(dbf, comps, phases, composition, T_liquid, adaptive=True, eq_kwargs={'calc_opts': {'points': points_dict}})
        tock = time.time()
        scheil_results.append(sol_res)
        scheil_is_feasible = get_amount_disallowed_phases_scheil(sol_res, allowed_phases) < tolerance_deleterious_phases
        print(f'Scheil time: {tock - tick: 0.2f} s - {"feasible" if scheil_is_feasible else "infeasible"}')

    print('Done simulating')

    # Plot feasibility on a ternary triangular diagram
    indep_comp_vars = [v.X(ic) for ic in indep_comps]
    handles = [
        mpl.patches.Patch(facecolor='red'),
        mpl.patches.Patch(facecolor='darkred'),
        mpl.patches.Patch(facecolor='green'),
    ]
    labels = [
        'Equilibrium infeasible',
        'Equilibrium feasible; Scheil infeasible',
        'Feasible',
    ]

    fig = plt.figure()
    ax = fig.add_subplot(projection='triangular')
    for composition, eq_result, scheil_result in zip(compositions_list, eq_results, scheil_results):
        x_plot, y_plot = composition[indep_comp_vars[0]], composition[indep_comp_vars[1]]


        eq_is_feasible = get_amount_disallowed_phases(eq_result, allowed_phases).max() < tolerance_deleterious_phases
        if not eq_is_feasible:
            ax.scatter(x_plot, y_plot, label='Equilibrium infeasible', c='red')
            continue

        assert scheil_result is not None, "Scheil calculation must exist to determine feasibility if equilibrium is feasible."
        scheil_is_feasible = get_amount_disallowed_phases_scheil(scheil_result, allowed_phases) < tolerance_deleterious_phases
        if not scheil_is_feasible:
            ax.scatter(x_plot, y_plot, label='Equilibrium feasible; Scheil infeasible', c='darkred')
            continue

        ax.scatter(x_plot, y_plot, label='Equilibrium and Scheil feasible', c='green')

    fmtted_comps = '-'.join(sorted(set(comps) - {'VA'}))
    ax.set_title(f"{fmtted_comps}\nTolerance: {tolerance_deleterious_phases}")
    ax.set_xlabel(f'X({indep_comps[0]})')
    ax.set_ylabel(f'X({indep_comps[1]})', labelpad=-50)
    fig.legend(handles=handles, labels=labels, loc='lower left', bbox_to_anchor=(0.45, 0.5))
    ax.tick_params(labelsize=8)

    fig.savefig(f'{fmtted_comps}.pdf', bbox_inches='tight')