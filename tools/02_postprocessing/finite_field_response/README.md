# Finite-field BEC and electronic polarizability

This directory records the complete, environment-independent ABACUS workflow
for computing Born effective charges (BECs) and the clamped-ion electronic
polarizability tensor by centered finite differences.  It deliberately does
not contain scheduler commands, executable paths, pseudopotential paths, MPI
settings, or machine-specific configuration.

## What is computed

Run seven self-consistent calculations for one fixed structure:

```text
zero
x_minus  x_plus
y_minus  y_plus
z_minus  z_plus
```

For field direction `beta` and Cartesian response component `alpha`, the tools
use the following convention:

```text
polarizability[beta, alpha]
    = 4 pi sqrt(2) [P_alpha(+E_beta) - P_alpha(-E_beta)] / (2 E)

BEC[atom, beta, alpha]
    = [F_atom,alpha(+E_beta) - F_atom,alpha(-E_beta)]
      / [2 E * 13.605693122994 * 1.889725988579 * sqrt(2)]
```

`finite_field_amp` is in the ABACUS finite-field convention, polarization is
read in `e/bohr^2`, force is read in `eV/Angstrom`, polarizability is
dimensionless, and BEC is in units of the elementary charge.  Tensor rows are
the applied-field direction; columns are the measured polarization or force
direction.  This convention matches a row-major extxyz tensor such as
`xx xy xz yx yy yz zx zy zz`.

The `sqrt(2)` factors are part of the current ABACUS finite-field amplitude
conversion.  Do not silently remove them when comparing with another code.

## Important force rule

BECs in this workflow are obtained from nonzero-field force differences.
Therefore, all six nonzero-field calculations **must** use `cal_force 1`.
`cal_stress` is needed only in `zero` if zero-field stress is also wanted.

If only polarizability is requested, the six nonzero-field calculations may
use `cal_force 0`; this faster layout cannot subsequently produce BECs.

## Finite-field stress is not implemented

The current homogeneous finite-field implementation does **not** provide the
complete stress at nonzero electric field.  In particular:

- PW and LCAO both explicitly reject `finite_field 1` together with
  `cal_stress 1`.  Neither implementation includes the derivative of the
  finite-field electric enthalpy with respect to cell strain, so a complete
  finite-field stress is unavailable for either basis.

Consequently, this workflow always sets `cal_stress 0` in all six nonzero-field
cases.  `cal_stress 1` is used only in `zero`, where the ordinary zero-field
stress is well defined and may be compared with reference data.  Do not use
these calculations for piezoelectric response, electrostriction, variable-cell
relaxation under field, or any other property requiring a derivative of the
electric enthalpy with respect to strain.

## 1. Prepare a template

Create a directory containing a valid, converged zero-field ABACUS setup:

```text
template/
  INPUT
  STRU
  KPT
```

The structure, pseudopotentials/orbitals, functional, basis, cutoff, k mesh,
smearing, number of bands, and SCF tolerances must remain identical in all
seven calculations.  Use an insulating structure and disable symmetry unless
the finite-field implementation explicitly supports the desired symmetry.

For LCAO, the recommended overlap backend is currently:

```text
finite_field_lcao_overlap rayleigh_expansion
finite_field_lcao_lmax    6
```

Converge the orbital basis, k mesh, cutoff, field magnitude, SCF threshold, and
Rayleigh `lmax` for the target system.  A good initial field magnitude is
`5e-4`, but it is not universal.  Repeat with half the field and verify that
the response is stable.

For a full LCAO tensor, use an ABACUS revision that reports all three Cartesian
components in each line beginning with
`Finite-field LCAO polarization_e_per_bohr2`.  Older revisions reported only
the active direction and consequently produced zero transverse components.

## 2. Generate the seven cases

For both BEC and polarizability:

```bash
python3 prepare_cases.py template run --field 5e-4 \
  --properties bec,polarizability
```

For polarizability only:

```bash
python3 prepare_cases.py template run --field 5e-4 \
  --properties polarizability
```

The command copies `INPUT`, `STRU`, and `KPT`, then changes only the suffix,
field direction/amplitude, and force/stress switches.  Existing case
directories are refused unless `--overwrite` is supplied.  `--overwrite`
recreates all seven case directories, including removal of stale `OUT.*`
results, so use it only when those old calculations are no longer needed.

Inspect the generated inputs before running ABACUS.  Execute ABACUS once in
each of the seven case directories using any suitable local or batch launcher.
The execution mechanism is intentionally outside this workflow.

## 3. Analyze completed calculations

```bash
python3 analyze_response.py run --output response.json
```

The field magnitude is read from the six generated `INPUT` files and checked
for consistency.  It can also be asserted explicitly:

```bash
python3 analyze_response.py run --field 5e-4 --output response.json
```

The analyzer produces both properties by default and fails if any nonzero-field
force table is absent.  For a deliberately polarization-only calculation, use:

```bash
python3 analyze_response.py run --polarizability-only --output response.json
```

The analyzer requires all seven SCFs to be converged.  Polarization output is
required in the six nonzero-field logs; it is not required in `zero`.  It
writes:

- `response.json`: inputs, convergence, timing, polarization tensor, BECs,
  acoustic sum, and diagnostic warnings;
- `response.bec.csv`: one row per atom, in ABACUS force-output order;
- `response.polarizability.csv`: the 3 x 3 tensor.

ABACUS prints atoms grouped by species in its force table.  If a downstream
dataset uses a different atom order, supply a JSON list mapping each ABACUS
grouped index to its desired zero-based index:

```bash
python3 analyze_response.py run --atom-map grouped_to_target.json
```

For example, `[2, 0, 1]` means grouped atom 0 is written as target atom 2.
The JSON report retains both grouped-order and mapped-order tensors.

## Validation checklist

Before accepting a result, verify all of the following:

1. All seven logs contain `#SCF IS CONVERGED#` and no non-convergence marker.
2. The six nonzero inputs are exact `+E/-E` pairs with identical settings.
3. BEC runs contain one force row per atom in every nonzero-field output.
4. No nonzero-field result is interpreted as a finite-field stress; stress is
   taken only from the `zero` calculation.
5. `max_abs_acoustic_sum_e` is small; a large value signals numerical or
   parsing problems, or an unconverged response.
6. For a nonmagnetic insulating equilibrium structure, compare
   `max_abs_polarizability_antisymmetric` with the desired tolerance.
7. Inspect warnings about polarization branch jumps.
8. Repeat at half field.  A stable derivative rules out finite-field
   nonlinearity but does not establish basis-set convergence.
9. Compare LCAO against the same finite-field algorithm in a converged PW basis
   before attributing differences to the overlap backend.

Do not mix this periodic Berry-phase workflow (`finite_field`) with the slab
sawtooth electric potential (`efield`).
