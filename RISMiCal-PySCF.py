#!/usr/bin/env python3
"""
RISMiCal-PySCF.py
An open-source, purely in-memory wrapper for QM/3D-RISM (KSDFT/3D-RISM, 3D-RISM-SCF, QM/MM/3D-RISM) calculations 
using RISMiCal and PySCF. Eliminates heavy file I/O bottlenecks.
"""

import os
import sys
import shutil
import subprocess
import re
import numpy as np
from scipy.spatial.distance import cdist

try:
    from pyscf import gto, dft, qmmm, tdscf # [MODIFIED] Added tdscf import
    from pyscf.tools import cubegen
except ImportError:
    print("[ERROR] PySCF is not installed. Please run: pip install pyscf")
    sys.exit(1)

# ==============================================================================
# Configuration
# ==============================================================================
RISMICAL_CMD = "rismical.x"

# Physical constants for energy and length conversions
HARTREE_TO_JMOL = 2625499.6394799
COULOMB_TO_JMOL = 1389354.56
ANG_TO_BOHR     = 1.8897261246

# Change d to e 
def parse_fortran_float(val_str):
    try: return float(str(val_str).lower().replace('d', 'e'))
    except ValueError: return 0.0

# Make list of atoms in QM part
def parse_qmpart(qmpart_str, total_atoms):
    if not qmpart_str or not str(qmpart_str).strip():
        return list(range(1, total_atoms + 1))
    indices = []
    for p in str(qmpart_str).split(','):
        if not p.strip(): continue
        val = int(p.strip())
        if val > 0:
            indices.append(val)
            last_idx = val
        elif val < 0:
            indices.extend(range(last_idx + 1, abs(val) + 1))
            last_idx = abs(val)
    return sorted(list(set(indices)))

# Extract parameters in namelist
# block_str : namelist name, params : parameters in namelist
def extract_namelist(block_str, params):
    placeholders = []
    def repl(m):
        placeholders.append(m.group(0))
        return f"__QUOTE_{len(placeholders)-1}__"
    masked_str = re.sub(r'"[^"]*"|\'[^\']*\'', repl, block_str)
    matches = list(re.finditer(r'([a-zA-Z_]\w*)\s*=', masked_str))
    for i, match in enumerate(matches):
        key = match.group(1).lower()
        start_idx = match.end()
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(masked_str)
        val_str = masked_str[start_idx:end_idx].strip().rstrip(',')
        for j, p in enumerate(placeholders): val_str = val_str.replace(f"__QUOTE_{j}__", p)
        if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
            val_str = val_str[1:-1]
        params[key] = val_str

# Read input file
def read_input_file(inp_file):
    params, udata, udata_line_indices = {}, [], []
    with open(inp_file, 'r', errors='replace') as f: lines = f.readlines()
    in_rismicalqm, in_grid3d, in_udata = False, False, False
    rismicalqm_block_str, grid3d_block_str = "", ""
    for i, line in enumerate(lines):
        clean_line = line.split('!')[0].strip()
        if not clean_line: continue
        upper_line = clean_line.upper()
        if upper_line.startswith('$RISMICALQM') or upper_line.startswith('&RISMICALQM'):
            in_rismicalqm = True
            header_len = len('$RISMICALQM') if upper_line.startswith('$RISMICALQM') else len('&RISMICALQM')
            rismicalqm_block_str += clean_line[header_len:] + " "
        elif upper_line.startswith('$GRID3D') or upper_line.startswith('&GRID3D'):
            in_grid3d = True
            header_len = len('$GRID3D') if upper_line.startswith('$GRID3D') else len('&GRID3D')
            grid3d_block_str += clean_line[header_len:] + " "
        elif upper_line.startswith('$UDATA') or upper_line.startswith('&UDATA'):
            in_udata = True
        elif in_rismicalqm and (upper_line == '$END' or upper_line == '/'): in_rismicalqm = False
        elif in_grid3d and (upper_line == '$END' or upper_line == '/'): in_grid3d = False
        elif in_udata and (upper_line == '$END' or upper_line == '/'): in_udata = False
        elif in_rismicalqm: rismicalqm_block_str += clean_line + " "
        elif in_grid3d: grid3d_block_str += clean_line + " "
        elif in_udata:
            parts = clean_line.split()
            if len(parts) >= 7: udata.append(parts); udata_line_indices.append(i)
    extract_namelist(rismicalqm_block_str, params)
    extract_namelist(grid3d_block_str, params)
    return params, udata, udata_line_indices, lines

# Build pySCF mol
def build_pyscf_mol(params, udata, qm_indices):
    """ Build the PySCF Mole object purely in memory. """
    mol_str = ""
    for idx in qm_indices:
        atom = udata[idx-1]
        x, y, z = parse_fortran_float(atom[4]), parse_fortran_float(atom[5]), parse_fortran_float(atom[6])
        mol_str += f"{atom[0]} {x} {y} {z}; "
    
    basis  = params.get('basis', '6-31g*')
    charge = int(params.get('charge', '0'))
    spin   = int(params.get('spin', '0'))
    
    mol = gto.Mole()
    mol.atom = mol_str
    mol.basis = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 0  # Suppress PySCF standard output to keep console clean
    mol.build()
    return mol

# [MODIFIED] Extended to handle optional TD-DFT calculations for excited states
# Run pySCF
def run_pyscf_scf(mol, params, ext_coords=None, ext_charges=None):
    """ Run PySCF DFT with optional background charges and optional TD-DFT. """
    xc = params.get('xc', 'b3lyp')
    mf = dft.RKS(mol)
    mf.xc = xc
    
    if ext_coords is not None and len(ext_coords) > 0:
        # qmmm.mm_charge modifies the Hamiltonian to include external point charges
        mf = qmmm.mm_charge(mf, ext_coords, ext_charges)
        
    mf.kernel()
    if not mf.converged:
        print("\n  [ERROR] PySCF SCF calculation did not converge!")
        sys.exit(1)
        
    e_tot = mf.e_tot
    dm = mf.make_rdm1()
    
    # [NEW] Check if excited state calculation is requested (e.g., param 'td' is set)
    # Assumes target root is specified via 'root' parameter, defaults to 1.
    if 'td' in params or 'cis' in params:
        td_obj = tdscf.TDDFT(mf)
        td_obj.nstates = int(params.get('nstates', '3'))
        td_obj.kernel()
        
        target_root = int(params.get('root', '1'))
        if target_root > len(td_obj.e):
            print(f"\n  [ERROR] Requested root {target_root} exceeds computed states {len(td_obj.e)}.")
            sys.exit(1)
            
        # Update total energy: E_GS + Excitation Energy
        e_tot = mf.e_tot + td_obj.e[target_root - 1]
        
        # Calculate unrelaxed density matrix for the excited state
        # Note: PySCF's default TDDFT doesn't natively do Z-vector (relaxed density) easily, 
        # using unrelaxed as an approximation for the potential.
        dm_ex = td_obj.get_abinit_1pdm(target_root)
        # Add transition density correction to ground state density
        dm = mf.make_rdm1() + dm_ex
        
        # Calculate Mulliken charges for the excited state density
        mol_charges = mol.atom_charges()
        qm_charges = mol_charges - np.einsum('ij,ji->i', dm, mf.get_ovlp()).real
    else:
        # Ground state charges
        _, qm_charges = mf.mulliken_pop(verbose=0)
        
    return e_tot, dm, qm_charges

# Calculate electrostatic interaction energy between solvent and MM atoms
def process_qv_and_get_ext_charges(qv_file, udata, qvcutoff, qvcore, mm_indices):
    """ Reads solvent charges from .qv, applies filters, and combines with MM charges for PySCF. """
    ext_c, ext_q = [], []
    e_mv_jmol = 0.0
    
    # 1. Read and filter Solvent Charges
    if os.path.exists(qv_file):
        qv_coords, qv_charges = [], []
        with open(qv_file, 'r', errors='replace') as fin:
            for line in fin:
                clean = line.split('!')[0].strip() 
                if not clean: continue
                parts = clean.split()
                if len(parts) >= 4:
                    try:
                        q = float(parts[3])
                        if abs(q) >= qvcutoff:
                            qv_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                            qv_charges.append(q)
                    except ValueError: continue
                        
        if qv_coords:
            qv_c_arr, qv_q_arr = np.array(qv_coords), np.array(qv_charges)
            all_solute_coords = [[parse_fortran_float(row[4]), parse_fortran_float(row[5]), parse_fortran_float(row[6])] for row in udata]
            
            dists = cdist(qv_c_arr, np.array(all_solute_coords))
            valid_mask = np.min(dists, axis=1) >= qvcore
            solv_c, solv_q = qv_c_arr[valid_mask], qv_q_arr[valid_mask]
            
            ext_c.extend(solv_c.tolist())
            ext_q.extend(solv_q.tolist())

            # Calculate MM - Solvent interaction
            if mm_indices and len(solv_c) > 0:
                mm_c_arr = np.array([[parse_fortran_float(udata[i-1][4]), parse_fortran_float(udata[i-1][5]), parse_fortran_float(udata[i-1][6])] for i in mm_indices])
                mm_q_arr = np.array([parse_fortran_float(udata[i-1][3]) for i in mm_indices])
                inv_dists = np.where(cdist(mm_c_arr, solv_c) > 1e-6, 1.0/cdist(mm_c_arr, solv_c), 0.0)
                e_mv_jmol = np.dot(mm_q_arr, np.dot(inv_dists, solv_q)) * COULOMB_TO_JMOL

    # 2. Add MM Atoms to the external charge array
    if mm_indices:
        for i in mm_indices:
            atom = udata[i-1]
            ext_c.append([parse_fortran_float(atom[4]), parse_fortran_float(atom[5]), parse_fortran_float(atom[6])])
            ext_q.append(parse_fortran_float(atom[3]))
            
    # 3. Write back the combined filtered charges to .qv for the record (optional but good for debugging)
    if ext_c:
        with open(qv_file, 'w') as fout:
            for c, q in zip(ext_c, ext_q):
                fout.write(f" {c[0]:11.6f} {c[1]:11.6f} {c[2]:11.6f}  {q:.6e}\n")

    return np.array(ext_c) if ext_c else None, np.array(ext_q) if ext_q else None, e_mv_jmol

# Generate cube file
def generate_and_write_cube(mol, dm, outfile, ngrid3d, rdelta3d, udata, mm_indices):
    """
    Computes MEP directly in memory using PySCF, adds MM potential, 
    and writes the final 3D-RISM ready Cube file.
    """
    origin_ang = -rdelta3d * (ngrid3d / 2.0)
    origin_bohr = origin_ang * ANG_TO_BOHR
    step_bohr = rdelta3d * ANG_TO_BOHR
    extent_bohr = step_bohr * (ngrid3d - 1)
    
    # 1. Initialize PySCF Cube Object with EXACT grid match
    cube = cubegen.Cube(mol, nx=ngrid3d, ny=ngrid3d, nz=ngrid3d)
    cube.boxorig = np.array([origin_bohr, origin_bohr, origin_bohr])
    cube.box = np.diag([extent_bohr, extent_bohr, extent_bohr])
    
    # 2. Compute QM Electrostatic Potential (Nuclei + Electron Density)
    pot = cube.mep(dm)
    
    # 3. Add MM Electrostatic Potential
    if mm_indices:
        X = cube.boxorig[0] + np.arange(ngrid3d) * step_bohr
        Y = cube.boxorig[1] + np.arange(ngrid3d) * step_bohr
        Z = cube.boxorig[2] + np.arange(ngrid3d) * step_bohr
        XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing='ij')
        XX_flat, YY_flat, ZZ_flat = XX.ravel(), YY.ravel(), ZZ.ravel()
        
        v_mm = np.zeros_like(XX_flat)
        for i in mm_indices:
            atom = udata[i-1]
            c_bohr = np.array([parse_fortran_float(atom[4]), parse_fortran_float(atom[5]), parse_fortran_float(atom[6])]) * ANG_TO_BOHR
            q = parse_fortran_float(atom[3])
            
            d_sq = (XX_flat - c_bohr[0])**2 + (YY_flat - c_bohr[1])**2 + (ZZ_flat - c_bohr[2])**2
            v_mm += q / np.sqrt(np.maximum(d_sq, 1e-12))
            
        pot += v_mm.reshape(ngrid3d, ngrid3d, ngrid3d)
        
    # 4. Write to disk
    cube.write(pot, outfile, comment="QM+MM Electrostatic Potential generated by RISMiCal-PySCF")

# Update RISMiCal input
def update_rismical_input(inp_file, lines, udata, udata_line_indices, qm_indices, charges):
    for c_idx, idx in enumerate(qm_indices):
        row = udata[idx-1]
        p1, p2 = parse_fortran_float(row[1]), parse_fortran_float(row[2])
        x, y, z = parse_fortran_float(row[4]), parse_fortran_float(row[5]), parse_fortran_float(row[6])
        lines[udata_line_indices[idx-1]] = f"{row[0]:<7s}{p1:8.4f}{p2:10.4f}{charges[c_idx]:12.6f}     {x:8.4f}   {y:8.4f}   {z:8.4f}\n"
    with open(inp_file, 'w', errors='replace') as f: f.writelines(lines)

# Check RISMiCal termination
def check_rismical_termination(rsmout_file):
    if not os.path.exists(rsmout_file): return False
    with open(rsmout_file, 'r', errors='replace') as f:
        for line in f.readlines()[-30:]:
            if "RISMiCal computation is completed normally" in line: return True
    return False

# Read xmu file to get solvent free energy and solute-solvent electrostatic interaction energy
def read_xmu(xmu_file):
    sfe_sc, se_es = 0.0, 0.0
    with open(xmu_file, 'r', errors='replace') as f:
        for line in f:
            if 'SFE_SC=' in line: sfe_sc = float(line.split('=')[1].split('!')[0].strip())
            elif 'SE_ES=' in line: se_es = float(line.split('=')[1].split('!')[0].strip())
    return sfe_sc, se_es

#
# Main
#
def main():
    # [NEW] Check for FC flag and safely remove it from argv so inp_file parsing works
    fc_mode = False
    if "-FC" in sys.argv:
        fc_mode = True
        sys.argv.remove("-FC")

    if len(sys.argv) < 2: 
        print("Usage: python RISMiCal-PySCF.py <input_file> [-FC]"); sys.exit(1)
        
    inp_file = sys.argv[1]
    base_name = os.path.splitext(inp_file)[0]
    ascii_cube = f"{base_name}.cub"
    xmu_file, qv_file, org_inp_file = f"{base_name}.xmu", f"{base_name}.qv", f"{base_name}.org.inp"
    
    if not os.path.exists(org_inp_file): shutil.copy(inp_file, org_inp_file)
    
    params, udata, udata_indices, all_lines = read_input_file(inp_file)
    
    # Check PySCF specific params
    xc = params.get('xc', 'b3lyp')
    basis = params.get('basis', '6-31g*')
    
    scfconv  = parse_fortran_float(params.get('scfconv', '1e-4'))
    qvcutoff = parse_fortran_float(params.get('qvcutoff', '1e-6'))
    qvcore   = parse_fortran_float(params.get('qvcore', '0.5'))
    ngrid3d  = int(parse_fortran_float(params.get('ngrid3d', '128')))
    rdelta3d = parse_fortran_float(params.get('rdelta3d', '0.5'))
    
    total_atoms = len(udata)
    qm_indices = parse_qmpart(params.get('qmpart', ''), total_atoms)
    mm_indices = [i for i in range(1, total_atoms + 1) if i not in qm_indices]
    
    print(f"--- 3D-RISM-PySCF Started ({base_name}) ---")
    print(f" Engine     : PySCF ({xc}/{basis})")
    print(f" Total Atoms: {total_atoms} (QM: {len(qm_indices)}, MM: {len(mm_indices)})")
    
    mol = build_pyscf_mol(params, udata, qm_indices)

    # =========================================================
    # [NEW] Franck-Condon (FC) Mode Execution
    # =========================================================
    if fc_mode:
        print("\n--- Franck-Condon (FC) State Calculation ---")
        if not os.path.exists(qv_file):
            print(f"  [ERROR] FC mode requires an existing '{qv_file}' in the current directory.")
            sys.exit(1)
            
        print(f"  Using existing frozen solvent/MM charges from: {qv_file}")
        
        # Read external charges from the existing .qv file
        ext_coords, ext_charges, e_mv = process_qv_and_get_ext_charges(qv_file, udata, qvcutoff, qvcore, mm_indices)
        
        # Run PySCF single point calculation with the frozen environment
        e_qm_hartree, dm, new_chg = run_pyscf_scf(mol, params, ext_coords, ext_charges)
        e_qm = e_qm_hartree * HARTREE_TO_JMOL
        
        print("\n  >>> FC Calculation Completed Successfully! <<<")
        print(f"  QM Energy (with frozen env): {e_qm:.4f} J/mol")
        if mm_indices:
            print(f"  MM-Solv Int. (E_MV)        : {e_mv:.4f} J/mol")
        sys.exit(0) # Terminate the script without running RISMiCal
    # =========================================================

    # ---------------------------------------------------------
    # Pre-Step 1: Pure QM Vacuum
    # ---------------------------------------------------------
    print("\n--- Pre-Step 1: QM Vacuum (E_gas) ---")
    e_gas_hartree, _, _ = run_pyscf_scf(mol, params)
    e_gas = e_gas_hartree * HARTREE_TO_JMOL
    
    # ---------------------------------------------------------
    # Pre-Step 2: QM + MM Vacuum
    # ---------------------------------------------------------
    print("--- Pre-Step 2: QM+MM Vacuum (E_QMMM_gas) ---")
    if mm_indices:
        mm_c = np.array([[parse_fortran_float(udata[i-1][4]), parse_fortran_float(udata[i-1][5]), parse_fortran_float(udata[i-1][6])] for i in mm_indices])
        mm_q = np.array([parse_fortran_float(udata[i-1][3]) for i in mm_indices])
        e_qmmm_gas_hartree, dm_qmmm, init_qm_charges = run_pyscf_scf(mol, params, ext_coords=mm_c, ext_charges=mm_q)
        e_qmmm_gas = e_qmmm_gas_hartree * HARTREE_TO_JMOL
    else:
        e_qmmm_gas = e_gas
        dm_qmmm, init_qm_charges = run_pyscf_scf(mol, params)[1:3] # Re-run to get density matrix
        print("  No MM atoms specified. Skipping...")
        
    e_qmmm_int = e_qmmm_gas - e_gas

    # ---------------------------------------------------------
    # Initializing Solvent Distribution
    # ---------------------------------------------------------
    print("\n--- Initializing Solvent Distribution ---")
    generate_and_write_cube(mol, dm_qmmm, ascii_cube, ngrid3d, rdelta3d, udata, mm_indices)
    update_rismical_input(inp_file, all_lines, udata, udata_indices, qm_indices, init_qm_charges)
    
    current_rsmout = f"{base_name}.rsmout.0"
    with open(current_rsmout, "w", errors='replace') as f_rsm: 
        subprocess.run([RISMICAL_CMD, "3d", inp_file], stdout=f_rsm, stderr=subprocess.STDOUT)
    if not check_rismical_termination(current_rsmout):
        print(f"  [ERROR] RISMiCal did not terminate normally. Check {current_rsmout}"); sys.exit(1)

    # ---------------------------------------------------------
    # Main SCF Loop
    # ---------------------------------------------------------
    prev_e = 0.0
    iter_num = 1
    print("\n--- Starting QM/MM/3D-RISM SCF Loop ---")
    while True:
        print(f"\n[Iteration {iter_num}]")
        
        # 1. Retrieve RISM energies
        sfe, se = read_xmu(xmu_file)
        
        # 2. Get External Charges (Solvent + MM)
        ext_coords, ext_charges, e_mv = process_qv_and_get_ext_charges(qv_file, udata, qvcutoff, qvcore, mm_indices)
        
        # 3. Run PySCF with External Charges
        e_qm_hartree, dm, new_chg = run_pyscf_scf(mol, params, ext_coords, ext_charges)
        e_qm = e_qm_hartree * HARTREE_TO_JMOL
        
        # 4. Energy Evaluation & Convergence Check
        e_tot = e_qm + e_mv - se + sfe
        print(f"  E_QM    = {e_qm:.4f} J/mol")
        if mm_indices: print(f"  E_MV    = {e_mv:.4f} J/mol")
        print(f"  SE_ES   = {se:.4f} J/mol")
        print(f"  SFE_SC  = {sfe:.4f} J/mol")
        print(f"  E_TOTAL = {e_tot:.4f} J/mol")
        
        d_e_h = abs(e_tot - prev_e) / HARTREE_TO_JMOL
        print(f"  Delta E = {d_e_h:.6e} Hartree (Threshold: {scfconv})")
        
        if iter_num > 1 and d_e_h <= scfconv: 
            break
            
        prev_e = e_tot
        
        # 5. Generate New Potentials for RISMiCal
        generate_and_write_cube(mol, dm, ascii_cube, ngrid3d, rdelta3d, udata, mm_indices)
        update_rismical_input(inp_file, all_lines, udata, udata_indices, qm_indices, new_chg)
        
        current_rsmout = f"{base_name}.rsmout.{iter_num}"
        with open(current_rsmout, "w", errors='replace') as f_rsm: 
            subprocess.run([RISMICAL_CMD, "3d", inp_file], stdout=f_rsm, stderr=subprocess.STDOUT)
            
        if not check_rismical_termination(current_rsmout):
            print(f"  [ERROR] RISMiCal did not terminate normally in Iteration {iter_num}. Check {current_rsmout}"); sys.exit(1)
            
        iter_num += 1
        
    print("\n" + "="*55)
    print(" SUMMARY OF QM/MM/3D-RISM-SCF (PySCF) CALCULATION")
    print("="*55)
    print(" >>> SCF Converged! <<<")
    print(f" Gas-phase Energy (E_gas)    : {e_gas:15.5f} J/mol")
    if mm_indices:
        print(f" QMMM Gas Energy (E_QMMM_gas): {e_qmmm_gas:15.5f} J/mol")
        print(f" QM-MM Interaction (gas)     : {e_qmmm_int:15.5f} J/mol")
    print(f" Total Free Energy (E_TOTAL) : {e_tot:15.5f} J/mol")
    print("-" * 55)
    print(" Generated Files:")
    print(f"   Input Backup  : {org_inp_file}")
    print(f"   RISMiCal LOGs : {base_name}.rsmout.*")
    print(f"   ASCII Cube    : {ascii_cube}")
    print(f"   RISMiCal OUT  : {xmu_file}")
    print(f"   Ext. Charges  : {qv_file}")
    print("="*55)

if __name__ == "__main__": 
    main()