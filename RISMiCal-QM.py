#!/usr/bin/env python3
"""
RISMiCal-QM.py
A wrapper script for performing QM/MM/3D-RISM-SCF calculations using RISMiCal and Gaussian 16.
It automates the SCF iterative process, external charge generation, filtering, 
grid synchronization, and custom MM potential integrations.
"""

import os
import sys
import shutil
import subprocess
import re
import numpy as np
from scipy.spatial.distance import cdist

# ==============================================================================
# Commands and Paths Configuration
# ==============================================================================
G16_CMD      = "g16"
FORMCHK_CMD  = "formchk"
CUBEGEN_CMD  = "cubegen"
RISMICAL_CMD = "rismical.x"
# ==============================================================================

# Physical constants for energy and length conversions
HARTREE_TO_JMOL = 2625499.6394799
COULOMB_TO_JMOL = 1389354.56      # 1 e^2 / Angstrom -> J/mol
ANG_TO_BOHR     = 1.8897261246    # Angstrom -> Bohr

def parse_fortran_float(val_str):
    try:
        return float(str(val_str).lower().replace('d', 'e'))
    except ValueError:
        return 0.0

def parse_qmpart(qmpart_str, total_atoms):
    if not qmpart_str or not str(qmpart_str).strip():
        return list(range(1, total_atoms + 1))
    indices = []
    parts = str(qmpart_str).split(',')
    last_idx = 0
    for p in parts:
        if not p.strip(): continue
        val = int(p.strip())
        if val > 0:
            indices.append(val)
            last_idx = val
        elif val < 0:
            indices.extend(range(last_idx + 1, abs(val) + 1))
            last_idx = abs(val)
    return sorted(list(set(indices)))

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
            continue
        elif upper_line.startswith('$GRID3D') or upper_line.startswith('&GRID3D'):
            in_grid3d = True
            header_len = len('$GRID3D') if upper_line.startswith('$GRID3D') else len('&GRID3D')
            grid3d_block_str += clean_line[header_len:] + " "
            continue
        elif upper_line.startswith('$UDATA') or upper_line.startswith('&UDATA'):
            in_udata = True; continue
        if in_rismicalqm and (upper_line == '$END' or upper_line == '/'):
            in_rismicalqm = False; continue
        elif in_grid3d and (upper_line == '$END' or upper_line == '/'):
            in_grid3d = False; continue
        elif in_udata and (upper_line == '$END' or upper_line == '/'):
            in_udata = False; continue
        
        if in_rismicalqm: rismicalqm_block_str += clean_line + " "
        elif in_grid3d: grid3d_block_str += clean_line + " "
        elif in_udata:
            parts = clean_line.split()
            if len(parts) >= 7: udata.append(parts); udata_line_indices.append(i)
                
    extract_namelist(rismicalqm_block_str, params)
    extract_namelist(grid3d_block_str, params)
    return params, udata, udata_line_indices, lines

def write_gaussian_input(gjf_file, params, udata, qm_indices, iter_num, qv_file):
    qmopt_raw = params.get('qmopt', '')
    raw_lines = qmopt_raw.split('\\')
    link0_lines, route_line, title_line, chg_mult = [], "", "RISMiCal-QM Job", "0 1"
    for line in raw_lines:
        line = line.strip()
        if line.startswith('%'): link0_lines.append(line)
        elif line.startswith('#'): route_line = line
        elif line and line[0].isdigit() and ' ' in line: chg_mult = line
        elif line: title_line = line
            
    if not re.search(r'(?i)\bpop=', route_line): route_line += " Pop=MK"
            
    # iter_num == -1 specifies Pure QM Vacuum (No MM, No Charge)
    if iter_num == -1:
        route_line = re.sub(r'(?i)\bcharge\b', '', route_line)
        route_line = re.sub(r'(?i)\bnosymm\b', '', route_line)
    else:
        if not re.search(r'(?i)\bcharge\b', route_line): route_line += " Charge NoSymm"

    with open(gjf_file, 'w') as f:
        for l in link0_lines: f.write(f"{l}\n")
        f.write(f"{route_line}\n\n{title_line}\n\n{chg_mult}\n")
        for idx in qm_indices:
            atom = udata[idx - 1]
            x, y, z = parse_fortran_float(atom[4]), parse_fortran_float(atom[5]), parse_fortran_float(atom[6])
            f.write(f"{atom[0]:<2s}  {x:11.6f}  {y:11.6f}  {z:11.6f}\n")
        f.write("\n")
        if iter_num >= 0 and os.path.exists(qv_file): f.write(f"@{qv_file}\n\n")

def write_mm_qv(qv_file, udata, mm_indices):
    """ Creates a .qv file containing ONLY MM atoms (used for QM+MM Gas Phase calculations). """
    with open(qv_file, 'w') as f:
        for i in mm_indices:
            atom = udata[i-1]
            x, y, z = parse_fortran_float(atom[4]), parse_fortran_float(atom[5]), parse_fortran_float(atom[6])
            q = parse_fortran_float(atom[3])
            f.write(f" {x:11.6f} {y:11.6f} {z:11.6f}  {q:.6e}\n")

def check_gaussian_termination(gout_file):
    if not os.path.exists(gout_file): return False
    with open(gout_file, 'r', errors='replace') as f:
        for line in f.readlines()[-20:]:
            if "Normal termination" in line: return True
    return False

def check_rismical_termination(rsmout_file):
    if not os.path.exists(rsmout_file): return False
    with open(rsmout_file, 'r', errors='replace') as f:
        for line in f.readlines()[-30:]:
            if "RISMiCal computation is completed normally" in line: return True
    return False

def read_charges_from_gout(gout_file, natoms_qm):
    with open(gout_file, 'r', errors='replace') as f: lines = f.readlines()
    start_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if "Charges from ESP fit" in lines[i] or "Fitting point charges" in lines[i]:
            start_idx = i; break
    if start_idx == -1:
        for i in range(len(lines)-1, -1, -1):
            if "Mulliken charges:" in lines[i]: start_idx = i; break
    charges = []
    if start_idx != -1:
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) == 3 and parts[0].isdigit() and not parts[1].isdigit():
                try: charges.append(float(parts[2]))
                except: pass
            if len(charges) == natoms_qm: break
    return charges if len(charges) == natoms_qm else [0.0]*natoms_qm

def run_cubegen(fchk_file, ascii_cube_file, ngrid3d, rdelta3d):
    origin_ang = -rdelta3d * (ngrid3d / 2.0)
    grid_spec = f"0 {origin_ang:.6f} {origin_ang:.6f} {origin_ang:.6f}\n"
    grid_spec += f"{ngrid3d} {rdelta3d:.6f} 0.0 0.0\n"
    grid_spec += f"{ngrid3d} 0.0 {rdelta3d:.6f} 0.0\n"
    grid_spec += f"{ngrid3d} 0.0 0.0 {rdelta3d:.6f}\n"
    
    result = subprocess.run([CUBEGEN_CMD, "0", "potential=scf", fchk_file, ascii_cube_file, "-1"],
                            input=grid_spec, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if not os.path.exists(ascii_cube_file):
        print(f"\n  [ERROR] {ascii_cube_file} was not generated by cubegen.")
        if result.stderr: print(f"  [cubegen STDERR] {result.stderr.strip()}")
        sys.exit(1)

def add_mm_potential_to_cube(ascii_cube_file, udata, mm_indices):
    """
    Reads the ASCII cube generated by cubegen, calculates the electrostatic potential
    generated by the MM atoms, and adds it to the grid. 
    Memory-optimized to handle large grids without exceeding RAM.
    """
    if not mm_indices: return

    with open(ascii_cube_file, 'r', errors='replace') as f:
        lines = f.readlines()

    natoms = int(lines[2].split()[0])
    abs_natoms = abs(natoms)
    header_end = 6 + abs_natoms

    x0, y0, z0 = map(float, lines[2].split()[1:4])
    nx, dx = float(lines[3].split()[0]), float(lines[3].split()[1])
    ny, dy = float(lines[4].split()[0]), float(lines[4].split()[2])
    nz, dz = float(lines[5].split()[0]), float(lines[5].split()[3])
    nx, ny, nz = int(nx), int(ny), int(nz)

    # Read original QM potential
    data_flat = []
    for line in lines[header_end:]:
        data_flat.extend(map(float, line.split()))
    pot = np.array(data_flat, dtype=np.float64)

    # Extract MM atoms and convert coords to Bohr (Cube files operate in Bohr)
    mm_c, mm_q = [], []
    for i in mm_indices:
        atom = udata[i-1]
        mm_c.append([parse_fortran_float(atom[4]) * ANG_TO_BOHR, 
                     parse_fortran_float(atom[5]) * ANG_TO_BOHR, 
                     parse_fortran_float(atom[6]) * ANG_TO_BOHR])
        mm_q.append(parse_fortran_float(atom[3]))

    # Generate 1D Grid Vectors
    X = x0 + np.arange(nx) * dx
    Y = y0 + np.arange(ny) * dy
    Z = z0 + np.arange(nz) * dz

    # Create coordinate grid (Memory efficient, shape exactly matches Cube nesting)
    XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing='ij')
    XX_flat, YY_flat, ZZ_flat = XX.ravel(), YY.ravel(), ZZ.ravel()

    # Calculate MM Potential V = sum(q/r). V is in Hartree/e
    v_mm = np.zeros(len(XX_flat), dtype=np.float64)
    for c, q in zip(mm_c, mm_q):
        d_sq = (XX_flat - c[0])**2 + (YY_flat - c[1])**2 + (ZZ_flat - c[2])**2
        d_sq = np.maximum(d_sq, 1e-12) # Prevent div by zero if MM atom lies on grid
        v_mm += q / np.sqrt(d_sq)

    pot += v_mm

    # Helper function to split array into chunks of 6 elements to emulate Fortran 6E13.5
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    with open(ascii_cube_file, 'w') as f:
        f.writelines(lines[:header_end])
        for chunk in chunker(pot, 6):
            # Formats exactly as "  1.23456E-01" (13 chars)
            line = "".join([f"{v:13.5E}" for v in chunk]) + "\n"
            f.write(line)

def update_rismical_input(inp_file, lines, udata, udata_line_indices, qm_indices, charges):
    for c_idx, idx in enumerate(qm_indices):
        row = udata[idx-1]
        p1, p2 = parse_fortran_float(row[1]), parse_fortran_float(row[2])
        x, y, z = parse_fortran_float(row[4]), parse_fortran_float(row[5]), parse_fortran_float(row[6])
        lines[udata_line_indices[idx-1]] = f"{row[0]:<7s}{p1:8.4f}{p2:10.4f}{charges[c_idx]:12.6f}     {x:8.4f}   {y:8.4f}   {z:8.4f}\n"
    with open(inp_file, 'w', errors='replace') as f: f.writelines(lines)

def process_qv_and_calc_emv(qv_file, udata, qvcutoff, qvcore, mm_indices):
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
                    
    if not qv_coords: return 0.0
        
    qv_c_arr, qv_q_arr = np.array(qv_coords), np.array(qv_charges)
    all_solute_coords = [[parse_fortran_float(row[4]), parse_fortran_float(row[5]), parse_fortran_float(row[6])] for row in udata]
    
    dists = cdist(qv_c_arr, np.array(all_solute_coords))
    valid_mask = np.min(dists, axis=1) >= qvcore
    qv_c_valid, qv_q_valid = qv_c_arr[valid_mask], qv_q_arr[valid_mask]
    
    e_mv_jmol = 0.0
    if mm_indices and len(qv_c_valid) > 0:
        mm_c = np.array([[parse_fortran_float(udata[i-1][4]), parse_fortran_float(udata[i-1][5]), parse_fortran_float(udata[i-1][6])] for i in mm_indices])
        mm_q = np.array([parse_fortran_float(udata[i-1][3]) for i in mm_indices])
        inv_dists = np.where(cdist(mm_c, qv_c_valid) > 1e-6, 1.0/cdist(mm_c, qv_c_valid), 0.0)
        e_mv_jmol = np.dot(mm_q, np.dot(inv_dists, qv_q_valid)) * COULOMB_TO_JMOL
        
    with open(qv_file, 'w') as fout:
        for c, q in zip(qv_c_valid, qv_q_valid):
            fout.write(f" {c[0]:11.6f} {c[1]:11.6f} {c[2]:11.6f}  {q:.6e}\n")
        if mm_indices:
            for i in mm_indices:
                atom = udata[i-1]
                x, y, z = parse_fortran_float(atom[4]), parse_fortran_float(atom[5]), parse_fortran_float(atom[6])
                q = parse_fortran_float(atom[3])
                fout.write(f" {x:11.6f} {y:11.6f} {z:11.6f}  {q:.6e}\n")
    return e_mv_jmol

def read_xmu(xmu_file):
    sfe_sc, se_es = 0.0, 0.0
    with open(xmu_file, 'r', errors='replace') as f:
        for line in f:
            if 'SFE_SC=' in line: sfe_sc = float(line.split('=')[1].split('!')[0].strip())
            elif 'SE_ES=' in line: se_es = float(line.split('=')[1].split('!')[0].strip())
    return sfe_sc, se_es

def read_eqm(gout_file):
    eqm = 0.0
    with open(gout_file, 'r', errors='replace') as f:
        for line in reversed(list(f)):
            if 'SCF Done:' in line:
                eqm = float(line.split()[4])
                break
    return eqm * HARTREE_TO_JMOL

def main():
    if len(sys.argv) < 2: 
        print("Usage: python RISMiCal-QM.py <input_file>"); sys.exit(1)
        
    inp_file = sys.argv[1]
    base_name = os.path.splitext(inp_file)[0]
    gjf_file, chk_file, fchk_file = f"{base_name}.gjf", f"{base_name}.chk", f"{base_name}.fchk"
    gout_file, ascii_cube = f"{base_name}.gout", f"{base_name}.cub"
    xmu_file, qv_file, org_inp_file = f"{base_name}.xmu", f"{base_name}.qv", f"{base_name}.org.inp"
    
    if not os.path.exists(org_inp_file): shutil.copy(inp_file, org_inp_file)
    
    params, udata, udata_indices, all_lines = read_input_file(inp_file)
    if params.get('qm', 'g16').lower() != 'g16':
        print("Error: Currently only qm=\"g16\" is supported."); sys.exit(1)
        
    scfconv  = parse_fortran_float(params.get('scfconv', '1e-4'))
    qvcutoff = parse_fortran_float(params.get('qvcutoff', '1e-6'))
    qvcore   = parse_fortran_float(params.get('qvcore', '0.5'))
    ngrid3d  = int(parse_fortran_float(params.get('ngrid3d', '128')))
    rdelta3d = parse_fortran_float(params.get('rdelta3d', '0.5'))
    
    total_atoms = len(udata)
    qm_indices = parse_qmpart(params.get('qmpart', ''), total_atoms)
    mm_indices = [i for i in range(1, total_atoms + 1) if i not in qm_indices]
    
    print(f"--- 3D-RISM-QM Started ({base_name}) ---")
    print(f" Total Atoms: {total_atoms} (QM: {len(qm_indices)}, MM: {len(mm_indices)})")
    
    # ---------------------------------------------------------
    # Pre-Step 1: Pure QM Vacuum
    # ---------------------------------------------------------
    print("\n--- Pre-Step 1: QM Vacuum (E_gas) ---")
    write_gaussian_input(gjf_file, params, udata, qm_indices, iter_num=-1, qv_file="")
    subprocess.run([G16_CMD, gjf_file, gout_file])
    if not check_gaussian_termination(gout_file):
        print("  [ERROR] Gaussian 16 terminated abnormally in Pre-Step 1."); sys.exit(1)
    e_gas = read_eqm(gout_file)
    shutil.copy(gout_file, f"{base_name}.gout.gas")

    # ---------------------------------------------------------
    # Pre-Step 2: QM + MM Vacuum
    # ---------------------------------------------------------
    print("--- Pre-Step 2: QM+MM Vacuum (E_QMMM_gas) ---")
    if mm_indices:
        write_mm_qv(qv_file, udata, mm_indices)
        write_gaussian_input(gjf_file, params, udata, qm_indices, iter_num=0, qv_file=qv_file)
        subprocess.run([G16_CMD, gjf_file, gout_file])
        if not check_gaussian_termination(gout_file):
            print("  [ERROR] Gaussian 16 terminated abnormally in Pre-Step 2."); sys.exit(1)
        e_qmmm_gas = read_eqm(gout_file)
        shutil.copy(gout_file, f"{base_name}.gout.qmmm_gas")
    else:
        e_qmmm_gas = e_gas
        print("  No MM atoms specified. Skipping...")
        
    e_qmmm_int = e_qmmm_gas - e_gas

    # ---------------------------------------------------------
    # Initializing Solvent Distribution (Iteration 0 for RISM)
    # ---------------------------------------------------------
    print("\n--- Initializing Solvent Distribution ---")
    new_chg = read_charges_from_gout(gout_file, len(qm_indices))
    subprocess.run([FORMCHK_CMD, chk_file, fchk_file], stdout=subprocess.DEVNULL)
    run_cubegen(fchk_file, ascii_cube, ngrid3d, rdelta3d)
    add_mm_potential_to_cube(ascii_cube, udata, mm_indices)
    
    update_rismical_input(inp_file, all_lines, udata, udata_indices, qm_indices, new_chg)
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
        
        # 1. Process Solvent + MM external charges
        sfe, se = read_xmu(xmu_file)
        e_mv = process_qv_and_calc_emv(qv_file, udata, qvcutoff, qvcore, mm_indices)
        
        # 2. Run Gaussian
        write_gaussian_input(gjf_file, params, udata, qm_indices, iter_num, qv_file)
        subprocess.run([G16_CMD, gjf_file, gout_file])
        if not check_gaussian_termination(gout_file):
            print(f"  [ERROR] Gaussian 16 terminated abnormally in Iteration {iter_num}. Check {gout_file}"); sys.exit(1)
            
        shutil.copy(gout_file, f"{base_name}.gout.{iter_num}")
        e_qm = read_eqm(gout_file)
        new_chg = read_charges_from_gout(gout_file, len(qm_indices))
        
        # 3. Energy Evaluation & Convergence Check
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
        
        # 4. Generate New Potentials for RISMiCal
        subprocess.run([FORMCHK_CMD, chk_file, fchk_file], stdout=subprocess.DEVNULL)
        run_cubegen(fchk_file, ascii_cube, ngrid3d, rdelta3d)
        add_mm_potential_to_cube(ascii_cube, udata, mm_indices)
        
        update_rismical_input(inp_file, all_lines, udata, udata_indices, qm_indices, new_chg)
        current_rsmout = f"{base_name}.rsmout.{iter_num}"
        with open(current_rsmout, "w", errors='replace') as f_rsm: 
            subprocess.run([RISMICAL_CMD, "3d", inp_file], stdout=f_rsm, stderr=subprocess.STDOUT)
            
        if not check_rismical_termination(current_rsmout):
            print(f"  [ERROR] RISMiCal did not terminate normally in Iteration {iter_num}. Check {current_rsmout}"); sys.exit(1)
            
        iter_num += 1
        
    print("\n" + "="*55)
    print(" SUMMARY OF QM/MM/3D-RISM-SCF CALCULATION")
    print("="*55)
    print(" >>> SCF Converged! <<<")
    print(f" Gas-phase Energy (E_gas)    : {e_gas:15.5f} J/mol")
    if mm_indices:
        print(f" QMMM Gas Energy (E_QMMM_gas): {e_qmmm_gas:15.5f} J/mol")
        print(f" QM-MM Interaction (gas)     : {e_qmmm_int:15.5f} J/mol")
    print(f" Total Free Energy (E_TOTAL) : {e_tot:15.5f} J/mol")
    print("-" * 55)
    print(f" QM Energy (E_QM)            : {e_qm:15.5f} J/mol")
    if mm_indices:
        print(f" MM-Solv Int. (E_MV)         : {e_mv:15.5f} J/mol")
    print(f" Electrostatic Int. (SE_ES)  : {se:15.5f} J/mol")
    print(f" Solvation Free E. (SFE_SC)  : {sfe:15.5f} J/mol")
    print("-" * 55)
    print(" Effective Charges (ESP/MK) for QM region:")
    for i, q in zip(qm_indices, new_chg):
        atom_name = udata[i-1][0]
        print(f"   Atom {i:2d} ({atom_name:2s}) : {q:9.6f} e")
    print("-" * 55)
    print(" Generated Files:")
    print(f"   Input Backup  : {org_inp_file}")
    print(f"   Gaussian INP  : {gjf_file}")
    print(f"   Gaussian LOGs : {base_name}.gout.*")
    print(f"   RISMiCal LOGs : {base_name}.rsmout.*")
    print(f"   ASCII Cube    : {ascii_cube}")
    print(f"   RISMiCal OUT  : {xmu_file}")
    print(f"   Ext. Charges  : {qv_file}")
    print("="*55)

if __name__ == "__main__": 
    main()
