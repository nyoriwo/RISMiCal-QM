#!/usr/bin/env python3
"""
RISMiCal-QM.py
A wrapper script for performing 3D-RISM-SCF calculations using RISMiCal and Gaussian 16.
It automates the SCF iterative process, external charge generation, filtering, 
convergence checking, and perfectly synchronizes the 3D grid between Gaussian and RISMiCal.
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
# Modify these absolute paths if the executables are not in your system's $PATH.
# ==============================================================================
G16_CMD      = "g16"
FORMCHK_CMD  = "formchk"
CUBEGEN_CMD  = "cubegen"
RISMICAL_CMD = "rismical.x"
# ==============================================================================

# Physical constants for energy conversions
HARTREE_TO_JMOL = 2625499.6394799
COULOMB_TO_JMOL = 1389354.56      # Conversion factor: 1 e^2 / Angstrom -> J/mol

def parse_fortran_float(val_str):
    """
    Converts Fortran-style scientific notation strings (e.g., '1.d-6', '3.150d0') 
    into standard Python float objects safely.
    """
    try:
        return float(str(val_str).lower().replace('d', 'e'))
    except ValueError:
        return 0.0

def parse_qmpart(qmpart_str, total_atoms):
    """
    Parses the 'qmpart' string to determine which atoms are treated quantum mechanically.
    """
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
    """
    Helper function to robustly extract parameters from a Fortran namelist string block.
    Masks strings inside quotes to prevent splitting errors on '=' or ','.
    """
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
        
        for j, p in enumerate(placeholders): 
            val_str = val_str.replace(f"__QUOTE_{j}__", p)
            
        if (val_str.startswith('"') and val_str.endswith('"')) or (val_str.startswith("'") and val_str.endswith("'")):
            val_str = val_str[1:-1]
            
        params[key] = val_str

def read_input_file(inp_file):
    """
    Reads the RISMiCal input file to extract:
    1. Namelist parameters from $RISMICALQM and $GRID3D.
    2. Atom definitions from $UDATA.
    """
    params = {}
    udata = []
    udata_line_indices = []
    
    with open(inp_file, 'r', errors='replace') as f:
        lines = f.readlines()
        
    in_rismicalqm = False
    in_grid3d = False
    in_udata = False
    rismicalqm_block_str = ""
    grid3d_block_str = ""
    
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
            in_udata = True
            continue
            
        if in_rismicalqm and (upper_line == '$END' or upper_line == '/'):
            in_rismicalqm = False
            continue
        elif in_grid3d and (upper_line == '$END' or upper_line == '/'):
            in_grid3d = False
            continue
        elif in_udata and (upper_line == '$END' or upper_line == '/'):
            in_udata = False
            continue
            
        if in_rismicalqm:
            rismicalqm_block_str += clean_line + " "
        elif in_grid3d:
            grid3d_block_str += clean_line + " "
        elif in_udata:
            parts = clean_line.split()
            if len(parts) >= 7:
                udata.append(parts)
                udata_line_indices.append(i)
                
    extract_namelist(rismicalqm_block_str, params)
    extract_namelist(grid3d_block_str, params)
        
    return params, udata, udata_line_indices, lines

def write_gaussian_input(gjf_file, params, udata, qm_indices, iter_num, qv_file):
    """
    Generates the input file (.gjf) for Gaussian 16.
    """
    qmopt_raw = params.get('qmopt', '')
    raw_lines = qmopt_raw.split('\\')
    link0_lines, route_line, title_line, chg_mult = [], "", "RISMiCal-QM Job", "0 1"
    
    for line in raw_lines:
        line = line.strip()
        if line.startswith('%'): link0_lines.append(line)
        elif line.startswith('#'): route_line = line
        elif line and line[0].isdigit() and ' ' in line: chg_mult = line
        elif line: title_line = line
            
    if not re.search(r'(?i)\bpop=', route_line): 
        route_line += " Pop=MK"
            
    if iter_num == 0:
        route_line = re.sub(r'(?i)\bcharge\b', '', route_line)
        route_line = re.sub(r'(?i)\bnosymm\b', '', route_line)
    else:
        if not re.search(r'(?i)\bcharge\b', route_line): 
            route_line += " Charge NoSymm"

    with open(gjf_file, 'w') as f:
        for l in link0_lines: f.write(f"{l}\n")
        f.write(f"{route_line}\n\n{title_line}\n\n{chg_mult}\n")
        
        for idx in qm_indices:
            atom = udata[idx - 1]
            x = parse_fortran_float(atom[4])
            y = parse_fortran_float(atom[5])
            z = parse_fortran_float(atom[6])
            f.write(f"{atom[0]:<2s}  {x:11.6f}  {y:11.6f}  {z:11.6f}\n")
        f.write("\n")
        
        if iter_num > 0 and os.path.exists(qv_file): 
            f.write(f"@{qv_file}\n\n")

def check_gaussian_termination(gout_file):
    """ Checks if Gaussian terminated normally. """
    if not os.path.exists(gout_file): return False
    with open(gout_file, 'r', errors='replace') as f:
        lines = f.readlines()[-20:]
        for line in lines:
            if "Normal termination" in line:
                return True
    return False

def check_rismical_termination(rsmout_file):
    """ Checks if RISMiCal terminated normally. """
    if not os.path.exists(rsmout_file): return False
    with open(rsmout_file, 'r', errors='replace') as f:
        lines = f.readlines()[-30:]
        for line in lines:
            if "RISMiCal computation is completed normally" in line:
                return True
    return False

def read_charges_from_gout(gout_file, natoms_qm):
    """ Extracts ESP/MK charges from Gaussian log safely. """
    with open(gout_file, 'r', errors='replace') as f: 
        lines = f.readlines()
        
    start_idx = -1
    for i in range(len(lines)-1, -1, -1):
        if "Charges from ESP fit" in lines[i] or "Fitting point charges" in lines[i]:
            start_idx = i; break
            
    if start_idx == -1:
        for i in range(len(lines)-1, -1, -1):
            if "Mulliken charges:" in lines[i]: 
                start_idx = i; break
                
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
    """
    Runs the 'cubegen' utility to create an ASCII cube file of the SCF potential
    exactly matching the grid coordinates required by RISMiCal.
    """
    # 1. Calculate Grid Origin in Angstroms: -rdelta3d * (ngrid3d / 2)
    origin_ang = -rdelta3d * (ngrid3d / 2.0)
    
    # 2. Construct Custom Grid Specification String
    # Format for cubegen when npts=-1:
    # IFlag, Origin_X, Origin_Y, Origin_Z (IFlag is output unit, 0 is standard)
    # Ngrid_X, Step_X, 0.0, 0.0           (If Ngrid > 0, units are interpreted as Angstroms by Gaussian)
    # Ngrid_Y, 0.0, Step_Y, 0.0
    # Ngrid_Z, 0.0, 0.0, Step_Z
    
    grid_spec = f"0 {origin_ang:.6f} {origin_ang:.6f} {origin_ang:.6f}\n"
    grid_spec += f"{ngrid3d} {rdelta3d:.6f} 0.0 0.0\n"
    grid_spec += f"{ngrid3d} 0.0 {rdelta3d:.6f} 0.0\n"
    grid_spec += f"{ngrid3d} 0.0 0.0 {rdelta3d:.6f}\n"
    
    # 3. Execute cubegen and capture stderr for debugging
    result = subprocess.run(
        [CUBEGEN_CMD, "0", "potential=scf", fchk_file, ascii_cube_file, "-1"],
        input=grid_spec,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    
    if not os.path.exists(ascii_cube_file):
        print(f"\n  [ERROR] {ascii_cube_file} was not generated by cubegen.")
        if result.stderr:
            print(f"  [cubegen STDERR] {result.stderr.strip()}")
        sys.exit(1)

def update_rismical_input(inp_file, lines, udata, udata_line_indices, qm_indices, charges):
    """ Updates the $UDATA section of the RISMiCal input file in-place. """
    for c_idx, idx in enumerate(qm_indices):
        row = udata[idx-1]
        p1 = parse_fortran_float(row[1])
        p2 = parse_fortran_float(row[2])
        x  = parse_fortran_float(row[4])
        y  = parse_fortran_float(row[5])
        z  = parse_fortran_float(row[6])
        
        lines[udata_line_indices[idx-1]] = f"{row[0]:<7s}{p1:8.4f}{p2:10.4f}{charges[c_idx]:12.6f}     {x:8.4f}   {y:8.4f}   {z:8.4f}\n"
        
    with open(inp_file, 'w', errors='replace') as f: 
        f.writelines(lines)

def process_qv_and_calc_emv(qv_file, udata, qvcutoff, qvcore, mm_indices):
    """ Processes the .qv file, applies filters, and handles MM atoms securely. """
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
                except ValueError:
                    continue
                    
    if not qv_coords:
        return 0.0
        
    qv_c_arr = np.array(qv_coords)
    qv_q_arr = np.array(qv_charges)
    
    all_solute_coords = []
    for row in udata:
        x = parse_fortran_float(row[4])
        y = parse_fortran_float(row[5])
        z = parse_fortran_float(row[6])
        all_solute_coords.append([x, y, z])
    all_solute_arr = np.array(all_solute_coords)
    
    dists = cdist(qv_c_arr, all_solute_arr)
    min_dists = np.min(dists, axis=1)
    
    valid_mask = min_dists >= qvcore
    qv_c_valid = qv_c_arr[valid_mask]
    qv_q_valid = qv_q_arr[valid_mask]
    
    e_mv_jmol = 0.0
    if mm_indices and len(qv_c_valid) > 0:
        mm_c = np.array([[parse_fortran_float(udata[i-1][4]), 
                          parse_fortran_float(udata[i-1][5]), 
                          parse_fortran_float(udata[i-1][6])] for i in mm_indices])
        mm_q = np.array([parse_fortran_float(udata[i-1][3]) for i in mm_indices])
        
        inv_dists = np.where(cdist(mm_c, qv_c_valid) > 1e-6, 1.0/cdist(mm_c, qv_c_valid), 0.0)
        e_mv_jmol = np.dot(mm_q, np.dot(inv_dists, qv_q_valid)) * COULOMB_TO_JMOL
        
    with open(qv_file, 'w') as fout:
        for c, q in zip(qv_c_valid, qv_q_valid):
            fout.write(f" {c[0]:11.6f} {c[1]:11.6f} {c[2]:11.6f}  {q:.6e}\n")
            
        if mm_indices:
            for i in mm_indices:
                atom = udata[i-1]
                x = parse_fortran_float(atom[4])
                y = parse_fortran_float(atom[5])
                z = parse_fortran_float(atom[6])
                q = parse_fortran_float(atom[3])
                fout.write(f" {x:11.6f} {y:11.6f} {z:11.6f}  {q:.6e}\n")
                
    return e_mv_jmol

def read_xmu(xmu_file):
    """ Extracts SFE_SC and SE_ES in J/mol safely. """
    sfe_sc, se_es = 0.0, 0.0
    with open(xmu_file, 'r', errors='replace') as f:
        for line in f:
            if 'SFE_SC=' in line:
                sfe_sc = float(line.split('=')[1].split('!')[0].strip())
            elif 'SE_ES=' in line:
                se_es = float(line.split('=')[1].split('!')[0].strip())
    return sfe_sc, se_es

def read_eqm(gout_file):
    """ Extracts final QM Energy safely. """
    eqm = 0.0
    with open(gout_file, 'r', errors='replace') as f:
        for line in reversed(list(f)):
            if 'SCF Done:' in line:
                eqm = float(line.split()[4])
                break
    return eqm * HARTREE_TO_JMOL

def main():
    if len(sys.argv) < 2: 
        print("Usage: python RISMiCal-QM.py <input_file>")
        sys.exit(1)
        
    inp_file = sys.argv[1]
    base_name = os.path.splitext(inp_file)[0]
    
    gjf_file      = f"{base_name}.gjf"
    chk_file      = f"{base_name}.chk"
    fchk_file     = f"{base_name}.fchk"
    gout_file     = f"{base_name}.gout"
    ascii_cube    = f"{base_name}.cub"
    xmu_file      = f"{base_name}.xmu"
    qv_file       = f"{base_name}.qv"
    org_inp_file  = f"{base_name}.org.inp"
    
    if not os.path.exists(org_inp_file): 
        shutil.copy(inp_file, org_inp_file)
    
    params, udata, udata_indices, all_lines = read_input_file(inp_file)
    qm_engine = params.get('qm', 'g16').lower()
    if qm_engine != 'g16':
        print("Error: Currently only qm=\"g16\" is supported.")
        sys.exit(1)
        
    # Extract settings
    scfconv  = parse_fortran_float(params.get('scfconv', '1e-4'))
    qvcutoff = parse_fortran_float(params.get('qvcutoff', '1e-6'))
    qvcore   = parse_fortran_float(params.get('qvcore', '0.5'))
    
    # Extract Grid Configuration for Custom cubegen execution
    ngrid3d  = int(parse_fortran_float(params.get('ngrid3d', '128')))
    rdelta3d = parse_fortran_float(params.get('rdelta3d', '0.5'))
    
    total_atoms = len(udata)
    qm_indices = parse_qmpart(params.get('qmpart', ''), total_atoms)
    mm_indices = [i for i in range(1, total_atoms + 1) if i not in qm_indices]
    
    prev_e = 0.0
    iter_num = 0
    print(f"--- 3D-RISM-QM Started ({base_name}) ---")
    print(f" Total Atoms: {total_atoms} (QM: {len(qm_indices)}, MM: {len(mm_indices)})")
    print(f" Grid Config: {ngrid3d}^3 points, spacing {rdelta3d} A")
    print(f" Filters    : qvcutoff {qvcutoff}, qvcore {qvcore} A")
    
    while True:
        print(f"\n[Iteration {iter_num}]")
        
        write_gaussian_input(gjf_file, params, udata, qm_indices, iter_num, qv_file)
        subprocess.run([G16_CMD, gjf_file, gout_file])
        
        if not check_gaussian_termination(gout_file):
            print(f"\n  [ERROR] Gaussian 16 did not terminate normally in iteration {iter_num}.")
            print(f"          Please check the log file: {gout_file}")
            sys.exit(1)
            
        shutil.copy(gout_file, f"{base_name}.gout.{iter_num}")
        
        new_chg = read_charges_from_gout(gout_file, len(qm_indices))
        subprocess.run([FORMCHK_CMD, chk_file, fchk_file], stdout=subprocess.DEVNULL)
        
        # Execute customized cubegen to match RISMiCal grid
        run_cubegen(fchk_file, ascii_cube, ngrid3d, rdelta3d)
        
        update_rismical_input(inp_file, all_lines, udata, udata_indices, qm_indices, new_chg)
        current_rsmout = f"{base_name}.rsmout.{iter_num}"
        with open(current_rsmout, "w", errors='replace') as f_rsm: 
            subprocess.run([RISMICAL_CMD, "3d", inp_file], stdout=f_rsm, stderr=subprocess.STDOUT)
            
        if not check_rismical_termination(current_rsmout):
            print(f"\n  [ERROR] RISMiCal did not terminate normally (not converged) in iteration {iter_num}.")
            print(f"          Please check the log file: {current_rsmout}")
            sys.exit(1)
        
        sfe, se = read_xmu(xmu_file)
        e_qm = read_eqm(gout_file)
        
        e_mv = process_qv_and_calc_emv(qv_file, udata, qvcutoff, qvcore, mm_indices)
        
        e_tot = e_qm + e_mv - se + sfe
        print(f"  E_QM    = {e_qm:.4f} J/mol")
        if mm_indices: print(f"  E_MV    = {e_mv:.4f} J/mol")
        print(f"  SE_ES   = {se:.4f} J/mol")
        print(f"  SFE_SC  = {sfe:.4f} J/mol")
        print(f"  E_TOT   = {e_tot:.4f} J/mol")
        
        d_e_h = abs(e_tot - prev_e) / HARTREE_TO_JMOL
        print(f"  Delta E = {d_e_h:.6e} Hartree (Threshold: {scfconv})")
        
        if iter_num > 0 and d_e_h <= scfconv: 
            break
            
        prev_e = e_tot
        iter_num += 1
        
    print("\n" + "="*50)
    print(" SUMMARY OF 3D-RISM-QM CALCULATION")
    print("="*50)
    print(" >>> SCF Converged! <<<")
    print(f" Total Free Energy (E_TOTAL) : {e_tot:15.5f} J/mol")
    print(f" QM Energy (E_QM)            : {e_qm:15.5f} J/mol")
    if mm_indices:
        print(f" MM-Solv Int. (E_MV)         : {e_mv:15.5f} J/mol")
    print(f" Electrostatic Int. (SE_ES)  : {se:15.5f} J/mol")
    print(f" Solvation Free E. (SFE_SC)  : {sfe:15.5f} J/mol")
    print("-" * 50)
    print(" Effective Charges (ESP/MK) for QM region:")
    for i, q in zip(qm_indices, new_chg):
        atom_name = udata[i-1][0]
        print(f"   Atom {i:2d} ({atom_name:2s}) : {q:9.6f} e")
    print("-" * 50)
    print(" Generated Files:")
    print(f"   Input Backup  : {org_inp_file}")
    print(f"   Gaussian INP  : {gjf_file}")
    print(f"   Gaussian LOGs : {base_name}.gout.*")
    print(f"   RISMiCal LOGs : {base_name}.rsmout.*")
    print(f"   ASCII Cube    : {ascii_cube}")
    print(f"   RISMiCal OUT  : {xmu_file}")
    print(f"   Ext. Charges  : {qv_file}")
    print("="*50)

if __name__ == "__main__": 
    main()
