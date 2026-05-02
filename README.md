# RISMiCal-QM

[![Language](https://img.shields.io/badge/Language-Python_3-blue.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

[English](#english) | [日本語](#japanese)

<a id="english"></a>
## Overview
**RISMiCal-QM** is a Python wrapper script designed to perform robust QM/3D-RISM (KSDFT/3D-RISM, 3D-RISM-SCF, QM/MM/3D-RISM) calculations by coupling **Gaussian 16** (Quantum Mechanics engine) and **RISMiCal** (3D-RISM solver). It fully automates the self-consistent iterations between the solute's electronic structure, the MM atomic charges, and the solvent's 3D spatial distribution.

## Key Features
* **Baseline Energy Profiling (Pre-steps):** Automatically evaluates the pure QM vacuum energy (E_gas) and the QM+MM vacuum energy (E_QMMM_gas) before introducing the solvent, allowing for precise calculation of QM-MM interactions.
* **Rigorous QM/MM Potential Integration:** Since Gaussian's cubegen only outputs the QM electrostatic potential, the script calculates the exact Coulomb potential of the MM atoms in Python and directly merges it into the 3D ASCII cube grid for RISMiCal.
* **Perfect Grid Synchronization:** Directly controls Gaussian's cubegen via standard input to ensure the electrostatic potential grid perfectly matches the custom -rdelta3d * (ngrid3d/2) cubic grid required by RISMiCal.
* **Robust Coulomb Singularity Avoidance:** Automatically filters out solvent grid points that are dangerously close to any solute atoms (qvcore filter) to prevent NaN errors caused by division-by-zero during Gaussian's external charge integration.
* **Fully Automated SCF Cycles:** Automates the execution of Gaussian and rismical.x, updating atomic charges (ESP/MK) and solvent potentials iteratively until energy convergence is achieved.
* **Fortran Format Resilience:** Safely parses Fortran-style scientific notations (e.g., 3.150d0) and ignores unexpected string outputs from Fortran executables.

## Requirements
* **Python 3.6+**
* **NumPy**
* **SciPy** (Used for fast pairwise distance calculations)
* **Gaussian 16** (Commands g16, formchk, cubegen must be in $PATH)
* **RISMiCal** (Command rismical.x must be in $PATH)

## Usage
Run the script by passing the RISMiCal input file as an argument:

    python3 RISMiCal-QM.py input.inp

## Input File Configuration ($RISMICALQM)
Add the $RISMICALQM (or &RISMICALQM) namelist to your standard RISMiCal input file to control the QM/MM and SCF behaviors.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| qm | String | "g16" | The quantum chemistry engine to be used (currently only g16 is supported). |
| qmopt | String | "" | Gaussian Route section parameters. Use \ to represent newlines. (e.g., "%mem=16GB\%nproc=8\#p B3LYP/6-31G(d)"). If Pop= is not specified, Pop=MK will be added automatically. |
| qmpart | String | All atoms | Comma-separated list specifying the QM atoms. (e.g., "1,2,5,-10" means atoms 1, 2, and 5 to 10 are QM atoms. The rest are treated as MM). |
| scfconv | Float | 1e-4 | Threshold for total energy convergence (Hartree). |
| qvcutoff | Float | 1e-6 | Magnitude filter for solvent charges. Grid points with absolute charge < qvcutoff are ignored to speed up Gaussian integration. |
| qvcore | Float | 0.5 | Distance filter (in Angstroms). Solvent grid points within this radius from ANY solute atom are discarded to prevent Coulomb singularity in Gaussian. |
| ngrid3d | Integer| 128 | Number of grids per axis (Must match the value in $GRID3D). |
| rdelta3d| Float | 0.5 | Grid spacing in Angstroms (Must match the value in $GRID3D). |

## Generated Files
During the execution, the script generates various files, which are securely backed up iteratively:
* *.org.inp: Backup of the original input file.
* *.gjf / *.gout / *.chk / *.fchk: Gaussian I/O files.
* *.gout.gas / *.gout.qmmm_gas: Log files from the vacuum pre-steps.
* *.cub: Electrostatic potential ASCII cube (QM + MM potentials combined).
* *.qv: Filtered external point charges (solvent + MM atoms) given back to Gaussian.

---

<a id="japanese"></a>
## 概要
**RISMiCal-QM** は、**Gaussian 16**（量子化学計算プログラム）と **RISMiCal**（3D-RISMソルバ）を連携させ、QM/3D-RISM (KSDFT/3D-RISM, 3D-RISM-SCF, QM/MM/3D-RISM) 計算を実行するためのPythonラッパースクリプトです。溶質の電子状態、MM原子電荷、および溶媒の3D空間分布の間の自己無撞着場（SCF）サイクルを完全自動で最適化します。

## 主な機能
* **基準エネルギーの自動評価 (Pre-steps):** 溶媒を導入する前に、純粋なQM領域の真空エネルギー（E_gas）と、QM+MMの真空エネルギー（E_QMMM_gas）を自動計算し、QM-MM間の相互作用エネルギーを正確に評価します。
* **厳密なQM/MMポテンシャルの統合:** Gaussianの cubegen はQMのポテンシャルしか出力しないため、Python側でMM原子群が作るクーロンポテンシャルを計算し、3DのASCII Cubeグリッドに直接加算・合成してからRISMiCalに渡します。
* **グリッドの完全同期:** cubegen に標準入力経由で直接原点とステップ幅を渡すことで、RISMiCalが要求する -rdelta3d * (ngrid3d/2) の立方グリッドと静電ポテンシャルマップを厳密に一致させます。
* **クーロン特異点（発散）の自動回避:** Gaussianが外部電荷を読み込む際のゼロ除算エラー（NaN の発生）を防ぐため、溶質原子に近すぎる溶媒グリッド点を自動的に間引く距離フィルタ（qvcore）を搭載しています。
* **SCFサイクルの完全自動化:** Gaussianと rismical.x の実行を自動化し、全エネルギーが収束するまで原子電荷と溶媒ポテンシャルの更新を反復します。
* **Fortranフォーマットへの対応:** Fortran特有の指数表記（例: 3.150d0）を安全にパースします。

## 動作環境
* **Python 3.6+**
* **NumPy**
* **SciPy**（高速な距離計算に使用）
* **Gaussian 16**（g16, formchk, cubegen にパスが通っていること）
* **RISMiCal**（rismical.x にパスが通っていること）

## 使い方
RISMiCalのインプットファイルを引数に指定してスクリプトを実行します：

    python3 RISMiCal-QM.py input.inp

## インプットファイルの設定 ($RISMICALQM)
通常のRISMiCalインプットファイルに $RISMICALQM（または &RISMICALQM）ネームリストを追加し、計算条件を指定します。

| パラメータ | 型 | デフォルト値 | 説明 |
| :--- | :--- | :--- | :--- |
| qm | String | "g16" | 使用する量子化学計算エンジン（現在は g16 のみ対応）。 |
| qmopt | String | "" | Gaussianのインプット指定（Link 0コマンドおよびRouteセクション）。改行は \ で表現します（例: "%mem=16GB\%nproc=8\#p B3LYP/6-31G(d)"）。Pop= が指定されていない場合は、自動的に Pop=MK が付加されます。 |
| qmpart | String | 全原子 | QM領域として扱う原子のリストをカンマ区切りで指定します（例: "1,2,5,-10" は、1, 2番原子と、5〜10番原子をQMとして扱います。それ以外はMMとなります）。 |
| scfconv | Float | 1e-4 | 全エネルギーの収束閾値（Hartree単位）。 |
| qvcutoff | Float | 1e-6 | 溶媒電荷の絶対値によるフィルタリング閾値。Gaussianの積分計算を高速化するため、この値未満の微小な電荷は無視されます。 |
| qvcore | Float | 0.5 | 距離フィルタ（A単位）。Gaussian内でのクーロン発散を防ぐため、いずれかの溶質原子からこの半径内にある溶媒グリッド点は除外されます。 |
| ngrid3d | Integer| 128 | 1軸あたりのグリッド数（$GRID3D 内の設定値と一致させる必要があります）。 |
| rdelta3d| Float | 0.5 | グリッド間隔（A単位）（$GRID3D 内の設定値と一致させる必要があります）。 |

## 生成されるファイル群
実行中、以下のファイルが生成され、各イテレーションごとに安全に別名保存されます。
* *.org.inp: 実行時のオリジナルインプットファイルのバックアップ。
* *.gjf / *.gout / *.chk / *.fchk: Gaussianの入出力・チェックポイントファイル.
* *.gout.gas / *.gout.qmmm_gas: Pre-step（真空状態）のGaussianログファイル.
* *.cub: RISMiCal用に合成された静電ポテンシャルマップ（QM + MMの寄与を含む）のASCII形式Cubeファイル.
* *.qv: フィルタリング処理を経て、Gaussianに外部電荷として渡される溶媒（およびMM原子）の点電荷ファイル.

---
*Developed by Noriwo*