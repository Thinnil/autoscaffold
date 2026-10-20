# AutoScaffold Suite

Interactive computational protein design pipelines automating **RFdiffusion**, **ProteinMPNN**, and **ColabFold** (AlphaFold-Multimer).

AutoScaffold provides a unified, production-grade terminal interface (TUI) and modular CLI commands for generative biomolecular engineering, de novo enzyme scaffolding, discontinuous binding surface scaffolding, receptor trap engineering, and automated AlphaFold validation.

---

## 🌟 Highlights & Features

- **Unified Master Suite (`autoscaffold_suite.sh`)**:
  - Interactive ANSI terminal menu (TUI) with clean navigation and signal traps.
  - Standalone and CLI sub-pipeline execution (`./autoscaffold_suite.sh motif`, `./autoscaffold_suite.sh binder`, `./autoscaffold_suite.sh discontinuous`, `./autoscaffold_suite.sh chimera`).
  - System diagnostics pre-flight checks.
- **🧬 Chimera Scaffolding from Multi-Chain PDB or PyMOL Sessions (`.pse`)**:
  - **Dual-Input Capability**: Inspects and extracts individual chains or molecular objects from multi-chain PDBs or PyMOL session files (`.pse`) with headless PyMOL and standalone fallback parsers.
  - **Automatic Entity Detection**: Identifies chain IDs, sequence lengths, residue intervals, and sequence strings without external dependencies.
  - **Interactive Structural Selection**: Seamlessly guides the user to select Donor (active motif) and Recipient / Vessel chassis.
  - **4 Chimeric Architecture Modes**: `replace` (graft into excised vessel region), `ligate_nterm` (N-terminal fusion), `ligate_cterm` (C-terminal fusion), and `insert_loop` (internal loop insertion).
  - **Coordinate Pinning**: Enforces `denoiser.noise_scale_ca=0.0` and `denoiser.noise_scale_frame=0.0` in RFdiffusion to strictly freeze catalytic / epitope 3D geometry.
  - **Fixed Residue Masking**: Locks donor motif residues during ProteinMPNN sequence design to preserve active site identity.
  - **Automated Validation**: Kabsch SVD C-alpha alignment (< 1.0 Å RMSD cutoff) and pLDDT quality metrics.
- **🧬 Motif-Constrained Scaffolding for Discontinuous Binding Surfaces**:
  - **Euclidean Contig Mapping**: Computes 3D distances between adjacent motif segment termini and enforces minimum loop lengths based on extended chain geometry (3.8 Å/residue).
  - **Dynamic Residue Budgeting**: Distributes scaffold lengths dynamically across $N+1$ segments around $N$ discontinuous motif fragments.
  - **Coordinate Pinning**: Enforces `denoiser.noise_scale_ca=0.0` and `denoiser.noise_scale_frame=0.0` in RFdiffusion to prevent catalytic or epitope drift.
  - **Targeted Sequence Design**: Uses ProteinMPNN with fixed-position locking for motif positions while redesigning core and exterior residues.
  - **Automated ColabFold Validation & Filtering**: Evaluates designs using pure NumPy Kabsch C-alpha RMSD, scaffold-only mean pLDDT, and inter-chain ipTM.
- **Hardware & Environment Diagnostics (`diagnostics.sh`)**:
  - Automatically assesses host OS, CPU cores, system RAM, NVIDIA GPU/VRAM, CUDA version, and CLI tools.
  - Verifies presence of Conda/Mamba/Micromamba environments (`SE3nv`, `mlfold`, `colabfold`).
- **Autonomous Local Installer (`install_dependencies.sh`)**:
  - Can bootstrap standalone `micromamba` (zero root required).
  - Automatically clones and sets up RFdiffusion and ProteinMPNN into `./tools/`.
  - Downloads official RFdiffusion model checkpoints.
  - Creates self-contained environments in `./envs/`.
- **In-Place Self-Upgrade Engine (`self_update.sh`)**:
  - Checks remote GitHub repository for updates and compares commit hashes.
  - Automatically backs up active scripts (`.backup_<timestamp>`).
  - Pulls updates and reapplies executable permissions without touching user weights or configs.
- **Robust Algorithmic Corrections**:
  - **Dynamic Chain Extraction**: Scaffolds motif residues for any user-selected chain (not hardcoded to chain A).
  - **Target-Binder Chain Identity**: Correctly preserves target chain identities while letting ProteinMPNN redesign the binder.
  - **Safe String & Multi-chain Handling**: Python-based FASTA complex formatter replaces brittle bash grep/cut.
  - **Safe Execution**: Array-based argument expansion replacing insecure `eval` calls.

---

## 🚀 Interactive TUI & Pipelines

Launch the main suite:
```bash
chmod +x autoscaffold_suite.sh autoscaffold.sh de_novo.sh discontinuous_scaffold.sh chimera_scaffold.sh scripts/*.sh scripts/*.py
./autoscaffold_suite.sh
```

### Main Menu Options:
```text
======================================================================
        🧬 AUTOSCAFFOLD SUITE: AUTONOMOUS PROTEIN ENGINEERING 🧬        
       RFdiffusion  •  ProteinMPNN  •  ColabFold  •  PyMOL / PSE       
======================================================================
Select a workflow to launch:
  [1] 🏗️  De Novo Motif Scaffolding (Active site & epitope contigs)
  [2] 🎯 De Novo Target-Binder Design (Interface hotspot scaffolding)
  [3] 🧩 Discontinuous Binding Surface Scaffolding
  [4] 🧬 Chimera Scaffolding: Graft Motif into Vessel Protein (PDB/PSE)
  [5] 🩺 System Diagnostics & Environment Pre-Flight Checks
  [6] 📦 Automatic Dependency & Environment Installer
  [7] 🔄 In-Place Self-Update & GitHub Upgrade
  [8] 🚪 Exit
```

---

## 💻 CLI & Standalone Usage

You can run sub-pipelines directly via CLI flags or standalone scripts:

### 1. 🧬 Chimera Scaffolding: Graft Motif into Vessel Protein (PDB / PyMOL .pse)
Preserve the exact 3D tertiary structure and native sequence of an active motif/epitope ("Donor") while grafting or ligating it into a host scaffold protein ("Recipient / Vessel"):

```bash
# Launch interactive wizard via Master Suite CLI:
./autoscaffold_suite.sh chimera

# Or non-interactive CLI execution with options:
./autoscaffold_suite.sh chimera \
    --input complex.pdb \
    --donor_id A \
    --vessel_id B \
    --motif_residues "30-45" \
    --mode replace \
    --replace_range "55-70" \
    --linker_n "6-10" \
    --linker_c "6-10" \
    --num_designs 10 \
    --out_dir outputs/chimera_scaffold

# Or using PyMOL session file (.pse):
./chimera_scaffold.sh \
    --input session.pse \
    --donor_id 1 \
    --vessel_id 2 \
    --motif_residues "30-45,78-82" \
    --mode insert_loop \
    --split_after 54 \
    --num_designs 10

# Or standalone Python orchestrator:
python scripts/chimera_scaffold.py \
    --input_file session.pse \
    --donor_id DONOR_OBJ \
    --vessel_id VESSEL_OBJ \
    --motif_residues "30-45" \
    --chimera_mode replace \
    --vessel_replace_range "55-70" \
    --linker_n "6-10" \
    --linker_c "6-10" \
    --num_designs 10 \
    --mpnn_seqs_per_bb 2 \
    --mpnn_batch_size 1 \
    --rmsd_threshold 1.0 \
    --plddt_threshold 75.0 \
    --out_dir outputs/chimera_scaffold
```

#### Supported Chimera Architecture Modes:
- **`replace`**: Excises a designated residue interval on the vessel protein (e.g. `--replace_range 55-70`) and grafts the donor active motif into that location, bridged by de novo RFdiffusion linkers.
- **`ligate_nterm`**: Fuses the donor active motif to the N-terminus of the vessel framework via a de novo linker.
- **`ligate_cterm`**: Fuses the donor active motif to the C-terminus of the vessel framework via a de novo linker.
- **`insert_loop`**: Splits the vessel protein at a specific residue index (e.g. `--split_after 54`) and inserts the donor active motif into the loop with flanking linkers.

#### Key Verification & Quality Filtering:
- **Kabsch RMSD Cutoff**: $\le 1.0$ Å against reference crystal motif coordinates (enforces strict tertiary geometry preservation).
- **Coordinate Pinning**: `noise_scale_ca=0.0` and `noise_scale_frame=0.0` preventing drift during RFdiffusion.
- **Fixed-Position Locking**: ProteinMPNN mask locks donor motif amino acids to preserve sequence identity.
- **pLDDT Filtering**: $\ge 75.0$ global confidence threshold.

### 2. Discontinuous Contact Motif Scaffolding
Scaffold non-contiguous binding residues or catalytic geometries into a single continuous chain:
```bash
# Using the Master Suite CLI:
./autoscaffold_suite.sh discontinuous [complex.pdb] [motif_chain] [motif_residues] [target_chain]

# Or dedicated shell wrapper:
./discontinuous_scaffold.sh [complex.pdb] [motif_chain] [motif_residues] [target_chain]

# Or standalone Python orchestrator:
python scripts/discontinuous_motif_scaffold.py \
    --complex_pdb complex.pdb \
    --target_chain A \
    --motif_chain B \
    --motif_residues "310-313,338-347,594-596,612-625" \
    --min_total_len 70 \
    --max_total_len 95 \
    --num_designs 50 \
    --mpnn_seqs_per_bb 4 \
    --mpnn_batch_size 4 \
    --colabfold_batch_size 0 \
    --colabfold_num_recycles 3 \
    --colabfold_num_models 1 \
    --out_dir outputs/discontinuous_scaffold \
    --plddt_cutoff 80.0 \
    --rmsd_cutoff 1.2 \
    --iptm_cutoff 0.70
```

#### Filtering Thresholds & Criteria:
- **Scaffold pLDDT**: $\ge 80.0$ (mean pLDDT calculated strictly over newly scaffolded residues, excluding target chain).
- **Motif Kabsch RMSD**: $\le 1.2$ Å (C-alpha coordinate RMSD between AF2 model motif residues and crystal seed coordinates).
- **Interface ipTM**: $\ge 0.70$ (predicted interface TM score for target:scaffold complex).

### 3. De Novo Motif Scaffolding
```bash
./autoscaffold_suite.sh motif [source.pdb] [chain_id] [num_motifs]
# Or standalone:
./autoscaffold.sh [source.pdb] [chain_id] [num_motifs]
```

### 4. Target-Binder Design & ColabFold
```bash
./autoscaffold_suite.sh binder [target.pdb] [target_chain] [residue_range]
# Or standalone:
./de_novo.sh [target.pdb] [target_chain] [residue_range]
```

### 5. Diagnostics & Health Check
```bash
./autoscaffold_suite.sh diagnostics
```

### 6. Automated Dependency & Environment Installer
Automatically bootstraps micromamba, clones RFdiffusion & ProteinMPNN into `./tools/`, downloads weights, and configures environments into `./envs/`:
```bash
./autoscaffold_suite.sh install
```

### 7. In-Place Self-Update & GitHub Upgrade
Checks remote GitHub repository for new commits, backs up active scripts, and merges updates:
```bash
./autoscaffold_suite.sh update
```

---

## ⚙️ Configuration (`config.env`)

AutoScaffold automatically reads `config.env` if present, falling back to `config.env.example` or shell environment variables:

```bash
# Path to RFdiffusion repository
RFDIFF_DIR="$HOME/programs/RFdiffusion"

# Path to ProteinMPNN repository
MPNN_DIR="$HOME/programs/ProteinMPNN"

# Conda / Mamba Environment Names
CONDA_ENV_RFDIFF="SE3nv"
CONDA_ENV_MPNN="mlfold"
CONDA_ENV_COLABFOLD="colabfold"

# Local directories (if using local self-contained installer)
LOCAL_TOOLS_DIR="$SCRIPT_DIR/tools"
LOCAL_ENVS_DIR="$SCRIPT_DIR/envs"
```

---

## 📂 Output Directory Structure

Each design run organizes files systematically:

```text
<output_dir>/
├── motif_seed.pdb               # Extracted crystal motif coordinates
├── complex_reference.pdb        # Cleaned target + motif complex input
├── contig_plan.json             # Geometric loop calculations and contig formula
├── metrics_summary.csv          # Ranked scoring table (pLDDT, RMSD, ipTM, pTM)
├── results/                     # Top passing candidates meeting all filter criteria
├── 1_rfdiffusion/               # Backbones (.pdb), trajectory data (.trb), run script
├── 2_mpnn_inputs/               # Fixed-position JSONLs, parsed chains, MPNN runner
├── 3_mpnn_outputs/              # Designed sequences (.fa) and scores
└── 4_colabfold_validation/      # ColabFold multimer FASTA and prediction models
```

---

## 📜 License
MIT License.
