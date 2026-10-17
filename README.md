# AutoScaffold

Interactive computational protein design pipelines automating **RFdiffusion**, **ProteinMPNN**, and **ColabFold** (AlphaFold-Multimer).

---

## 🚀 Pipelines Overview

This repository provides two specialized workflows for generative protein engineering:

### 1. `autoscaffold.sh` — De Novo Active Site & Epitope Motif Scaffolding
- **Application**: Enzyme engineering, catalytic triads, binding site grafting, epitope scaffolding.
- **Workflow**:
  1. Interactive topology assistant constructs `contigmap.contigs` formulas with flanking scaffold regions.
  2. Runs **RFdiffusion** in scaffold mode to build 3D backbone folds holding catalytic residues in accurate geometry.
  3. Automatically parses `.trb` files (`con_hal_idx0`) to map the original active site coordinates.
  4. Generates a `fixed_positions_jsonl` dictionary for **ProteinMPNN** to redesign the scaffold while locking the active site amino acid identities.

### 2. `de_novo.sh` — Target-Binder Design & Validation
- **Application**: Designing custom binding proteins ("Pac-Man" binders, receptor traps, therapeutic neutralizers).
- **Workflow**:
  1. Validates target PDB and automatically resolves residue ranges for the target chain.
  2. Generates binder backbones in complex with the target using **RFdiffusion** (with optional hotspot targeting).
  3. Fixes the target chain and designs sequences specifically for the binder chain using **ProteinMPNN**.
  4. Formats multimer FASTA (`target:binder`) and triggers **ColabFold** (`colabfold_batch`) to perform end-to-end AlphaFold-Multimer validation.
  5. Retains the top-ranked model (`rank_001`) and cleans up redundant files to save disk storage.

---

## 🛠️ Prerequisites & Environments

The pipelines assume standard Conda environments installed on your machine or cluster:

| Tool | Conda Environment | Default Directory |
| :--- | :--- | :--- |
| **RFdiffusion** | `SE3nv` | `$HOME/programs/RFdiffusion` |
| **ProteinMPNN** | `mlfold` | `$HOME/programs/ProteinMPNN` |
| **ColabFold** | `colabfold` | System PATH (`colabfold_batch`) |

> [!TIP]
> You can override the tool directories at runtime by setting environment variables before running the scripts:
> ```bash
> export RFDIFF_DIR="/path/to/RFdiffusion"
> export MPNN_DIR="/path/to/ProteinMPNN"
> ```

---

## 💻 Quick Start

Make the scripts executable:
```bash
chmod +x autoscaffold.sh de_novo.sh
```

### Running Motif Scaffolding:
```bash
./autoscaffold.sh
```

### Running Binder Design & ColabFold Validation:
```bash
./de_novo.sh
```

---

## 📂 Output Directory Structure

Each pipeline produces an organized run directory:

```text
<output_dir>/
├── 1_rfdiffusion/              # Backbones (.pdb) and trajectory data (.trb)
├── 2_mpnn_inputs/              # Parsed chains and fixed-residue JSONLs
├── 3_mpnn_outputs/             # Designed sequences (.fa) and scores
└── 4_colabfold_validation/     # (de_novo only) AlphaFold-Multimer top models
```

---

## 📜 License
MIT License.
