# Technical Specification: Chimera Scaffolding from PDB or PyMOL (.pse) Files

- **Document Version**: 1.0.0
- **Status**: Approved Technical Specification (Prompt Polisher Step)
- **Target Subsystem**: `autoscaffold` suite (`autoscaffold_suite.sh`, `scripts/chimera_scaffold.py`, `scripts/pse_pdb_extractor.py`)
- **Primary AI & Structural Engines**: RFdiffusion, ProteinMPNN, PyMOL/Biopython parser, Kabsch Alignment / RMSD Validator

---

## 1. Executive Summary & Objective

The objective of the **Chimera Scaffolding Workflow** is to generate chimeric protein models that preserve the exact tertiary 3D structure and amino acid sequence of a designated active motif/epitope (the "Donor"), while grafting and ligating it into a host scaffold protein (the "Recipient / Vessel"). De novo linkers and connecting junctions are designed using **RFdiffusion**, and junction/interface residues are redesigned with **ProteinMPNN** while locking donor motif residues in place.

Users can supply either:
1. **Case A: Multi-chain PDB file** containing both donor and recipient structures in separate chains.
2. **Case B: PyMOL session file (`.pse`)** containing two or more distinct molecular objects.

The system automatically detects, parses, and presents the structures/chains to the user, prompts for key design parameters (donor motif residues, recipient protein, insertion/ligation mode), constructs the exact RFdiffusion contig string, coordinates ProteinMPNN fixed-residue masking, and verifies generated models with Kabsch RMSD and clash filters.

---

## 2. Input Formats & Automatic Detection Engine

### 2.1 Case A: Multi-Chain PDB Parsing
When a `.pdb` file is loaded:
- **Parser Engine**: Biopython `PDBParser(QUIET=True)` or custom high-speed PDB line parser scanning `ATOM` / `HETATM` records.
- **Extraction Metrics**:
  - Distinct Chain IDs ($A, B, C, \dots$).
  - Per-chain residue count, start and end residue sequence numbers (resseq and insertion code).
  - Sequence identity string (1-letter code reconstructed from 3-letter codes).
  - Missing residues or non-canonical backbone checks.
- **Output Data Structure**:
  ```python
  {
      "chains": {
          "A": {
              "length": 142,
              "range": (1, 142),
              "sequence": "MKTIIALSYIFCLVFA...",
              "source_file": "input.pdb"
          },
          "B": {
              "length": 88,
              "range": (1, 88),
              "sequence": "EVQLVESGGGLVQPGG...",
              "source_file": "input.pdb"
          }
      }
  }
  ```

### 2.2 Case B: PyMOL Session (`.pse`) Parsing
PyMOL session files (`.pse`) are gzip-compressed Python pickler dumps containing PyMOL's internal state, molecular objects, selections, and representations.

- **Extraction Strategies**:
  1. **Primary Strategy (PyMOL Python API / Headless CLI)**:
     If PyMOL is installed in the conda environment or available on `PATH`:
     ```python
     from pymol import cmd
     cmd.load(pse_path)
     objects = cmd.get_names("objects")
     for obj in objects:
         cmd.save(f"extracted_{obj}.pdb", obj)
     ```
  2. **Standalone Fallback Strategy (Direct Decompression & Unpickling without PyMOL)**:
     When PyMOL binary is not directly installed in the runtime environment:
     - Decompress using `gzip.GzipFile` or `zlib.decompress`.
     - PyMOL `.pse` stores molecular data in a dictionary under key `'names'` where each entry is a list containing the object name, type (type 1 = molecule), and coordinate arrays/atom definitions.
     - A robust fallback extractor (`scripts/pse_pdb_extractor.py`) parses the molecule dictionary, extracts atom records (name, chain, resn, resi, x, y, z, b-factor), and writes each individual object to an isolated, standardized PDB file (`extracted_obj1.pdb`, `extracted_obj2.pdb`).

- **Detected Objects Presentation**:
  The user is shown a table of detected objects, their chain IDs, residue ranges, and extracted lengths.

---

## 3. Interactive Workflow & User Selection Engine

The CLI/script presents an interactive terminal wizard (or accepts CLI flags for non-interactive automation):

```
======================================================
 🧬 AUTOSCAFFOLD: CHIMERA SCAFFOLDING WIZARD 🧬
======================================================
[+] Detected 2 structural units in 'session.pse':
  [1] Object/Chain: DONOR_TARGET (Length: 120 aa, Range: 1-120)
  [2] Object/Chain: VESSEL_HOST  (Length: 245 aa, Range: 1-245)

▶ Step 1: Select Donor Object/Chain (motif to preserve) [1/2]: 1
▶ Step 2: Select Donor Residue(s) / Range(s) to fix in 3D:
   Examples: '25-42', '12,15,48-52', 'A30-A45'
   Selection: 30-45, 78-82

▶ Step 3: Select Recipient / Vessel Object/Chain: 2
▶ Step 4: Select Chimera Architecture Mode:
   [1] Graft / Replacement: Replace a vessel region with donor motif via de novo linkers
   [2] Terminal / Loop Ligation: Fuse donor motif to vessel terminus/loop via bridging linkers
   [3] Surface Pocket Scaffolding: Anchor donor motif onto vessel core, de novo rebuilding the interface
   Choice [1/2/3]: 1

▶ Step 5 (Mode 1): Specify Vessel Segment to Replace / Excised Range (e.g. 50-75): 55-70
▶ Step 6: Linker Flexibility (e.g. 5-15 aa N-terminal, 8-20 aa C-terminal):
   Linker 1 (Vessel -> Motif): 5-12
   Linker 2 (Motif -> Vessel): 5-12
```

### 3.1 The Three Chimeric Modes
1. **Mode 1: Graft / Segment Replacement**
   - A variable, flexible, or non-essential loop/domain of the recipient vessel is excised.
   - The donor active motif is inserted in its place.
   - RFdiffusion generates de novo linker bridges at both N- and C-junctions connecting the retained vessel segments with the donor motif.
2. **Mode 2: Terminal / Loop Ligation**
   - The donor motif is ligated directly to either the N-terminus, C-terminus, or an internal loop insertion point of the vessel.
   - RFdiffusion designs a structured bridging adapter (e.g., 10-25 aa) that positions the motif in a fixed spatial relationship relative to the vessel.
3. **Mode 3: Surface Pocket Scaffolding**
   - The donor motif consists of a discontinuous catalytic triad or binding pocket.
   - The vessel core is retained as the structural chassis.
   - RFdiffusion rebuilds the surrounding secondary structure elements to hold the donor residues rigidly in their precise native 3D geometry.

---

## 4. Contig Formulation for RFdiffusion Chimera Design

RFdiffusion defines chimeric architectures using contig syntax `[chain][start]-[end]/[linker_range]/[chain][start]-[end]`.

### Contig Rules for Chimera Generation:
Let:
- Vessel Chain be $V$, retained ranges: $V_{start1}-V_{end1}$ and $V_{start2}-V_{end2}$.
- Donor Chain be $D$, fixed motif range: $D_{m\_start}-D_{m\_end}$.
- De novo designed linkers be specified with min-max lengths, e.g., `5-12`.

#### Example 1: Replacement / Graft (Mode 1)
- Vessel: Residues 1-54 retained, 55-70 excised, 71-245 retained.
- Donor motif: Residues 30-45.
- Linkers: 6-10 aa on each flank.
- **Resulting RFdiffusion Contig**:
  ```text
  'contigmap.contigs=[V1-54/6-10/D30-45/6-10/V71-245]'
  ```

#### Example 2: Discontinuous Donor Motif Grafting
- Donor motif has two separated functional segments: $D15-22$ and $D80-85$.
- Donor motif internal spatial distance is preserved:
  ```text
  'contigmap.contigs=[V1-60/8-15/D15-22/4-8/D80-85/8-15/V95-200]'
  ```

#### Example 3: C-Terminal Ligation (Mode 2)
- Vessel: Full chain $V1-245$.
- Donor: $D1-40$.
- Connecting structured linker: 12-18 aa.
  ```text
  'contigmap.contigs=[V1-245/12-18/D1-40]'
  ```

---

## 5. Targeted Sequence Design with ProteinMPNN

When designing sequences for the newly diffused backbone:
1. **Donor Motif Residues**:
   - Strictly **LOCKED** (both 3D backbone and 1-letter amino acid sequence identity are identical to native).
   - In ProteinMPNN: Set in `fixed_positions.jsonl` so sampling is frozen at native amino acid identities.
2. **Recipient Vessel Residues**:
   - User choice:
     - *Option A (Default)*: Keep vessel core fixed, only allow redesign within $8.0\text{ \AA}$ contact sphere of the junction/linker.
     - *Option B (Full Redesign)*: Redesign vessel surface to optimize folding stability and solubility of the chimera.
3. **De Novo Linkers and Junctions**:
   - Fully redesigned using ProteinMPNN with soluble/temperature-controlled sampling (e.g. sampling temperature $T=0.1$ for high stability, $T=0.2$ for diversity).

---

## 6. Verification, Validation & Filtering Pipeline

Generated chimera models must pass rigorous automated quality control before being accepted:

1. **Kabsch RMSD Calculation (`scripts/kabsch_rmsd.py`)**:
   - Superimpose the generated chimera onto the original donor structure using only the $C_\alpha$ (and backbone $N, C, O$) atoms of the designated motif residues.
   - Threshold: **Motif RMSD $\le 1.0\text{ \AA}$** (strictly $< 0.8\text{ \AA}$ for catalytic triads).
2. **Backbone AlphaFold2 / ESMFold Quality (pLDDT & pTM)**:
   - Predict structural model from the ProteinMPNN generated sequence.
   - Filter:
     - Global $\text{pLDDT} \ge 75$
     - Motif local $\text{pLDDT} \ge 85$
     - Scaffold-to-motif alignment $\text{RMSD} \le 1.5\text{ \AA}$
3. **Steric Clash & Geometry Checks**:
   - Detect atomic overlaps $< 2.2\text{ \AA}$ between non-bonded heavy atoms.
   - Verify Ramachandran phi/psi angles of designed linkers.

---

## 7. Integration Architecture into `autoscaffold_suite.sh`

### 7.1 Unified CLI & Interactive Menu
`autoscaffold_suite.sh` is updated with a dedicated mode:
- **CLI Syntax**:
  ```bash
  ./autoscaffold_suite.sh chimera \
      --input session.pse \
      --donor DONOR_OBJ:30-45,78-82 \
      --vessel VESSEL_OBJ \
      --mode graft \
      --replace-vessel 55-70 \
      --linker-n 6-10 \
      --linker-c 6-10 \
      --num-designs 10 \
      --outdir ./chimera_out
  ```
- **Interactive Mode**:
  If no flags are passed, `./autoscaffold_suite.sh` presents an interactive menu:
  ```text
  1) De Novo Motif Scaffolding
  2) Discontinuous Active Site Scaffolding
  3) Chimera Scaffolding (PDB or PyMOL .pse) [NEW]
  4) Model Quality & RMSD Validation
  5) Exit
  ```

### 7.2 Directory & Component Layout
```
autoscaffold/
├── autoscaffold_suite.sh               # Master CLI entrypoint & interactive menu
├── config.env.example                  # Environment configuration & binary paths
├── scripts/
│   ├── pse_pdb_extractor.py            # Extracts objects/chains from .pse and multi-chain PDBs
│   ├── chimera_scaffold.py             # Orchestrates RFdiffusion contig & ProteinMPNN chimera execution
│   ├── extract_fixed_motif.py          # Extracts designated motif coordinates and sequences
│   ├── kabsch_rmsd.py                  # High-precision Kabsch RMSD alignment & validation
│   └── format_complex_fasta.py         # Multi-chain FASTA and MPNN mask formatting
└── tests/
    └── test_chimera_pipeline.py        # Automated unit & integration tests
```

---

## 8. Step-by-Step Implementation Roadmap

1. **Step 1: Extractor Module (`scripts/pse_pdb_extractor.py`)**:
   Implement dual-mode parsing for multi-chain PDBs and `.pse` PyMOL sessions (headless PyMOL API with fallback zlib/pickle parser).
2. **Step 2: Chimera Scaffolding Engine (`scripts/chimera_scaffold.py`)**:
   Implement contig string assembly for Mode 1 (Graft), Mode 2 (Ligate), and Mode 3 (Pocket Scaffolding), with RFdiffusion and ProteinMPNN execution hooks.
3. **Step 3: Suite Integration (`autoscaffold_suite.sh`)**:
   Add interactive prompts, input validation, and headless execution pipeline to the suite bash script.
4. **Step 4: Quality Verification (`scripts/kabsch_rmsd.py`)**:
   Implement automated superimposition and RMSD scoring against native donor motif.
5. **Step 5: Testing & Verification (`tests/test_chimera_pipeline.py`)**:
   Write integration tests verifying PDB parsing, PSE extraction, contig generation, and RMSD calculation.
