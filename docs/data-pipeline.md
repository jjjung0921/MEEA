# Data pipeline — collection and preprocessing as the paper does it

The scripts under `scripts/data/` rebuild every data artifact this repo reads,
from the sources named in the paper (Commun Chem 7:52, *Data availability* and
Methods → *Test dataset*). They replace the ad-hoc path this repo grew: the old
`scripts/build_policy_data.py`, `scripts/build_policy_from_tdc.py` and
`scripts/map_and_build_policy_from_tdc.py` built policy data from USPTO-50k (or
from PyTDC, with RXNMapper filling in atom maps), which is a different and much
smaller corpus than the one the paper trains on. Those scripts still work and
are left in place, but nothing here depends on them.

## What the paper uses

| artifact | source | paper |
|---|---|---|
| USPTO reactions, train/val/test split | the Retro* data package | "USPTO dataset is prepossessed and split into train/validation/test sets by Retro*" |
| building blocks | eMolecules free download | "commercially available molecules in eMolecules are used as the building block set" |
| USPTO test benchmark | Retro* (190 molecules, with reference routes) | Table 1 |
| nine extra test sets | MSU 2D data library | logS, logP, LD50, Ames, BBBP, DPP4, BACE, SVS, ClinTox |
| natural products | BioNavi-NP (388 molecules) | Synthesis of natural products |

Scale check from the paper: the USPTO training set holds **299,202 molecules**,
and MEEA* finds routes for 299,024 of them. If a rebuilt policy dataset comes
out at tens of thousands of rows, the corpus being read is USPTO-50k, not the
Retro* split.

## Stages

```bash
python scripts/data/prepare_all.py --reactions <corpus file>
```

runs them in order; each is also a standalone script.

### 1. `fetch_sources.py` — get the raw files

```bash
python scripts/data/fetch_sources.py --list     # what is needed and where from
python scripts/data/fetch_sources.py            # download what can be downloaded
python scripts/data/fetch_sources.py --check    # verify, no network
```

Only eMolecules has a scriptable URL (~350 MB). The Retro* package sits on
Dropbox and the MSU datasets are a browsable page, so those are reported as
manual downloads instead of being guessed at. Everything that lands is recorded
in `data/raw/manifest.json` with size and sha256.

Note: the paper cites the eMolecules `2023-12-01` snapshot, which eMolecules no
longer serves (404) — only the last few monthly snapshots stay online. The
fetcher defaults to a current one; override with `MEEA_EMOLECULES_SNAPSHOT`.

### 2. `prepare_retro_star.py` — unpack the Retro* package

Extracts `data/raw/retro_data.zip`, then **reports what it found** in
`data/raw/retro_star_layout.json` (building blocks, test routes, one-step
model, candidate reaction corpora) and copies:

- building blocks → `prepare_data/origin_dict.csv`
- `routes_possible_test_hard.pkl` → `prepare_data/`

Pass `--building-blocks` / `--routes` if the names in the package differ from
what it guessed. The reaction corpus it lists is the `--reactions` argument of
the next two stages.

### 3. `extract_templates.py` — the label space

```bash
python scripts/data/extract_templates.py --reactions <corpus> --min-count 1
```

rdchiral template extraction over the **training split only** — a template
mined from a validation or test reaction leaks that reaction into the policy's
label set. Writes `prepare_data/uspto_template.pkl.gz` and
`prepare_data/template_stats.json` (extraction failures, unique templates,
coverage). Requires atom-mapped reactions; if mapping is absent, every
extraction fails and the stage stops instead of writing an empty label space.

### 4. `build_policy_dataset.py` — the policy training data

```bash
python scripts/data/build_policy_dataset.py --reactions <corpus>
```

Writes four files, all from the same rows:

- `prepare_data/policyTrain.pkl.gz`, `policyTest.pkl.gz` — `{'smiles': packed
  Morgan fp, 'template': label}`, the schema `policyNet.py` already reads
- `prepare_data/policySmilesTrain.pkl.gz`, `policySmilesTest.pkl.gz` —
  `{'smiles': [SMILES], 'fp': packed, 'template': label}`

Both representations covering identical rows is what makes the
fingerprint-vs-SMILES backbone comparison controlled. `scripts/build_smiles_policy_data.py`,
which recovered SMILES after the fact by matching fingerprints against
`uspto50k.tab` and lost 11% of the rows, is no longer needed once this stage has
run.

Atom maps are stripped from the product before fingerprinting — the model is
given the molecule, not the mapping. The split follows the corpus when it states
one, otherwise rows are assigned by a hash of the product SMILES, so the same
molecule always lands on the same side.

### 5. `build_building_blocks.py` — eMolecules → `origin_dict.csv`

```bash
python scripts/data/build_building_blocks.py [--max-atoms 0]
```

Canonicalises every SMILES before writing, because the search decides a
molecule is purchasable by string equality against this set: an uncanonicalised
building block can never match. Writes the `(,mol,idx)` csv that
`prepare_starting_molecules` reads, plus a stats file. Use this when you want
the paper's building block set rather than the one shipped in the Retro*
package.

### 6. `filter_test_molecules.py` — the extra test sets

```bash
python scripts/data/filter_test_molecules.py --input data/raw/mol_props/logS.csv \
    --name logS --reactions <corpus> --policy ./saved_model/policy_model.ckpt
```

Applies the paper's three removal rules:

1. molecules present in the USPTO database or in the building block set
2. molecules solved by a heuristic BFS planner within 50 steps
3. molecules solvable by the one-step model in a single step

Rule 1 needs no model. Rules 2 and 3 call the expansion policy, so they run only
with `--policy`; without it the stage applies rule 1, records that the others
were skipped, and still writes `test_dataset/<name>.pkl`. Comparing the kept
counts against Supplementary Table 2 of the paper is the check that the
filtering matched.

## Output map

```
data/raw/                      downloaded sources + manifest.json
prepare_data/origin_dict.csv   building blocks (search reads this)
prepare_data/uspto_template.pkl.gz   template label space
prepare_data/policy{Train,Test}.pkl.gz          fingerprint policy data
prepare_data/policySmiles{Train,Test}.pkl.gz    SMILES policy data
prepare_data/routes_possible_test_hard.pkl      USPTO benchmark routes
test_dataset/<name>.pkl        filtered test molecules
prepare_data/*_stats.json      per-stage counts
```

## What this pipeline does not do

The cost estimator's training data is not built here. The paper constructs it by
running MEEA* over the USPTO training molecules and collecting the synthetic
routes it finds (299,024 of 299,202), with the synthetic cost of a reaction
taken as `log p_r(a|m)` from the expansion policy. That is a search run, not a
download, and it needs a trained policy first. `scripts/build_value_data_from_routes.py`
fabricates stand-in value data with heuristic costs and is not the paper's
procedure.
