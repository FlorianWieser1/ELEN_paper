#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J prep_AF3_DB
#SBATCH -o prep_AF3_DB.log
#SBATCH -e prep_AF3_DB.err
#SBATCH --gres gpu:1

import os
import sys
import glob
import json
import shutil
import logging
import argparse
import warnings
import subprocess

from Bio import BiopythonDeprecationWarning
from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.SeqUtils import seq1
from Bio.PDB.Superimposer import Superimposer

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

### Definitions
standard_protein_resnames = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "MSE", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
}
standard_dna_resnames = {"DA", "DT", "DG", "DC", "DI"}
standard_rna_resnames = {"A", "U", "G", "C", "I"}

conversion_map_dna = {
    "DA": "A",
    "DT": "T",
    "DG": "G",
    "DC": "C",
    "DI": "I"
}
conversion_map_rna = {
    "A": "A",
    "U": "U",
    "G": "G",
    "C": "C",
    "I": "I"
}

###############################################################################
def uppercase_letters():
    for c in range(ord('A'), ord('Z') + 1):
        yield chr(c)

###############################################################################
class ChainIDManager:
    def __init__(self):
        self.used_ids = set()
        self.generator = uppercase_letters()

    def get_chain_id_for_pdb_chain(self, pdb_chain_id: str) -> str:
        pdb_chain_id = pdb_chain_id.strip()
        if (len(pdb_chain_id) == 1 and pdb_chain_id.isalpha() and pdb_chain_id.isupper()
                and pdb_chain_id not in self.used_ids):
            self.used_ids.add(pdb_chain_id)
            return pdb_chain_id
        else:
            return self._next_letter()

    def get_new_chain_id(self) -> str:
        return self._next_letter()

    def _next_letter(self) -> str:
        while True:
            try:
                candidate = next(self.generator)
            except StopIteration:
                raise ValueError("Ran out of single-letter uppercase IDs (A-Z).")
            if candidate not in self.used_ids:
                self.used_ids.add(candidate)
                return candidate

###############################################################################
def compute_rmsd(fixed_file, moving_file):
    """
    Compute RMSD between CA atoms of two structures.
    Assumes both structures have similar ordering of CA atoms.
    """
    parser = PDBParser(QUIET=True)
    fixed_structure = parser.get_structure('fixed', fixed_file)
    moving_structure = parser.get_structure('moving', moving_file)

    fixed_atoms = []
    moving_atoms = []

    # Iterate over models, chains, residues in parallel
    for fixed_model, moving_model in zip(fixed_structure, moving_structure):
        for fixed_chain, moving_chain in zip(fixed_model, moving_model):
            for fixed_res, moving_res in zip(fixed_chain, moving_chain):
                # Use CA atoms for protein residues if available
                if 'CA' in fixed_res and 'CA' in moving_res:
                    fixed_atoms.append(fixed_res['CA'])
                    moving_atoms.append(moving_res['CA'])

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    return sup.rms

###############################################################################
def parse_pdb(path_pdb, ccd_lookup):
    parser = PDBParser(QUIET=True)
    structure_id = os.path.splitext(os.path.basename(path_pdb))[0]
    structure = parser.get_structure(structure_id, path_pdb)

    chain_id_mgr = ChainIDManager()

    chain_type_dict = {
        "protein": [],
        "dna": [],
        "rna": [],
        "ligand": {}
    }

    for model in structure:
        for chain in model:
            original_chain_id = chain.id
            safe_chain_id = chain_id_mgr.get_chain_id_for_pdb_chain(original_chain_id)

            protein_seq = []
            mods = []
            dna_seq = []
            rna_seq = []
            protein_pos = 0
            chain_ligand_map = {}

            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in ("HOH", "WAT"):
                    continue

                if resname in standard_protein_resnames:
                    protein_pos += 1
                    if resname == "MSE":
                        protein_seq.append("M")
                        mods.append({"ptmType": "MSE", "ptmPosition": protein_pos})
                    else:
                        protein_seq.append(seq1(resname))

                elif resname in standard_dna_resnames:
                    base = conversion_map_dna.get(resname, "")
                    dna_seq.append(base)

                elif resname in standard_rna_resnames:
                    base = conversion_map_rna.get(resname, "")
                    rna_seq.append(base)

                else:
                    ccd = resname
                    lig_chain_id = chain_id_mgr.get_new_chain_id()
                    chain_ligand_map.setdefault(ccd, set()).add(lig_chain_id)

            if protein_seq:
                chain_type_dict["protein"].append(
                    (safe_chain_id, "".join(protein_seq), mods)
                )
            if dna_seq:
                chain_type_dict["dna"].append(
                    (safe_chain_id, "".join(dna_seq))
                )
            if rna_seq:
                chain_type_dict["rna"].append(
                    (safe_chain_id, "".join(rna_seq))
                )

            for ccd, ligand_ids in chain_ligand_map.items():
                if ccd not in chain_type_dict["ligand"]:
                    chain_type_dict["ligand"][ccd] = set()
                chain_type_dict["ligand"][ccd].update(ligand_ids)

    af3_dict = build_af3_dict(structure_id, chain_type_dict)
    return af3_dict

###############################################################################
def build_af3_dict(structure_id, chain_type_dict):
    sequences_list = []

    # --- Updated Protein Entry Handling ---
    if chain_type_dict["protein"]:
        if len(chain_type_dict["protein"]) == 1:
            # Only one protein chain: output it directly.
            ch_id, seq_str, mods = chain_type_dict["protein"][0]
            protein_entry = {
                "protein": {
                    "id": [ch_id],
                    "sequence": seq_str
                }
            }
            if mods:
                protein_entry["protein"]["modifications"] = mods
            sequences_list.append(protein_entry)
        else:
            # Multiple protein chains exist.
            # Check if all chains have the same sequence.
            all_seqs = [seq for (_, seq, _) in chain_type_dict["protein"]]
            if all(seq == all_seqs[0] for seq in all_seqs):
                # They are identical; combine the chain IDs.
                all_chain_ids = [ch_id for (ch_id, _, _) in chain_type_dict["protein"]]
                mods = chain_type_dict["protein"][0][2]  # assume modifications are identical
                protein_entry = {
                    "protein": {
                        "id": all_chain_ids,
                        "sequence": all_seqs[0]
                    }
                }
                if mods:
                    protein_entry["protein"]["modifications"] = mods
                sequences_list.append(protein_entry)
            else:
                # Chains are different: output a separate entry for each.
                for (ch_id, seq_str, mods) in chain_type_dict["protein"]:
                    protein_entry = {
                        "protein": {
                            "id": [ch_id],
                            "sequence": seq_str
                        }
                    }
                    if mods:
                        protein_entry["protein"]["modifications"] = mods
                    sequences_list.append(protein_entry)
    # --- End Updated Protein Handling ---

    if chain_type_dict["dna"]:
        combined_ids = []
        combined_seq = []
        for (ch_id, seq_str) in chain_type_dict["dna"]:
            combined_ids.append(ch_id)
            combined_seq.append(seq_str)
        dna_entry = {
            "dna": {
                "id": combined_ids,
                "sequence": "".join(combined_seq)
            }
        }
        sequences_list.append(dna_entry)

    if chain_type_dict["rna"]:
        combined_ids = []
        combined_seq = []
        for (ch_id, seq_str) in chain_type_dict["rna"]:
            combined_ids.append(ch_id)
            combined_seq.append(seq_str)
        rna_entry = {
            "rna": {
                "id": combined_ids,
                "sequence": "".join(combined_seq)
            }
        }
        sequences_list.append(rna_entry)

    if chain_type_dict["ligand"]:
        for ccd, ligand_ids in sorted(chain_type_dict["ligand"].items()):
            ligand_entry = {
                "ligand": {
                    "id": sorted(ligand_ids),
                    "ccdCodes": [ccd]
                }
            }
            sequences_list.append(ligand_entry)
    af3_dict = {
        "name": structure_id,
        "sequences": sequences_list,
        "modelSeeds": [1]
    }
    return af3_dict

###############################################################################
def get_af3_jsons_from_pdbs(path_pdb, outpath_json):
    ccd_lookup = {}
    ccd_json_path = "CCDcodes.json"
    if os.path.isfile(ccd_json_path):
        with open(ccd_json_path, "r") as f:
            ccd_lookup = json.load(f)

    af3_data = parse_pdb(path_pdb, ccd_lookup)
    with open(outpath_json, "w") as outfh:
        json.dump(af3_data, outfh, indent=2)
    return outpath_json 

def run_af3(path_json, outpath):
    command = [
        "/home/florian_wieser/.conda/envs/af3/bin/python",
        "/home/florian_wieser/software/alphafold3/run/run_af3.py",
        "--json_path", path_json,
        "--output_dir", outpath,
        "--flash_attention_implementation", "xla",
        "--cuda_compute_7x", "1",
        "--num_diffusion_samples", "1",
        "--run_mmseqs"
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return e

def cif_to_pdb(cif_path, pdb_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", cif_path)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)
    logging.info(f"Converted {cif_path} to {pdb_path}")

###############################################################################
def main(args):
    if args.overwrite and os.path.exists(args.outpath):
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    outpath_models = os.path.join(args.outpath, "AF3_models")
    os.makedirs(outpath_models, exist_ok=True)

    pdb_files = glob.glob(os.path.join(args.inpath, "*.pdb"))
    if not pdb_files:
        logging.info(f"No PDB files found in {args.inpath}.")
        return

    too_many_file = "predict_AF3_too_many_elements.txt"
    error_file = "predict_AF3_pdb_errors.txt"
    high_rmsd_file = "predict_AF3_high_rmsd.txt"

    for path_pdb in pdb_files:
        pdb_id = os.path.splitext(os.path.basename(path_pdb))[0]
        try:
            logging.info(f"Processing {path_pdb}.")
            outpath_pdb = os.path.join(args.outpath, pdb_id)
            path_model_cif = os.path.join(outpath_pdb, pdb_id, f"{pdb_id}_model.cif")
            path_model_pdb = path_model_cif.replace(".cif", ".pdb")
            path_final = os.path.join(outpath_models, os.path.basename(path_model_pdb))

            os.makedirs(outpath_pdb, exist_ok=True)
            if not os.path.exists(path_final):
                outpath_json = os.path.join(outpath_pdb, f"{pdb_id}_AF3_input.json")
                try:
                    path_json = get_af3_jsons_from_pdbs(path_pdb, outpath_json)
                except ValueError as e:
                    if "Ran out of single-letter uppercase IDs" in str(e):
                        logging.error(f"Too many elements in {pdb_id}, skipping file.")
                        with open(too_many_file, "a") as f:
                            f.write(f"{pdb_id}\n")
                        continue
                    else:
                        raise
                logging.info(f"Generated {outpath_json}.")

                # Run AF3
                logging.info(f"Predicting {path_json} with AlphaFold3.")
                run_af3(path_json, outpath_pdb)
                if os.path.exists(path_model_cif):
                    cif_to_pdb(path_model_cif, path_model_pdb)
                    # Calculate RMSD between input PDB and predicted PDB
                    rmsd_value = compute_rmsd(path_pdb, path_model_pdb)
                    logging.info(f"RMSD for {pdb_id}: {rmsd_value:.2f} Å")
                    if rmsd_value > 20.0:
                        logging.warning(f"High RMSD {rmsd_value:.2f} for {pdb_id}, skipping move.")
                        with open(high_rmsd_file, "a") as f:
                            f.write(f"{pdb_id}: {rmsd_value:.2f}\n")
                    else:
                        if os.path.exists(path_model_pdb):
                            shutil.move(path_model_pdb, outpath_models)
                            path_final = os.path.join(outpath_models, os.path.basename(path_model_pdb))
                            logging.info(f"Moved prediction for {pdb_id} to final folder.")
            else:
                logging.info(f"Skipping. {path_final} exists already.")
        except Exception as e:
            logging.error(f"Error processing {pdb_id}: {str(e)}")
            with open(error_file, "a") as f:
                f.write(f"{pdb_id}: {str(e)}\n")
            continue

    logging.info("Done.")

###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-prep_AF3_DB-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    DEFAULT_PATH = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/FP"
    parser.add_argument('--inpath', default=f"{DEFAULT_PATH}/input",
                        help="Directory containing PDB files")
    parser.add_argument('--outpath', default=f"{DEFAULT_PATH}/AF3",
                        help="Output directory for AF3 JSON files")
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help="Overwrite existing output directory")
    args = parser.parse_args()
    main(args)
