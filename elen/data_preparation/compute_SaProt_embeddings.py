#!/home/florian_wieser/anaconda3/envs/SaProt/bin/python3

import os
import sys
import h5py
import glob
import torch
import shutil
import argparse as ap
import json

sys.path.append("/home/florian_wieser/software/SaProt/")
from utils.foldseek_util import get_struc_seq
from model.esm.base import EsmBaseModel
from transformers import EsmTokenizer
import transformers
transformers.logging.set_verbosity_error()

from Bio.PDB import PDBParser

def discard_pdb(path_pdb, path_discarded, step, error, log_file="discarded_pdb.log"):
    os.makedirs(path_discarded, exist_ok=True)
    shutil.move(path_pdb, path_discarded)
    with open(log_file, "a") as log:
        log.write(f"{path_pdb}, Step: {step}, Error: {error}\n")

def get_residue_ids(path_pdb: str) -> list:
    parser = PDBParser(QUIET=True)
    resnum_list = []
    structure = parser.get_structure("structure", path_pdb)
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':
                    chain_and_id = f"{chain.id}_{residue.id[1]}"
                    resnum_list.append(chain_and_id)
    return resnum_list

def get_saprot_sequence_embeddings(path_pdb):
    parsed_seqs = get_struc_seq("/home/florian_wieser/software/SaProt/bin/foldseek", path_pdb)
    if parsed_seqs:
        combined_seq = next(iter(parsed_seqs.values()))[2]
    else:
        print("No sequences found in parsed_seqs.")
        return None
    config = {
        "task": "base",
        "config_path": "/home/florian_wieser/software/SaProt/weights/PLMs/SaProt_650M_PDB",
        "load_pretrained": True
    }
    model = EsmBaseModel(**config)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer(combined_seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model.get_hidden_states(inputs)
    return embeddings[0]

def calculate_sequence_embeddings(
    inpath, saprot_model, outpath, write_json=False, path_discarded="discarded"
):
    path_out = os.path.join(outpath, f"{saprot_model}.h5")
    path_out_json = os.path.join(outpath, f"{saprot_model}.json")
    json_embeddings = {}

    with h5py.File(path_out, 'w') as f:
        for path_pdb in glob.glob(os.path.join(inpath, "*.pdb")):
            fname_pdb = os.path.basename(path_pdb)
            print(f"Computing SaProt embedding of {fname_pdb}.")
            try:
                sequence_embedding = get_saprot_sequence_embeddings(path_pdb)
                res_ids = get_residue_ids(path_pdb)
                if sequence_embedding is not None:
                    dset = f.create_dataset(
                        fname_pdb,
                        data=sequence_embedding.cpu(),
                        compression="gzip"
                    )
                    dset.attrs["res_ids"] = [rid.encode("utf-8") for rid in res_ids]
                    if write_json:
                        json_embeddings[fname_pdb] = {
                            "res_ids": res_ids,
                            "embedding": sequence_embedding.cpu().tolist()
                        }
            except Exception as e:
                print(f"Error processing {fname_pdb}: {e}")
                discard_pdb(path_pdb, path_discarded, "SaProt embedding calculation", e)
                continue

    if write_json:
        with open(path_out_json, "w") as f_json:
            json.dump(json_embeddings, f_json)
            
def run_compute_saprot_embeddings(
    inpath_models, saprot_model, outpath,
    overwrite=False, write_json=False, path_discarded="discarded"
):
    """
    Main entry point for calling from other scripts.
    Computes SaProt embeddings for all PDBs in inpath_models and writes outputs to outpath.
    """
    if overwrite and os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)
    calculate_sequence_embeddings(
        inpath_models, saprot_model, outpath, write_json=write_json, path_discarded=path_discarded
    )

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('--inpath_models', type=str, default="AF3_models", help="Input model .pdb files")
    parser.add_argument('--outpath', type=str, default="SaProt_embeddings", help="Output directory")
    parser.add_argument("--saprot_model", type=str, default="saprot_650M", help="Type of SaProt model")
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing output directory')
    parser.add_argument("--path_discarded", type=str, default=f"discarded", help="Output directory for failed PDB files.")
    parser.add_argument('--write_json', action='store_true', default=False,
                        help='Also write embeddings as a JSON file')
    args = parser.parse_args()
    run_compute_saprot_embeddings(
        args.inpath_models,
        args.saprot_model,
        args.outpath,
        overwrite=args.overwrite,
        write_json=args.write_json,
        path_discarded=args.path_discarded
    )
