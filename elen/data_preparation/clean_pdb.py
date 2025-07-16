#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
# TODO fix cleaning of MD frames
import os
import sys
import glob
import shutil
import logging
import argparse

"""
clean_pdb.py

This script reads PDB files from an input directory and writes cleaned PDB files to an output directory.
It retains only lines that start with ATOM, HETATM, TER, or END.
For ATOM and HETATM lines, water molecules and hydrogen atoms are removed.
It then reassigns chain identifiers so that all standard (ATOM/TER) chains get new IDs (A, B, C, …)
and any remaining hetero (HETATM) chains (that are not protein-like) get letters starting with the letter
following the last protein chain.
Additionally, it renumbers the residues in each protein chain (ATOM and TER lines) starting from 1.
"""

# Set of standard protein residue names.
# (Including MSE for selenomethionine, which is sometimes found in HETATM records but is protein.)
protein_residues = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL", "MSE"
}

def is_hydrogen(line):
    """Return True if the atom name (columns 13-16) indicates a hydrogen."""
    atom_name = line[12:16].strip()
    return atom_name.startswith("H")

def is_water(line):
    """Return True if the residue name (columns 18-20) indicates water."""
    res_name = line[17:20].strip()
    return res_name in ("HOH", "WAT")

def select_alternate_location(input_filename, output_filename):
    """
    Reads a PDB file from input_filename and writes a new file to output_filename.
    
    For records (ATOM/HETATM lines) that have alternate conformations encoded
    by an extra letter at the beginning of the residue name (e.g. "AGLU" or "BGLU"),
    only the A-version is kept. For those lines, the leading 'A' is removed
    so that "AGLU" becomes "GLU". Records with a B-version (i.e. residue name starting
    with "B") are skipped entirely.
    
    All other lines are written unchanged.
    """
    with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
        for line in infile:
            # Only process ATOM/HETATM lines.
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Split the line into tokens (fields separated by whitespace).
                # According to your file, token[3] is the residue identifier.
                tokens = line.split()
                if len(tokens) >= 4:
                    res_token = tokens[3]
                    # Check if the residue name token is prefixed with an alternate conformation indicator.
                    # (Assumes the valid residue name is three letters and the extra character is in front.)
                    if len(res_token) > 3 and res_token[0] in ("A", "B"):
                        if res_token[0] == "B":
                            # Skip the B-version record.
                            continue
                        else:
                            # For the A-version, remove the leading 'A'
                            tokens[3] = res_token[1:]
                            # Rebuild the line (this simple join may alter spacing compared to a strict PDB format)
                            line = " ".join(tokens) + "\n"
            # Write the (possibly modified) line to the output file.
            outfile.write(line)

###############################################################################
def main(args):
    # If the output path exists and overwrite is set, remove it first.
    outpath = args.inpath.rstrip("/") + "_cleaned"
    if os.path.exists(outpath) and args.overwrite:
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)  
    
    for path_pdb in glob.glob(os.path.join(args.inpath, "*.pdb")):
        logging.info(f"Processing {path_pdb}.")
        fname_pdb = os.path.basename(path_pdb) 
        path_pdb_out = os.path.join(outpath, fname_pdb)

        # Read the entire file.
        with open(path_pdb, "r") as fin:
            lines = fin.readlines()

        # Filter lines: keep only those that start with ATOM, HETATM, TER, or END.
        # For ATOM and HETATM lines, remove water molecules and hydrogen atoms.
        filtered_lines = []
        for line in lines:
            if line.startswith("ATOM") or line.startswith("HETATM") or line.startswith("TER") or line.startswith("END"):
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if is_water(line):
                        continue
                    if is_hydrogen(line):
                        continue
                filtered_lines.append(line)

        # Collect original chain IDs from the filtered lines.
        # For protein: lines with record types ATOM and TER,
        # and HETATM lines that have a standard protein residue name.
        protein_chains = []
        ligand_chains = []
        for line in filtered_lines:
            rec = line[0:6].strip()
            if rec in ("ATOM"):
                chain = line[21]
                if chain not in protein_chains:
                    protein_chains.append(chain)
            elif rec == "HETATM":
                resname = line[17:20].strip()
                chain = line[21]
                if resname in protein_residues:
                    if chain not in protein_chains:
                        protein_chains.append(chain)
                else:
                    if chain not in ligand_chains:
                        ligand_chains.append(chain)
            # END lines do not contain chain information.

        # Assign new chain IDs:
        # Protein chains will be renamed to A, B, C, ... in the order they appear.
        protein_map = {}
        for i, orig in enumerate(protein_chains):
            protein_map[orig] = chr(ord('A') + i)
        # Ligand chains get letters starting with the letter following the last protein chain.
        ligand_map = {}
        start_letter = chr(ord('A') + len(protein_chains))
        for i, orig in enumerate(ligand_chains):
            ligand_map[orig] = chr(ord(start_letter) + i)

        # Update chain IDs in all coordinate lines (ATOM, HETATM, TER).
        updated_lines = []
        for line in filtered_lines:
            rec = line[0:6].strip()
            if rec in {"ATOM", "HETATM"}:
                orig_chain = line[21]
                new_chain = None
                if rec == "HETATM":
                    # Determine whether this HETATM record is protein-like or a ligand.
                    resname = line[17:20].strip()
                    if resname in protein_residues:
                        new_chain = protein_map.get(orig_chain, None)
                    else:
                        new_chain = ligand_map.get(orig_chain, None)
                else:
                    new_chain = protein_map.get(orig_chain, None)
                if new_chain is not None:
                    # Replace the chain ID at column 22 (index 21)
                    line = line[:21] + new_chain + line[22:]
            updated_lines.append(line)

        # Renumber protein chain residues.
        # For each protein chain (i.e. new chain IDs from protein_map),
        # reset residue numbering to start at 1. This is applied to lines with record types ATOM and TER.
        renum_dict = {}  # key: new chain ID, value: (last_res_id, current_new_res)
        renumbered_lines = []
        for line in updated_lines:
            rec = line[0:6].strip()
            if rec in {"ATOM"}:
                chain = line[21]
                # Only renumber if this chain is one of the protein chains.
                if chain in protein_map.values():
                    # Extract the current residue identifier (columns 23-27)
                    res_id = line[22:27]  # 5-character field: residue number + insertion code
                    if chain not in renum_dict:
                        renum_dict[chain] = (res_id, 1)
                        new_res = 1
                    else:
                        last_res, current_new = renum_dict[chain]
                        # Increment residue counter when the residue identifier changes.
                        if res_id != last_res:
                            current_new += 1
                            renum_dict[chain] = (res_id, current_new)
                        new_res = renum_dict[chain][1]
                    # Create a new residue number string: right-justified in 4 columns.
                    new_res_str = f"{new_res:>4}"
                    # Replace columns 23-26 with the new residue number and blank out column 27 (insertion code).
                    line = line[:22] + new_res_str + " " + line[27:]
            renumbered_lines.append(line)

        # Write the cleaned, re-chained, and renumbered lines to the output file.
        with open(path_pdb_out, "w") as fout:
            for line in renumbered_lines:
                fout.write(line)

###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-clean_pdbs-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', default="inpath")
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing output')
    args = parser.parse_args()
    main(args)
