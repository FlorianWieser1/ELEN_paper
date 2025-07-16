#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
#SBATCH -J harmonize_pdbs 
#SBATCH -o harmonize_pdbs.log
#SBATCH -e harmonize_pdbs.err

import os
import sys
import glob
import shutil
import logging
import argparse as ap
from Bio import PDB
from elen.data_preparation.gromacs import convert_mse_to_met
from elen.shared_utils.utils_others import func_timer, discard_pdb

### HELPERS ###################################################################

def clean_and_renumber_dir(inpath, outpath):
    logging.info(f"Cleaning directory {inpath}.")
    for path_pdb in glob.glob(os.path.join(inpath, "*.pdb")):
        outpath_pdb = os.path.join(outpath, os.path.basename(path_pdb))
        renumber_and_clean_pdb_chains(path_pdb, outpath_pdb)

def renumber_and_clean_pdb_chains(path_pdb, outpath_pdb):
    """
    Processes a PDB file by removing water molecules, renumbering residues,
    renaming chains consecutively starting from 'A', and writing the cleaned structure
    to a new PDB file.
    """
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('PDB', path_pdb)
    except Exception as e:
        logging.error(f"Failed to parse {path_pdb}: {e}")
        return

    chain_letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]

    # Create a new structure and model for cleaned data
    new_structure = PDB.Structure.Structure(structure.id)
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)

    chain_index = 0

    # Assuming the input has a single model (structure[0])
    for old_chain in structure[0]:
        if chain_index >= len(chain_letters):
            logging.warning(f"Exceeded chain letters for {path_pdb}. Additional chains will not be renamed.")
            break  # Avoid exceeding available chain letters

        # Create a new chain with a new sequential ID
        new_chain_id = chain_letters[chain_index]
        new_chain = PDB.Chain.Chain(new_chain_id)
        chain_index += 1

        # Filter out water residues and renumber remaining residues
        non_water_residues = [res for res in old_chain if res.get_resname() != 'HOH']
        for i, old_residue in enumerate(non_water_residues, start=1):
            # Preserve the original hetero flag instead of forcing it to ' '
            new_residue = PDB.Residue.Residue((old_residue.id[0], i, ' '), 
                                              old_residue.get_resname(), 
                                              old_residue.segid)
            # Copy atoms from the old residue to the new residue
            for atom in old_residue:
                new_atom = atom.copy()
                new_residue.add(new_atom)

            new_chain.add(new_residue)

        new_model.add(new_chain)

    # Write out the entire modified structure to a single output file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    try:
        io.save(outpath_pdb)
        logging.debug(f"Saved cleaned structure to {outpath_pdb}")
    except Exception as e:
        logging.error(f"Failed to save {outpath_pdb}: {e}")

###############################################################################       

def remove_alternative_atom_locations_from_dir(inpath):
    logging.info(f"Removing alternate atom location from directory {inpath}.")
    for path_pdb in glob.glob(os.path.join(inpath, "*_wo_MSE.pdb")):
        if "AF3_models" in inpath:
            tag = "_af3.pdb"
            outpath_pdb = path_pdb.replace("_model_rep_wo_MSE.pdb", tag)
        else:
            tag = "_wo_alt.pdb"
            outpath_pdb = path_pdb.replace("_rep_wo_MSE.pdb", tag)
        select_alternate_location(path_pdb, outpath_pdb)
      
def select_alternate_location(input_filename, output_filename):
    """
    Reads a PDB file from input_filename and writes a new PDB file to output_filename.
    
    For ATOM/HETATM lines, it will keep only the first alternate location for each atom.
    The alternate location indicator (column 17) is removed (set to a blank) for the kept record.
    Subsequent records for the same atom (identified by chain, residue sequence, insertion code, and atom name)
    are skipped. All other lines (e.g. HEADER, TER, END) are written unchanged.
    
    Note: This function uses fixed‑width slicing according to the standard PDB format.
    """
    processed_atoms = set()
    with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
        for line in infile:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                if len(line) >= 27:
                    alt_loc = line[16]
                    atom_name = line[12:16].strip()
                    chain = line[21]
                    resseq = line[22:26]
                    icode = line[26]
                    atom_key = (chain, resseq, icode, atom_name)
                    if atom_key in processed_atoms:
                        continue
                    processed_atoms.add(atom_key)
                    if alt_loc != " ":
                        line = line[:16] + " " + line[17:]
            outfile.write(line)
       
### HELPERS HARMONIZATION #####################################################
def parse_pdb_residues(filename):
    """
    Reads the file and groups consecutive ATOM/HETATM lines into “residues.”
    Waters (resname HOH) and hydrogen atoms (element "H" in columns 77-78) are skipped.
    Returns a list of dictionaries, each with:
       "resname": the residue name (columns 18-20)
       "lines": a list of the original ATOM/HETATM lines for that residue.
    """
    residues = []
    current_res_lines = []
    current_res_key = None  # key is (chain, resseq, icode) as read in the file
    with open(filename, 'r') as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            resname = line[17:20].strip()
            if resname == "HOH":
                continue

            element = line[76:78].strip()
            if element.upper() == "H":
                continue

            chain = line[21]
            resseq = line[22:26]
            icode = line[26]
            resid_key = (chain, resseq, icode)
            if current_res_key is None or resid_key != current_res_key:
                current_res_lines = []
                residues.append({"resname": resname, "lines": current_res_lines})
                current_res_key = resid_key
            current_res_lines.append(line)
    return residues

def get_reference_mapping(ref_filename):
    """
    Parses the reference file and returns a list of mapping entries (one per residue)
    where each entry is a dict with keys:
      "chain": chain ID (column 22)
      "resseq": residue number (columns 23-26, as string)
      "icode": insertion code (column 27)
      "resname": residue name (columns 18-20)
    """
    mapping = []
    residues = parse_pdb_residues(ref_filename)
    for res in residues:
        first_line = res["lines"][0]
        new_chain = first_line[21]
        new_resseq = first_line[22:26]
        new_icode = first_line[26]
        mapping.append({
            "chain": new_chain,
            "resseq": new_resseq,
            "icode": new_icode,
            "resname": first_line[17:20].strip()
        })
    return mapping

def build_mapping_by_resname(ref_mapping):
    """
    From the flat list of mapping entries from the reference file,
    build a dictionary keyed by residue name.
    Each key holds a list (in order) of mapping entries for that residue type.
    """
    mapping_by_resname = {}
    for entry in ref_mapping:
        rname = entry["resname"]
        mapping_by_resname.setdefault(rname, []).append(entry)
    return mapping_by_resname

def apply_mapping_to_target(target_filename, mapping_by_resname):
    """
    Reads the target file and (for each residue) looks up its residue name in the
    mapping_by_resname dictionary to get the next available mapping entry.
    (If a residue type in the target is not available in the reference mapping, an error is raised.)
    Then reassigns for each atom line:
      - a new atom serial number (sequentially)
      - the chain ID, residue number and insertion code from the mapping entry.
    TER records are inserted when the (assigned) chain ID changes.
    Returns a list of new lines.
    """
    target_residues = parse_pdb_residues(target_filename)
    assigned_mapping = []
    for res in target_residues:
        rname = res["resname"].strip()
        if rname == "MET" and rname not in mapping_by_resname:
            if "MSE" in mapping_by_resname:
                rname = "MSE"
        if rname not in mapping_by_resname or len(mapping_by_resname[rname]) == 0:
            raise Exception(f"Error: No mapping available for residue {rname} in file {target_filename}")
        assigned_mapping.append(mapping_by_resname[rname].pop(0))
    
    new_lines = []
    atom_serial = 1
    for i, res in enumerate(target_residues):
        map_entry = assigned_mapping[i]
        new_chain = map_entry["chain"]
        try:
            new_resseq_int = int(map_entry["resseq"])
            new_resseq_formatted = f"{new_resseq_int:4d}"
        except ValueError:
            new_resseq_formatted = map_entry["resseq"].rjust(4)
        new_icode = map_entry["icode"]
        for line in res["lines"]:
            record_type = line[:6]
            new_atom_serial_str = f"{atom_serial:5d}"
            new_line = (record_type +
                        new_atom_serial_str +
                        line[11:21] +
                        new_chain +
                        new_resseq_formatted +
                        new_icode +
                        line[27:])
            new_lines.append(new_line)
            atom_serial += 1

        if i == len(target_residues)-1 or (assigned_mapping[i]["chain"] != assigned_mapping[i+1]["chain"]):
            ter_line = (f"TER   {atom_serial:5d}      {res['resname']:>3} {new_chain}"
                        f"{new_resseq_formatted}{new_icode}" +
                        " " * 30 + "\n")
            new_lines.append(ter_line)
            atom_serial += 1
    return new_lines

def harmonize_pdb_to_ref(path_pdb, path_ref, path_out):
    ref_mapping = get_reference_mapping(path_pdb)
    mapping_by_resname = build_mapping_by_resname(ref_mapping)
    new_lines = apply_mapping_to_target(path_ref, mapping_by_resname)
     
    with open(path_out, "w") as fout:
        for line in new_lines:
            fout.write(line)
        fout.write("END\n")
    return path_out

def remove_tmp_files(inpath):
    for path_pdb in glob.glob(os.path.join(inpath, f"*.pdb")):
        if not "_gro.pdb" in path_pdb and not "_af3.pdb" in path_pdb and not "_nat.pdb" in path_pdb:
            os.remove(path_pdb)

def replace_ATOM_from_dir(inpath):
    logging.info(f"Replacing ATOM with HETATM from directory {inpath}.")
    for path_pdb in glob.glob(os.path.join(inpath, "*.pdb")):
        outpath_pdb = path_pdb.replace(".pdb", "_rep.pdb")
        replace_ATOM_w_HETATM_atoms(path_pdb, outpath_pdb)
        
def replace_ATOM_w_HETATM_atoms(path_pdb, outpath_pdb):
    """
    Reads a PDB file and modifies its atom records as follows:
      - For lines starting with "ATOM": if the residue is not one of the 20 standard amino acids,
        the record is changed from ATOM to HETATM.
      - For lines starting with "HETATM": if the residue name is "MET", the record is changed
        from HETATM to ATOM.
    The modified records are written to a new PDB file.
    """
    standard_residues = {
        "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
        "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
        "THR", "VAL", "TRP", "TYR", "MSE"
    }
    
    with open(path_pdb, "r") as infile, open(outpath_pdb, "w") as outfile:
        for line in infile:
            if line.startswith("ATOM"):
                resname = line[17:20].strip()
                if resname not in standard_residues:
                    line = "HETATM" + line[6:]
            elif line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname == "MET" or resname == "MSE":
                    line = "ATOM  " + line[6:]
            outfile.write(line)
            
    return outpath_pdb

def convert_MSE_to_MET_in_dir(inpath):
    logging.info(f"Converting MSE residues to MET in directory {inpath}.")
    for path_pdb in glob.glob(os.path.join(inpath, "*.pdb")):
        outpath_pdb = path_pdb.replace("_rep.pdb", "_rep_wo_MSE.pdb")
        convert_mse_to_met(path_pdb, outpath_pdb)

# New function to check for modified cysteine residues and discard files if found
def check_and_discard_cysteine_modifications(inpath, path_discarded):
    """
    Checks all .pdb files in the given directory (inpath) for HETATM records with residue names
    CSO or CME (or other modified cysteine states if present). If any file contains such records,
    a message is logged and the file is moved to the discarded folder.
    """
    modified_cys = {"CSO", "CME"}
    key = os.path.basename(os.path.normpath(inpath))
    for pdb_file in glob.glob(os.path.join(inpath, "*.pdb")):
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("HETATM"):
                    resname = line[17:20].strip()
                    if resname in modified_cys:
                        logging.warning(f"File {pdb_file} contains modified cysteine residue {resname}. Moving to discarded.")
                        discard_pdb(pdb_file, path_discarded, "harmonization", "cysteine CSO CME")
                        break


def harmonize_dir(inpath_af, inpath, path_discarded):
    logging.info(f"Harmonizing .pdb files in directory {inpath} to AF3 models regarding chain numbering.")
    for path_af in glob.glob(os.path.join(inpath_af, "*_af3.pdb")):
        identifier = os.path.basename(path_af)[:4]
        try:
            logging.debug(f"Harmonizing {path_af}.")
            target_files = glob.glob(os.path.join(inpath, f"{identifier}*_wo_alt.pdb"))
            if not target_files:
                raise Exception(f"No matching pdb file found in {inpath} for identifier {identifier}.")
            path_target = target_files[0]
            if "native" in inpath:
                tag = "nat"
            elif "MD" in inpath:
                tag = "gro"
            else:
                tag = "harmonized"
            path_harmonized = os.path.join(inpath, f"{identifier}_{tag}.pdb")
            harmonize_pdb_to_ref(path_af, path_target, path_harmonized)
        except Exception as e:
            logging.error(f"Error harmonizing {identifier} for directory {inpath}: {e}")
            discard_pdb(path_af, path_discarded, "harmonization", "harmonization")


###############################################################################
@func_timer
def main(args):
    if os.path.exists(args.outpath) and args.overwrite:
        shutil.rmtree(args.outpath)
    os.makedirs(args.outpath, exist_ok=True)
    outpath_af = os.path.join(args.outpath, os.path.basename(os.path.normpath(args.inpath_AF)))
    outpath_natives = os.path.join(args.outpath, os.path.basename(os.path.normpath(args.inpath_natives)))
    outpath_md = os.path.join(args.outpath, os.path.basename(os.path.normpath(args.inpath_MD)))
    for directory in [outpath_af, outpath_natives, outpath_md]:
        os.makedirs(directory, exist_ok=True)
    
    # Clean .pdbs from waters and renumber residues to start with 1          
    for inpath, outpath in zip([args.inpath_AF, args.inpath_natives, args.inpath_MD], 
                               [outpath_af, outpath_natives, outpath_md]):
        clean_and_renumber_dir(inpath, outpath)
        
    # Replace ATOM records with HETATM if ligand (issue of MD .pdbs)        
    for inpath in [outpath_af, outpath_natives, outpath_md]:
        replace_ATOM_from_dir(inpath)
      
    # Convert MSE to MET
    for inpath in [outpath_af, outpath_natives, outpath_md]:
        convert_MSE_to_MET_in_dir(inpath)
        
    # Check for modified cysteine residues (CSO, CME) and discard files if found
    for inpath in [outpath_af, outpath_natives, outpath_md]:
        check_and_discard_cysteine_modifications(inpath, args.path_discarded)
        
    # Remove alternate atom locations    
    for inpath in [outpath_af, outpath_natives, outpath_md]:
        remove_alternative_atom_locations_from_dir(inpath)
        
    # Harmonize .pdbs regarding chain and ligand numbering 
    for inpath in [outpath_natives, outpath_md]:
        harmonize_dir(outpath_af, inpath, args.path_discarded)
        
    # Remove temporary files
    for inpath in [outpath_af, outpath_natives, outpath_md]:
        remove_tmp_files(inpath)

    logging.info("Done.")

            
###############################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='ELEN-harmonize_pdbs-%(levelname)s(%(asctime)s): %(message)s',
        datefmt='%y-%m-%d %H:%M:%S'
    )

    parser = ap.ArgumentParser(description="Process and harmonize PDB files by renumbering residues, removing water molecules, renaming chains, and ensuring atom consistency across datasets.")
    default_path = "/home/florian_wieser/projects/ELEN/elen_training/data_preparation/AF3_LiMD"
    parser.add_argument("--inpath_natives", type=str, default=f"{default_path}/natives", help="Input directory for native PDB files.")
    parser.add_argument("--inpath_AF", type=str, default=f"{default_path}/AF3_models", help="Input directory for AF3 model PDB files.")
    parser.add_argument("--inpath_MD", type=str, default=f"{default_path}/MD_frames", help="Input directory for MD frame PDB files.")
    parser.add_argument("--outpath", type=str, default=f"{default_path}/harmonized", help="Output directory for harmonized PDB files.")
    parser.add_argument("--path_discarded", type=str, default=f"discarded", help="Output directory for failed PDB files.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing output folders.")
    args = parser.parse_args()
    main(args)
