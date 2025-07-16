#!/home/florian_wieser/miniconda3/envs/elen_test/bin/python3
import freesasa
from Bio import PDB
import re

def build_freesasa_structure_from_biopython(biopy_struct, classifier=None):
    """
    Create a freesasa.Structure object from a Bio.PDB structure,
    adding every atom (ATOM/HETATM) so that HET groups won't be skipped.
    
    By default, freesasa.Structure's built-in parser may ignore unknown
    residues or fail to parse them. This approach ensures all atoms are included.
    
    :param biopy_struct: Bio.PDB structure (already parsed).
    :param classifier: Optional freesasa.Classifier; if None, use default element-based guess.
    :return: a freesasa.Structure object with all atoms from the Bio.PDB structure.
    """
    # If no classifier provided, we let freesasa guess radii by element
    # This is usually enough for simple ions like SO4, CA, etc.
    if classifier is None:
        classifier = freesasa.ClassifierParam()
        # This dummy classifier will let FreedSASA guess by element name.
        # Alternatively you could define custom lines with .addResidue() etc.
    
    fs_structure = freesasa.Structure(classifier)
    
    for model in biopy_struct:
        for chain in model:
            chain_id = chain.id
            for residue in chain:
                hetflag, resseq, icode = residue.id
                # resName might be something like "SO4", "EDO", etc.
                resname = residue.resname.strip()
                # FreedSASA wants integer residueNumber, so let's keep resseq
                # insertion code can be appended or just store resseq
                for atom in residue:
                    atom_name = atom.name.strip()  # e.g. "O4", "C1"
                    coord = atom.coord
                    # FreedSASA also needs the element if it can't guess from atomName
                    # We'll derive from atom.element or from the PDB line if present
                    # BioPython's atom.element can be '' if not guessed, so let's guess from name:
                    element = atom.element.strip()
                    if not element or element == "?":
                        # fallback: guess from the PDB atom name
                        # e.g. " O " -> "O", " S "->"S"
                        # FreedSASA will do partial guess anyway, but let's be explicit
                        element = re.sub("[^A-Za-z]","",atom_name)  # naive guess

                    fs_structure.addAtom(
                        atom_name,  # FreedSASA atom name
                        resname,    # FreedSASA residue name
                        chain_id,   # FreedSASA chain
                        resseq,     # FreedSASA residue number (int)
                        coord[0], coord[1], coord[2],  # x, y, z
                        element     # FreedSASA tries to interpret the element
                    )
    return fs_structure

def compute_ligand_sasa(pdb_file, chain_id, res_name, res_num):
    """
    1) Parse the PDB with BioPython (including HETATM).
    2) Build a FreeSASA structure with all atoms.
    3) Compute total SASA for the specified ligand (chain, resName, resNum).
    """
    parser = PDB.PDBParser(QUIET=True)
    structure_bio = parser.get_structure("myProt", pdb_file)

    # Build the freesasa structure so HETATM (like SO4) won't be skipped
    fs_struct = build_freesasa_structure_from_biopython(structure_bio)
    
    # Calculate SASA
    result = freesasa.calc(fs_struct)
    sasa_total = 0.0
    
    n_atoms = fs_struct.nAtoms()
    for iatom in range(n_atoms):
        a_chain   = fs_struct.chainLabel(iatom)
        a_resname = fs_struct.residueName(iatom)
        a_resnum  = fs_struct.residueNumber(iatom)

        if (a_chain == chain_id and a_resname == res_name and a_resnum == res_num):
            sasa_total += result.atomArea(iatom)

    return sasa_total

def main():
    # Example usage:
    pdb_path = "6edm_model.pdb"
    chain = "A"
    residue_name = "SO4"
    residue_num = 1  # might be e.g. 501 if that is how your PDB is enumerated

    sasa = compute_ligand_sasa(pdb_path, chain, residue_name, residue_num)
    print(f"Ligand {residue_name} chain={chain} resNum={residue_num} SASA = {sasa:.2f} Å^2")

if __name__ == "__main__":
    main()
