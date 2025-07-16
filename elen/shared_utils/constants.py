# constants.py

# Standard residues for protein amino acids
STANDARD_RESIDUES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'MSE'
}

AA_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# element mappings

# vanilla ELEN
ELEMENT_MAPPING = {
    'C': 0,
    'O': 1,
    'N': 2,
    'S': 3,
    'SE': 3,
}

# AF3LiMD ELEN
ELEMENT_MAPPING = {
        'BR' : 0,
        'C' : 1, 
        'CA' : 2,
        'CL' : 3,
        'CO' : 4,
        'F' : 5,
        'FE' : 6,
        'HG' : 7,
        'I' : 8,
        'K' : 9,
        'MG' : 10,
        'N' : 11,
        'NA' : 12,
        'O' : 13,
        'P' : 14,
        'S' : 15,
        'SE' : 16,
        'ZN' : 17 
}