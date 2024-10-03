"""
Code for the Inference of OpenFold
in the MSA free / template mode
using the BioPandas dataframe as an input
"""

import subprocess
import numpy as np
import pandas as pd
import pickle
import torch
from biopandas.pdb import PandasPdb
from scipy.spatial import distance
import json
import PPInterface.openfold.np.protein as protein
from functools import reduce

aa_3_to_1 = {
    "CYS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "ASN": "N",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "ARG": "R",
    "TRP": "W",
    "ALA": "A",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "UNK": "U",
    "-": "-",
}
aa_1_to_3 = {v: k for k, v in aa_3_to_1.items()}


def parse_ranges(s):
    result = []
    # Split the string by commas
    s = s.replace(" ","")
    parts = s.split(',')
    for part in parts:
        # Check if the part contains a range (indicated by '-')
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))  # Add the range of numbers
        else:
            result.append(int(part))  # Add individual number

    return result

def prepare_optimization_input(config):
    """
    Prepare the protein dataframes for the monte carlo simulation
    :param config:
    :param design_id: ID of the cluster of amino acid residues for design
    :return: protein dataframes for the good and bad coomplexes to design
    """
    ###  protein complex
    protein_design = load_protein(config.design_task.protein_design)
    ### chains to design within complex
    design_chain = config.design_task.protein_design_chain
    ### add interface_mask column to the complexes
    add_interface_mask_column(protein_design, design_chain)
    #nodes_to_design = map(int, config.design_task.protein_design_residues.split(","))
    nodes_to_design = parse_ranges(config.design_task.protein_design_residues)

    protein_design["design_mask"] = (protein_design["residue_number_original"].isin(nodes_to_design)) & (
            protein_design["chain_id_original"] == design_chain)

    return protein_design


def save_pdb(pdb_df, output_name, original_numbering=True):
    """
    :param pdb_df: pdb dataframe
    :param output_name: output pdb path
    :return:
    """
    prot = PandasPdb()
    pdb_df = pdb_df.copy()
    if original_numbering:
        pdb_df["residue_number"] = pdb_df["residue_number_original"]
        pdb_df["chain_id"] = pdb_df["chain_id_original"]
    prot.df["ATOM"] = pdb_df
    prot.to_pdb(output_name)

def make_dummy_protein(seq):
    N = len(seq)
    df = {
        "record_name": {"0": "ATOM", "1": "ATOM", "2": "ATOM", "3": "ATOM"},
        "atom_number": {"0": 1, "1": 2, "2": 3, "3": 4},
        "blank_1": {"0": "", "1": "", "2": "", "3": ""},
        "atom_name": {"0": "N", "1": "CA", "2": "C", "3": "O"},
        "alt_loc": {"0": "", "1": "", "2": "", "3": ""},
        "residue_name": {"0": "AAA", "1": "AAA", "2": "AAA", "3": "AAA"},
        "blank_2": {"0": "", "1": "", "2": "", "3": ""},
        "chain_id": {"0": "A", "1": "A", "2": "A", "3": "A"},
        "residue_number": {"0": 1, "1": 1, "2": 1, "3": 1},
        "insertion": {"0": "", "1": "", "2": "", "3": ""},
        "blank_3": {"0": "", "1": "", "2": "", "3": ""},
        "x_coord": {"0": 0.000, "1": 0.000, "2": 0.000, "3": 0.000},
        "y_coord": {"0": 0.000, "1": 0.000, "2": 0.000, "3": 0.000},
        "z_coord": {"0": 0.000, "1": 0.000, "2": 0.000, "3": 0.000},
        "occupancy": {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0},
        "b_factor": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0},
        "blank_4": {"0": "", "1": "", "2": "", "3": ""},
        "segment_id": {"0": "", "1": "", "2": "", "3": ""},
        "element_symbol": {"0": "N", "1": "C", "2": "C", "3": "O"},
        "charge": {"0": np.nan, "1": np.nan, "2": np.nan, "3": np.nan},
        "line_idx": {"0": 508, "1": 509, "2": 510, "3": 511},
        "residue_number_original": {"0": 1, "1": 1, "2": 1, "3": 1},
        "chain_id_original": {"0": "A", "1": "A", "2": "A", "3": "A"},
        "mask": {"0": True, "1": True, "2": True, "3": True},
    }
    protein = []
    n = 0
    for i in range(N):
        resi = pd.DataFrame(df).reset_index(drop=True)#, inplace=True)
        resi["residue_number"] = i+1
        resi["residue_name"] = aa_1_to_3[seq[i]]
        resi.index = [i+len(protein)*4 for i in range(4)]
        resi["atom_number"] = resi.index
        protein.append(resi)
    protein = pd.concat(protein)
    protein["residue_number_original"] = protein["residue_number"]
    protein["line_idx"] = protein["atom_number"]
    return protein



def kalign(
    seq1="MVLTIYPDELVQIVSDKIASNKDKPFWYILAESTLQKEVYFLLAH",
    seq2="MVLTIYPDELVQDKPFWYILAESTLQKEVYFLLAH",
):
    """
    Function to align two protein sequences using kalign
    :param seq1:
    :param seq2:
    :return:
    """
    fasta_sequences = f""">protein1
{seq1}
>protein2
{seq2}
"""
    command = ["kalign"]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = process.communicate(input=fasta_sequences)
    al_seq = []
    for r in stdout.split("\n"):
        if len(r) == 0:
            continue
        if r[0] == ">":
            al_seq.append([])
            continue
        if len(al_seq) == 0:
            continue
        al_seq[-1].append(r.rstrip())

    al_seq[0] = "".join(al_seq[0])
    al_seq[1] = "".join(al_seq[1])
    
    print(al_seq[0])
    print(al_seq[1])

    return al_seq





def update_internal_residue_numbering(pdb_df):
    """
    reset numbering of each chain, so first residue is 1, and next residue
    is is necessary for correct openfold inference
    """
    pdb_df_ca = pdb_df[pdb_df["atom_name"]=="CA"]
    numbering = {}
    n_chain_gap = 0
    for chain, df_ in pdb_df_ca.groupby(["chain_id_original"], sort=False):
        n = 0
        chain_numbering = [r for r,_ in df_.groupby(["residue_number_original","insertion"], sort=False)]
        n_gaps = 0
        for i in range(0, len(chain_numbering)):
            #we want to rename residues
            #1A, 1B, 1C, gap, gap, 4, ... 
            #to
            #1, 2, 3, gap, gap, 6, ...
            if i!=0 and chain_numbering[i-1][0] == chain_numbering[i][0]:
                n_gaps+=1
            else:
                n+=1
            numbering[(chain[0], chain_numbering[i][0], chain_numbering[i][1])] = n+n_gaps+n_chain_gap
            n_last = n+n_gaps+n_chain_gap
        #
        #we want to add +25 gap between chains
        #
        n_chain_gap=25+n_last-1
   
    pdb_df["residue_number"] = pdb_df.apply(lambda row: numbering.get((row['chain_id_original'], row['residue_number_original'], row['insertion'])), axis=1)


def load_protein(path):
    """load protein and change numbering from 1 to N_res
    add 25 residues gap between chains
    input: pdb_path
    output: pdb dataframe prepared for alphafold inference
    """

    pdb_df = PandasPdb().read_pdb(path).df["ATOM"]
    pdb_df = pdb_df[pdb_df["alt_loc"].isin(["A", ""])]
    pdb_df = pdb_df[pdb_df["element_symbol"] != "H"]
    pdb_df = pdb_df[pdb_df["element_symbol"] != "D"]

    pdb_df["residue_number_original"] = pdb_df["residue_number"]
    pdb_df["chain_id_original"] = pdb_df["chain_id"]

    update_internal_residue_numbering(pdb_df)

    pdb_df["chain_id"] = "A"
    pdb_df["mask"] = False

    pdb_df = pdb_df[pdb_df["element_symbol"] != "H"]
    return pdb_df


def mutate_protein(pdb_df, mutant_codes, ignore_not_found=False, wt_control=False):
    """
    function split mutant_codes to (1) single amino acid mutants & deletions (2) insertions
    input:
     pdb_df - pandas dataframe
    mutan_codes list of [{"aa_before":amino_acid_before
      "resi":residue_number,
      "chin_id":chain_id,
       "aa_after", amino_acid_after"}]
    output:
      pdb dataframe for mutant and wt
      for mutant residues only backbone atoms are kept both in mutant and wt dataframes
    """

    mutant_codes_1 = [k for k in mutant_codes if k["aa_before"] != "-"]
    mutant_codes_ins = [k for k in mutant_codes if k["aa_before"] == "-"]
    if wt_control:
        for i in range(len(mutant_codes_1)):
            mutant_codes_1[i]["aa_after"] = mutant_codes_1[i]["aa_before"]

    if len(mutant_codes_1) != 0:
        pdb_df = mutate_protein_(pdb_df, mutant_codes_1)

    if wt_control:
        return pdb_df

    assert len(mutant_codes_ins) == 0, print("Currently inserations are not supported")

    return pdb_df


def add_b_factor(pdb_df, values):
    """
    :param pdb_df: dataframe of protein
    :param values: values corresponding to CA atom that should be added to other atoms
    :return: dataframe of protein with b_factor column replaced by values vector
    """
    pdb_df_ca = pdb_df[pdb_df["atom_name"] == "CA"]
    assert pdb_df_ca.shape[0] == len(values), print("N_CA != len(values)")
    for i, p in enumerate(pdb_df_ca.iloc()):
        resi = p["residue_number"]
        chain = p["chain_id"]
        pdb_df.loc[
            (pdb_df["residue_number"] == resi) & (pdb_df["chain_id"] == chain),
            "b_factor",
        ] = values[i]
    return pdb_df


def mutate_protein_from_list(pdb_df, mutant_codes, ignore_not_found=False):
    """
    input:
     pdb_df - pandas dataframe
      mutan_codes in format {(aa_before, residue_number, chain_id): aa_after), ...}
    output:
      pdb dataframe for mutant and wt
      for mutant residues only backbone atoms are kept both in mutant and wt dataframes
    """

    """ conver single letter code to three letter code if it is the case """

    for aa_before, residue_number, aa_after in mutant_codes:
        pdb_df.loc[pdb_df["residue_number"]=="residue_number", 'C'] = 'updated_value'

        
    mutant_codes_3 = {}
    for k in mutant_codes:
        if len(k["aa_before"]) == 1:
            aa_before = aa_1_to_3[k["aa_before"]]
        if len(k["aa_after"]) == 1:
            aa_after = aa_1_to_3[k["aa_after"]]
        mutant_codes_3[(aa_before, k["resi"], k["chain_id"])] = aa_after

    mutant_codes = mutant_codes_3
    pdb_df_mutant = []
    n_mutants = 0
    pdb_df["residue_name_wt"] = pdb_df["residue_name"]
    n_shift = 0
    for k, df in pdb_df.groupby(
        ["residue_name", "residue_number"], sort=False
    ):
        df_mutant = df.copy()
        df_mutant["residue_number"] += n_shift
        if k in mutant_codes and mutant_codes[k] != "-":
            """amino acid substituion
            keep only backbone atoms from the template and rename residue
            """
            df_ = df[df["atom_name"].isin(["N", "CA", "C", "O"])]
            assert df_.shape[0] == 4
            df_.loc[:, "residue_name"] = mutant_codes[k]  # df_
            pdb_df_mutant.append(df_)  # mutant)
            n_mutants += 1
            continue
        if k in mutant_codes and mutant_codes[k] == "-":
            """deletion
            ignore the residue and update the numbering
            """
            n_mutants += 1
            n_shift -= 1
            continue
        pdb_df_mutant.append(df_mutant)
    if not ignore_not_found:
        assert n_mutants == len(mutant_codes)
    pdb_df_mutant = pd.concat(pdb_df_mutant)
    return pdb_df_mutant


def mutate_protein_(pdb_df, mutant_codes, ignore_not_found=False):
    """
    input:
     pdb_df - pandas dataframe
      mutan_codes in format {(aa_before, residue_number, chain_id): aa_after), ...}
    output:
      pdb dataframe for mutant and wt
      for mutant residues only backbone atoms are kept both in mutant and wt dataframes
    """

    """ conver single letter code to three letter code if it is the case """
    mutant_codes_3 = {}
    for k in mutant_codes:
        if len(k["aa_before"]) == 1:
            aa_before = aa_1_to_3[k["aa_before"]]
        if len(k["aa_after"]) == 1:
            aa_after = aa_1_to_3[k["aa_after"]]
        mutant_codes_3[(aa_before, k["resi"], k["chain_id"])] = aa_after

    mutant_codes = mutant_codes_3
    pdb_df_mutant = []
    n_mutants = 0
    pdb_df["residue_name_wt"] = pdb_df["residue_name"]
    n_shift = 0
    for k, df in pdb_df.groupby(
        ["residue_name", "residue_number_original", "chain_id_original"], sort=False
    ):
        df_mutant = df.copy()
        df_mutant["residue_number"] += n_shift
        if k in mutant_codes and mutant_codes[k] != "-":
            """amino acid substituion
            keep only backbone atoms from the template and rename residue
            """
            df_ = df[df["atom_name"].isin(["N", "CA", "C", "O"])]
            assert df_.shape[0] == 4
            # df_mutant["residue_name"] = mutant_codes[k]
            df_.loc[:, "residue_name"] = mutant_codes[k]  # df_
            pdb_df_mutant.append(df_)  # mutant)
            n_mutants += 1
            continue
        if k in mutant_codes and mutant_codes[k] == "-":
            """deletion
            ignore the residue and update the numbering
            """
            n_mutants += 1
            n_shift -= 1
            continue
        pdb_df_mutant.append(df_mutant)
    if not ignore_not_found:
        assert n_mutants == len(mutant_codes)
    pdb_df_mutant = pd.concat(pdb_df_mutant)
    return pdb_df_mutant


def pdbline_to_dict(line):
    """
    convert PDB string to DICT in BioPandas DataFrame format
    args:
         pdblineToDataframe(string, tuple, list)

         line -- string from PDB file (starts with ATOM ...) -> string
         resi_key

        returns
        change protein name
    """
    atom_name = line[13:15]
    if line[15] != " ":
        atom_name += line[15]
    residue_name = line[17:20]
    chain_name = line[21]
    residue_number = int(line[22:26])
    x, y, z = line[30:38], line[38:46], line[46:54]
    if x[0] != " ":
        x = x[1:]
    if y[0] != " ":
        y = y[1:]
    if z[0] != " ":
        z = z[1:]
    (
        x,
        y,
        z,
    ) = (
        float(x),
        float(y),
        float(z),
    )
    if len(line) >= 77:
        atom_type = line[77]
    else:
        atom_type = atom_name[0]
    b = float(line[61:66])
    atom_number = int(line[4:11])
    df_ = {
        c: ""
        for c in [
            "record_name",
            "atom_number",
            "blank_1",
            "atom_name",
            "alt_loc",
            "residue_name",
            "blank_2",
            "chain_id",
            "residue_number",
            "insertion",
            "blank_3",
            "x_coord",
            "y_coord",
            "z_coord",
            "occupancy",
            "b_factor",
            "blank_4",
            "segment_id",
            "element_symbol",
            "charge",
            "line_idx",
        ]
    }
    df_["atom_number"] = atom_number
    df_["record_name"] = "ATOM"
    df_["atom_name"] = atom_name
    df_["residue_name"] = residue_name
    df_["aa"] = aa_3_to_1[residue_name]
    df_["chain_id"] = chain_name
    df_["residue_number"] = residue_number
    df_["x_coord"] = x
    df_["y_coord"] = y
    df_["charge"] = 0
    df_["z_coord"] = z
    df_["occupancy"] = 1.0  # 1.0
    df_["b_factor"] = float(b)  # 105.55
    df_["element_symbol"] = atom_type
    return df_


def get_sequence(pdb_df):
    """
    return protein sequence
    :param pdb_df:
    :return:
    """
    return "".join(
        [
            aa_3_to_1[r["residue_name"]]
            for r in pdb_df[pdb_df["atom_name"] == "CA"].iloc()
        ]
    )


def pdb_str_to_dataframe(pdb_lines, pdb_df_prev=None):
    """
    convert PDB to biopandas dataframe
    :param pdb_lines: alphafold predictions
    :param pdb_df_old: dataframe that contains extra columns, e.g. original_numbering
    :return:
    """
    if isinstance(pdb_lines, str):
        pdb_lines = pdb_lines.split("\n")
    pdb_df = []
    for line in pdb_lines:
        if not line.startswith("ATOM"):
            continue
        pdb_df.append(pdbline_to_dict(line))
    columns = [
        "record_name",
        "atom_number",
        "blank_1",
        "atom_name",
        "alt_loc",
        "residue_name",
        "blank_2",
        "chain_id",
        "residue_number",
        "insertion",
        "blank_3",
        "x_coord",
        "y_coord",
        "z_coord",
        "occupancy",
        "b_factor",
        "blank_4",
        "segment_id",
        "element_symbol",
        "charge",
        "line_idx",
    ]
    pdb_df = pd.DataFrame(pdb_df)
    pdb_df = pdb_df.reindex(columns=columns)
    pdb_df["line_idx"] = pdb_df.index
    if pdb_df_prev is None:
        return pdb_df

    original_numbering = {
        k: (
            r.iloc()[0]["residue_number_original"],
            r.iloc()[0]["chain_id_original"],
            r.iloc()[0]["insertion"],
        )
        for k, r in pdb_df_prev.groupby(["residue_number", "chain_id"], sort=False)
    }

    #print(original_numbering)
    pdb_df["residue_number_original"] = [
        original_numbering[(r["residue_number"], r["chain_id"])][0]
        for r in pdb_df.iloc()
    ]
    pdb_df["chain_id_original"] = [
        original_numbering[(r["residue_number"], r["chain_id"])][1]
        for r in pdb_df.iloc()
    ]
    pdb_df["insertion"] = [
        original_numbering[(r["residue_number"], r["chain_id"])][2]
        for r in pdb_df.iloc()
    ]

    return pdb_df



def add_interface_mask_column(pdb_df, chain, R=4.5):  # chains_1, chains_2, R=4.5):
    """
    Function to add interface_mask column
    true if residue is in the chain A at within 4.5 distance from any atom of chain B

    :param pdb_df:
    :param chain:
    :param R:
    :return:
    """
    pdb_df["interface_mask"] = False
    pdb_df_A = pdb_df[pdb_df["chain_id_original"] == chain].reset_index(drop=True)
    pdb_df_B = pdb_df[pdb_df["chain_id_original"] != chain].reset_index(drop=True)
    xyz_1 = pdb_df_A[["x_coord", "y_coord", "z_coord"]].to_numpy()
    xyz_2 = pdb_df_B[["x_coord", "y_coord", "z_coord"]].to_numpy()
    cd = distance.cdist(xyz_1, xyz_2)
    ids = list(set(np.where(cd < R)[0]))
    interface_A = pdb_df_A.iloc()[ids]
    keys = {}
    for c in interface_A.iloc():
        keys[(c["chain_id_original"], c["residue_number"])] = True
    pdb_df["interface_mask"] = pdb_df.apply(lambda row: (row["chain_id_original"], row["residue_number"]) in keys,
                                            axis=1)

