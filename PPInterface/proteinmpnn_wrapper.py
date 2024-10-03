import numpy as np
import torch
import proteinmpnn
from scipy import stats

from proteinmpnn.run import load_protein_mpnn_model, set_seed, nll_score
from proteinmpnn.data import BackboneSample, untokenise_sequence

#from proteinmpnn.run import load_protein_mpnn_model, set_seed, nll_score
#from proteinmpnn_wrapper.src.proteinmpnn.ProteinMPNN.protein_mpnn_utils import tied_featurize
#from PPInterface.openfold_wrapper import OpenFoldWraper

from PPInterface.protein_utils import (
    load_protein,
    save_pdb,
    make_dummy_protein,
    get_sequence,
    mutate_protein,
    aa_3_to_1,
    aa_1_to_3,
    kalign,
)

#from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB
#from ace2_test import add_interface_mask_column

from PPInterface.protein_utils import add_interface_mask_column
from scipy.spatial import distance
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import os
import yaml
from omegaconf import OmegaConf
import logging
import pandas as pd
import logging
import sys
import os

### uncomment this
###
#from proteinmpnn_ddg import predict_logits_for_all_point_mutations_of_single_pdb
###
###

import json

# increase solubility
# proteinmpnn-ddg
# https://www.biorxiv.org/content/10.1101/2024.06.15.599145v3.full.pdf

def N_to_AA(x):
    """
    Function to convert proteinmpnn residues to amino acid sequence
    :param x:
    :return:
    """
    vocab = {
        3: "E",
        17: "V",
        13: "Q",
        9: "L",
        15: "S",
        5: "G",
        12: "P",
        14: "R",
        1: "C",
        0: "A",
        4: "F",
        16: "T",
        11: "N",
        19: "Y",
        10: "M",
        18: "W",
        8: "K",
        7: "I",
        2: "D",
        6: "H",
        20: "-",
    }
    x = np.array(x)
    if x.ndim == 1:
        return "".join([vocab[int(a)] for a in x])
    return ["".join([vocab[int(a)] for a in x_]) for x_ in x]

    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    states = len(alpha_1)
    alpha_3 = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "GAP",
    ]
    aa_1_N = {a: n for n, a in enumerate(alpha_1)}
    aa_3_N = {a: n for n, a in enumerate(alpha_3)}
    aa_N_1 = {n: a for n, a in enumerate(alpha_1)}
    aa_1_3 = {a: b for a, b in zip(alpha_1, alpha_3)}
    aa_3_1 = {b: a for a, b in zip(alpha_1, alpha_3)}
    # [[0,1,2,3]] -> ["ARND"]
    x = np.array(x)
    if x.ndim == 1:
        x = x[None]
    return ["".join([aa_N_1.get(a, "-") for a in y]) for y in x]

def AA_to_N(x):
    """
    Function to convert  amino acid sequence to proteinmpnn vocab
    :param x:
    :return:
    """
    vocab = {
        3: "E",
        17: "V",
        13: "Q",
        9: "L",
        15: "S",
        5: "G",
        12: "P",
        14: "R",
        1: "C",
        0: "A",
        4: "F",
        16: "T",
        11: "N",
        19: "Y",
        10: "M",
        18: "W",
        8: "K",
        7: "I",
        2: "D",
        6: "H",
    }
    vocab_r = {v: h for h, v in vocab.items()}
    return np.array([[vocab_r[a] for a in y] for y in x])


def plot_design_nodes_in_pymol(protein, nodes, config):
    """
    Function to plot residues selected for design in PyMOL
    :param protein:
    :param nodes:
    :param config:
    :return:
    """
    import pymol
    from pymol import cmd

    pymol_log_dir = config.paths.pymol_log_dir
    temp_path = config.paths.temp_dir

    Path(temp_path).mkdir(exist_ok=True)
    save_pdb(protein, temp_path + "/temp.pdb")
    cmd.do("delete *")
    cmd.set_key("F1", cmd.zoom, ["all within 5.0 of (sele)"], {"animate": 1})
    cmd.set_key("F2", cmd.zoom, [], {"animate": 1})
    cmd.do(f"load {temp_path}/temp.pdb, pr")
    cmd.do("remove heta")
    cmd.do("color gray, elem C")
    cmd.do("set ray_shadows,off")
    cmd.do(f"set cartoon_color=gray, pr")
    cmd.do(f"set sphere_scale, 0.25")
    cmd.do("set dash_gap, 0")

    chain_id = protein[protein["interface_mask"]].iloc()[0]["chain_id_original"]
    for i, n in enumerate(nodes):
        for resi in n:
            cmd.do(f"show spheres, resi {resi} & chain {chain_id} & name CA")
            cmd.do(
                f"color {i+1}, resi {resi} & chain {chain_id} & elem C"
            )  # & name CA")
            cmd.do(f"show sticks, resi {resi} & chain {chain_id}")
    cmd.do(f"show lines")
    cmd.set_view(
        """\
    -0.657753825,   -0.621259689,    0.425904661,\
    -0.285076171,   -0.318052769,   -0.904198349,\
     0.697201967,   -0.716153026,    0.032093499,\
     0.000020817,   -0.000064760, -260.029510498,\
     2.852099180,   39.968116760,   -1.194970131,\
   205.008544922,  315.048706055,  -20.000000000 """
    )

    cmd.do("save " + pymol_log_dir + "/search_clusters_final.pse")

    # return nodes_pdb_numbering

def split_graph(
    protein,
    G,
    temp_path="./temp/",
    weight_threshold=0.1,
    show_pymol=True,
    show_matplotlib=False,
    pymol_log_dir=None,
):
    """
    Having a residue interaction graph reconstructed using proteinmpnn
    we want to split on small subgraphs with a center on the node,
    and at most two residues distant from this node according to connectivity.
    This is done in order to reduce the search space using MCMC algorithm
    and at the same time to take into account epistaic effects when substituting one of the amino acids

    :param protein:
    :param G:
    :param temp_path:
    :param weight_threshold:
    :param show_pymol:
    :param show_matplotlib:
    :param pymol_log_dir:
    :return:
    """
    ### PDB dataframe containing only CA atoms
    protein_ca = protein[protein["atom_name"] == "CA"]
    pos = nx.spring_layout(G)
    nx.draw(
        G, pos, with_labels=True, node_color="lightblue", node_size=500, font_size=15
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    ### if weight < 0.1 we remove such edges
    ### weight is selected empirically
    ### please test other weights in production run
    filtered_edges = [
        (u, v) for u, v, w in G.edges(data=True) if w["weight"] >= 0.1
    ]
    G_filtered = G.edge_subgraph(filtered_edges).copy()
    G = G_filtered

    design_nodes_set = []
    for node in G_filtered.nodes():
        ### consider only nodes on interface
        if not protein_ca.iloc()[node]["interface_mask"]:
            continue
        chain = protein_ca.iloc()[node]["chain_id_original"]
        nodes_within_3_edges = nx.single_source_shortest_path_length(
            G_filtered, node, cutoff=2
        )
        reachable_nodes = list(nodes_within_3_edges.keys())
        reachable_nodes = [
            n
            for n in reachable_nodes
            if protein_ca.iloc()[n]["chain_id_original"] == chain
            and protein_ca.iloc()[n]["interface_mask"]
        ]
        design_nodes_set.append(reachable_nodes)

    def keep_non_overlapping_arrays(arrays):
        ### if there are two sets of nodes that overlap with each other
        ### we select the biggest one
        ### in production - consider to keep all sets of nodes for design
        arrays_sorted = sorted(arrays, key=len, reverse=True)
        selected_arrays = []
        seen_elements = set()
        for array in arrays_sorted:
            overlap = any(elem in seen_elements for elem in array)
            if not overlap:
                selected_arrays.append(array)
                seen_elements.update(array)
        return selected_arrays

    design_nodes_set = keep_non_overlapping_arrays(design_nodes_set)

    if not show_pymol:
        return design_nodes_set

    ### below the script to save the pymol session to visualize that
    ### nodes were selected correctly and we don't have major problems
    ### and edges makes sense

    import pymol
    from pymol import cmd

    Path(temp_path).mkdir(exist_ok=True)
    save_pdb(protein, temp_path + "/temp.pdb")
    cmd.do("delete *")
    cmd.set_key("F1", cmd.zoom, ["all within 5.0 of (sele)"], {"animate": 1})
    cmd.set_key("F2", cmd.zoom, [], {"animate": 1})
    cmd.do(f"load {temp_path}/temp.pdb, pr")
    cmd.do("remove heta")
    cmd.do("color gray, elem C")
    cmd.do("set ray_shadows,off")

    for i in G.nodes():
        r = protein_ca.iloc()[i]["residue_number_original"]
        c = protein_ca.iloc()[i]["chain_id_original"]
        sel1 = f"(resi {r} & chain {c})"
        if protein_ca.iloc()[i]["interface_mask"]:
            cmd.do(f"color green, {sel1} & elem C")
        else:
            cmd.do(f"color cyan, {sel1} & elem C")

    cmd.do(f"set cartoon_color=gray, pr")
    cmd.do(f"set sphere_scale, 0.15")
    cmd.do("set dash_gap, 0")

    for i1, i2 in G.edges():
        r1 = protein_ca.iloc()[i1]["residue_number_original"]
        c1 = protein_ca.iloc()[i1]["chain_id_original"]
        r2 = protein_ca.iloc()[i2]["residue_number_original"]
        c2 = protein_ca.iloc()[i2]["chain_id_original"]
        sel1 = f"(resi {r1} & chain {c1})"
        sel2 = f"(resi {r2} & chain {c2})"
        cmd.do(f"distance d_{r1}{c1}_{r2}{c2}, {sel1} & name CA, {sel2} & name CA")
        cmd.do(f"show spheres, {sel1} & name CA")
        cmd.do(f"show spheres, {sel2} & name CA")
        w = G.get_edge_data(i1, i2)["weight"]
        color = "gray"
        if w > 0.1:
            color = "cyan"
        if w > 0.25:
            color = "blue"
        if w > 0.5:
            color = "red"
        cmd.do(f"color {color}, d_{r1}{c1}_{r2}{c2}")
        cmd.do("hide labels, d_*")

    nodes_pdb_numbering = []
    for i, nodes in enumerate(design_nodes_set):
        sel = []
        nodes_pdb_numbering_ = []
        for node in nodes:
            r1 = protein_ca.iloc()[node]["residue_number_original"]
            c1 = protein_ca.iloc()[node]["chain_id_original"]
            sel_ = f"(resi {r1} & chain {c1})"
            sel.append(sel_)
            nodes_pdb_numbering_.append(r1)
        nodes_pdb_numbering.append(nodes_pdb_numbering_)
        sel = " or ".join(sel)
        cmd.do(f"create nodes_{i}, {sel}")
        #print(nodes)

    cmd.do("set sphere_scale, 0.25, nodes_*")

    if pymol_log_dir is not None:
        Path(pymol_log_dir).mkdir(exist_ok=True)

    v = """\
     0.239272401,    0.140876904,    0.960677564,\
    -0.970724106,    0.013394370,    0.239812493,\
     0.020918813,   -0.989933610,    0.139956385,\
     0.000182450,    0.000008287, -361.150665283,\
   -30.531616211,   24.303537369,   -8.271241188,\
   284.734039307,  437.567535400,  -20.000000000 """
    cmd.set_view(v)
    cmd.do("save " + pymol_log_dir + "/search_graph.pse")

    return nodes_pdb_numbering

    # exit(0)




class ProteinMPNNWrapper:
    """
    Class to inference proteinmpnn
    for the system of several subunits
    in case we want to optimize one of the subunit in context of other
    in case we want to co-optimize one of the subunit in context of other as well monomer itself
    """
    def __init__(self, config):
        self.model = load_protein_mpnn_model(model_type="vanilla", device="cpu")
        # model = load_protein_mpnn_model(model_type="vanilla", device="cpu")
        self.config = config
        self.DEVICE = self.config.other_settings.device
        self.work_dir = self.config.paths.work_dir

    def extract_coords(self, protein):
        """
        function to extract coordinates from the biopandas dataframe
        the serious issue is here, that the order of atom name should be "N", "C", "CA', "O" in PDB
        in the next version add fix of the order
        :param protein:
        :return:
        """
        xyz = []
        ban = []
        for r, res in protein.groupby(["residue_number"]):
            res = res[res["atom_name"].isin(["N", "C", "CA", "O"])]
            order_dict = {"N":0, "CA":1, "C":2, "O":3}
            res = res.sort_values(by='atom_name', key=lambda x: x.map(order_dict))

            # print(res.shape)
            assert res.shape[0] == 4
            #    ban.append(r[0])
            #    continue

            xyz.append(res[["x_coord", "y_coord", "z_coord"]].to_numpy())

        xyz1 = np.concatenate(xyz)
        return np.array(xyz1)  # , np.array(xyz2)


    def calculate_recovery(self, S, S_true, mask):
        """
        Function to calculate sequence recovery
        It is quite ugly right now. Show be rewritten in more elegant way
        :param S:
        :param S_true:
        :param mask:
        :return:
        """
        ##!!!! rewrite it
        seqs = N_to_AA(S.numpy())
        seqs_true = N_to_AA(S_true.numpy())
        m = mask.numpy()
        recs = []
        for i, m_ in enumerate(m):
            seq_pred = "".join([seqs[i][j] for j in range(len(m_)) if m_[j] == 1])
            seq_true = "".join([seqs_true[i][j] for j in range(len(m_)) if m_[j] == 1])

            #print(seq_pred)
            #print(seq_true)

            rec = [1 if s1 == s2 else 0 for s1, s2 in zip(seq_pred, seq_true)]
            recs.append(np.sum(rec) / len(rec))
        return recs



    def score_seq(self, sample, proteinmpnn_input, **kwargs):
        """
         get score of sequences which is NLL loss
         :param sample: proteinmpnn output sequence
         :param proteinmpnn_input: proteinmpnn input parameters
         :param score_swop: in this experiment we can put sequence calculated for the monomer to the dimer
         and vice versa. To evaluate how to sequence from the monomer fit to multimer.
         :return:
         """

        t = proteinmpnn_input
        t_score = {n: t[n] for n in ["X", "mask", "chain_encoding_all", "residue_idx"]}
        t_score["chain_M"] = t["chain_mask"]
        t_score["S"] = sample["S"]
        t_score["decoding_order"] = sample["decoding_order"]

        if "rand" in kwargs:
            randn = kwargs["rand"]
        else:
            randn = torch.randn(1, sample["S"].shape[1])

        with torch.no_grad():
            log_probs = self.model(
                randn=randn,#torch.randn(1, sample["S"].shape[1]),
                use_input_decoding_order=True,
                **t_score,
            )

        design_mask = proteinmpnn_input["mask"]
        if "design_mask" in kwargs:
            design_mask = kwargs["design_mask"]

        score = nll_score(sample["S"], log_probs, mask=design_mask)
        return {"nll_score":score,
                "log_probs":log_probs}


    def fix_unmasked_residues(self, sequence, t, design_mask):
        """
        This function is used to fix amino acid residues in the non design regions
        :param sequence: str sequence
        :param t: proteinmpnn input
        :param design_mask: design mask
        :return:
        """
        one_hot = 1 - torch.nn.functional.one_hot(
            torch.tensor(AA_to_N([sequence])), num_classes=21
        )
        one_hot[:, design_mask, :] *= 0
        t["omit_AA_mask"] = one_hot
        return t

    def prepare_sampling_input(
        self, protein_input, **kwargs
    ):
        """
        prepare input for proteinmpnn sampling
        :param protein_input: biopandas protein dataframe
        :param double_sampling: in this case we sample chain_id from both monomer and complex
        we do decoding based on probabilities p_i for residue i, which are p_i = p_i_monomer*p_i_complex
        it allows to more efficiently obtain mutations that both improve monomer state and complex
        :return:
        """

        protein = protein_input[protein_input["atom_name"] == "CA"]
        mask_ca = list(protein["design_mask"])
        design_mask = [True if m else False for m in mask_ca]
        mask = np.array([1 for s in mask_ca])
        chain_id = list(protein["chain_id_original"])

        sequence = get_sequence(protein)
        omit_AAs_np = np.zeros((protein.shape[0], 21))
        coords = self.extract_coords(protein_input)

        backbone = BackboneSample(
            bb_coords=self.extract_coords(protein_input),
            ca_only=False,
            chain_id="".join(chain_id),
            res_name=sequence,
            res_mask=mask,
        )

        t = backbone.to_protein_mpnn_input("sampling", device=self.DEVICE)
        n_atoms = len(sequence)
        t["X"] = torch.tensor(coords)
        t["X"] = t["X"].view(n_atoms, 4, 3)
        t["X"] = t["X"][None, ...].float()
        t["chain_mask"] = torch.tensor([1 for _ in sequence])[None, ...]
        t["chain_M_pos"] = torch.tensor([1 for _ in sequence])[None, ...]

        ### change only design_mask residues
        self.fix_unmasked_residues(sequence, t, design_mask)

        randn = torch.randn((1, backbone.n_residues), device=self.DEVICE)

        des_mask = torch.zeros(randn.shape)
        des_mask[:,torch.tensor(design_mask)] = 1.0

        randn = (des_mask + 0.0001) * (torch.abs(randn))

        if "design_chain" in kwargs:
            chain_mask = torch.tensor(list(protein["chain_id_original"]==kwargs["design_chain"]))
            chain_randn_mask = torch.zeros(randn.shape)
            chain_randn_mask[0, chain_mask] = 1.0
            randn = (chain_randn_mask + 0.0001) * (torch.abs(randn))

        return t, backbone, randn#decoding_order#torch.argsort(randn)


    def sample_protein(self, protein_input, **kwargs):
        """
        :param protein_input: protein pdb dataframe
        :return:
        """

        design_chain = None
        if "design_chain" in kwargs:
            design_chain=kwargs["design_chain"]

        protein = protein_input[protein_input["atom_name"] == "CA"]
        sequence = get_sequence(protein)

        t, backbone, randn = self.prepare_sampling_input(protein_input, **kwargs)
        decoding_order = torch.argsort(randn)
        #for i in decoding_order[0,:].numpy():
        #    print(i,protein.iloc()[i]["design_mask"],
        #          protein.iloc()[i]["chain_id_original"])

        design_mask = 1-torch.max(t["omit_AA_mask"], dim=2).values
        seq_design = "".join(
            [
                "_" if protein.iloc()[i]["design_mask"] else s
                for i, s in enumerate(sequence)
            ]
        )
        t = {n: torch.tensor(v, device=self.DEVICE) for n, v in t.items()}

        with torch.no_grad():
            t["temperature"] = 0.1
            sample = self.model.sample(
                randn=randn,
                **t,
            )

        rec_1 = self.calculate_recovery(sample["S"],
                                      t["S_true"],
                                      mask=torch.ones(sample["S"].shape)
                                      )
        rec_2 = self.calculate_recovery(sample["S"],
                                      t["S_true"],
                                      mask=design_mask
                                      )
        scores = self.score_seq(sample, t, randn=randn)

        return {"full_recovery": rec_1,
               "mask_recovery": rec_2,
               "nll_score":scores["nll_score"],
               "log_probs":scores["log_probs"][design_mask.bool()],
               "S":sample["S"]}



def test():
    protein = load_protein("../example/3rdd.pdb")
    protein = load_protein("../example/test_input.pdb")
    protein = load_protein("../example/6wmk.pdb")

    add_interface_mask_column(protein, chain="A")
    protein["design_mask"] = (protein["interface_mask"]) & (protein["chain_id_original"]=="A")
    #print(protein)

    config = OmegaConf.load("../example/config.yaml")
    pw = ProteinMPNNWrapper(config)

    pw.sample_protein(protein, design_chain="A")


def main():
    test()
    pass

if __name__ == "__main__":
    main()
