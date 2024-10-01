import numpy as np
import torch
import proteinmpnn
from scipy import stats
from proteinmpnn.run import load_protein_mpnn_model, set_seed, nll_score
from proteinmpnn.data import BackboneSample, untokenise_sequence
from proteinmpnn.ProteinMPNN.protein_mpnn_utils import tied_featurize
from PPInterface.openfold_wrapper import OpenFoldWraper
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
        print(nodes)

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
            # print(res.shape)
            if res.shape[0] != 4:
                ban.append(r[0])
                continue
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

            print(seq_pred)
            print(seq_true)

            rec = [1 if s1 == s2 else 0 for s1, s2 in zip(seq_pred, seq_true)]
            recs.append(np.sum(rec) / len(rec))
        return recs



    def score_seq_classic(self, sample, proteinmpnn_input, randn, design_mask=None):
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

        with torch.no_grad():
            log_probs = self.model(
                randn=randn,#torch.randn(1, sample["S"].shape[1]),
                use_input_decoding_order=True,
                **t_score,
            )
        mask = proteinmpnn_input["mask"]
        if design_mask is not None:
            mask =design_mask
        score = nll_score(sample["S"], log_probs, mask=design_mask)
        return score
        #sample_swop = torch.stack([sample["S"][1, ...], sample["S"][0, ...]], dim=0)
        #score_swop = nll_score(sample_swop, log_probs, mask=proteinmpnn_input["mask"])
        #return {"log_probs": log_probs, "score": score, "score_swap": score_swop}

    def score_seq(self, sample, proteinmpnn_input, score_swop=True):
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
        with torch.no_grad():
            log_probs = self.model(
                randn=torch.randn(1, sample["S"].shape[1]),
                use_input_decoding_order=True,
                **t_score,
            )
        score = nll_score(sample["S"], log_probs, mask=proteinmpnn_input["mask"])
        sample_swop = torch.stack([sample["S"][1, ...], sample["S"][0, ...]], dim=0)
        score_swop = nll_score(sample_swop, log_probs, mask=proteinmpnn_input["mask"])
        return {"log_probs": log_probs, "score": score, "score_swap": score_swop}


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
        self, protein_input, chain_id_design="A", double_sampling=True
    ):
        """
        prepare input for proteinmpnn sampling
        :param protein_input: biopandas protein dataframe
        :param chain_id_design: chain id of designed sequence
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

        if not double_sampling:
            return t

        ### if we do simulatenous sampling we combine within one both features for monomer and multimer
        for k, v in t.items():
            if k in ["omit_AAs_np", "bias_AAs_np"]:
                continue
            v_add = v
            if k == "mask":
                v_mask = torch.tensor(
                    [
                        1 if p["chain_id_original"] == chain_id_design else 0
                        for p in protein.iloc()
                    ]
                    , device=self.DEVICE)[None, :]
                v_add = v_mask
            t[k] = torch.concat([v, v_add])
        return t, backbone


    def interface_recovery_sampler(self, protein_input,
                                   chain_id_design="A",
                                   n_batch=1,
                                   temperature=0.1):
        """
        :param protein_input: protein pdb dataframe
        :param chain_id_design:  chain to design
        :param classic: Boolean. If fasle - just do the sample of monomer and multimer separately,
        in True - do sampling of monomer and multimer simultaneously. in this case we sample chain_id from both monomer and complex
        we do decoding based on probabilities p_i for residue i, which are p_i = p_i_monomer*p_i_complex
        it allows to more efficiently obtain mutations that both improve monomer state and complex
        :return:
        """

        protein = protein_input[protein_input["atom_name"] == "CA"]
        mask_ca = list(protein["design_mask"])
        design_mask = [True if m else False for m in mask_ca]
        sequence = get_sequence(protein)

        t, backbone = self.prepare_sampling_input(protein_input, chain_id_design)
        for k,t_ in t.items():
            t_shape = list(t_.shape)
            t_shape[0] = n_batch
            if not k.endswith("np"):
                t_ = t_[:1, ...]
                t_ = t_.expand(*t_shape)
                t[k] = t_
        #exit(0)
        #    print(k, t[k].shape)

        randn = torch.randn((1, backbone.n_residues), device=self.DEVICE)
        mask = torch.ones(len(sequence))
        mask[design_mask] = -1
        randn = torch.randn(1, len(sequence))
        randn = (mask + 0.0001) * (torch.abs(randn))
        seq_design = "".join(
            [
                "_" if protein.iloc()[i]["design_mask"] else s
                for i, s in enumerate(sequence)
            ]
        )
        t["temperature"] = 0.1
        t = {n: torch.tensor(v, device=self.DEVICE) for n, v in t.items()}
        sample = self.model.sampler(
            randn=randn,
            **t,
        )
        mask_rec = t["mask"].clone()
        mask_rec[:, ~torch.tensor(design_mask)] *= 0

        rec = self.calculate_recovery(sample["S"], t["S_true"], mask_rec)
        #print(rec)
        scores = self.score_seq_classic(sample, t, randn, design_mask=mask_rec)
        #print(scores)
        #exit(0)
        #score_seq_classic
        #scores = self.score_seq(sample, t)
        rec = self.calculate_recovery(sample["S"], t["S_true"], mask_rec)

        old_seq = N_to_AA(t["S_true"][0])
        new_seq = N_to_AA(sample["S"][0])

        protein_mutant = protein.copy()
        protein_mutant_ca = protein[protein_mutant["atom_name"]=="CA"].reset_index(drop=True)
        mut_codes = []
        for i in range(len(design_mask)):
            if not design_mask[i]:
                continue
            res = protein_mutant_ca.iloc()[i]
            resi = res["residue_number_original"]
            chain = res["chain_id_original"]
            s_before = old_seq[i]
            s_after = new_seq[i]
            mut_codes.append({"aa_before": s_before, "resi": resi,
             "chain_id": chain,
             "aa_after": s_after})
        #print(len(mut_codes))
        #exit(0)
        protein_mutant = mutate_protein(protein_input, mut_codes)

        return {"pdb":protein_mutant,
                "score":scores[0].detach().cpu().numpy(),
                "recovery":rec[0]}

    def classic_sample_protein(self, protein_input, chain_id_design="A", classic=False):
        """
        :param protein_input: protein pdb dataframe
        :param chain_id_design:  chain to design
        :param classic: Boolean. If fasle - just do the sample of monomer and multimer separately,
        in True - do sampling of monomer and multimer simultaneously. in this case we sample chain_id from both monomer and complex
        we do decoding based on probabilities p_i for residue i, which are p_i = p_i_monomer*p_i_complex
        it allows to more efficiently obtain mutations that both improve monomer state and complex
        :return:
        """
        protein = protein_input[protein_input["atom_name"] == "CA"]
        mask_ca = list(protein["design_mask"])
        design_mask = [True if m else False for m in mask_ca]
        sequence = get_sequence(protein)
        t, backbone = self.prepare_sampling_input(protein_input, chain_id_design)
        randn = torch.randn((1, backbone.n_residues), device=self.DEVICE)

        seq_design = "".join(
            [
                "_" if protein.iloc()[i]["design_mask"] else s
                for i, s in enumerate(sequence)
            ]
        )

        t["temperature"] = 1.0#0.1

        t = {n: torch.tensor(v, device=self.DEVICE) for n, v in t.items()}


        sample = self.model.sampler(
            randn=randn,
            **t,
        )

        mask_rec = t["mask"].clone()
        mask_rec[:, ~torch.tensor(design_mask)] *= 0

        scores = self.score_seq(sample, t )

        return sample["S"], scores

    def sample_protein(self, protein_input, chain_id_design="A", classic=False):
        """
        :param protein_input: protein pdb dataframe
        :param chain_id_design:  chain to design
        :param classic: Boolean. If fasle - just do the sample of monomer and multimer separately,
        in True - do sampling of monomer and multimer simultaneously. in this case we sample chain_id from both monomer and complex
        we do decoding based on probabilities p_i for residue i, which are p_i = p_i_monomer*p_i_complex
        it allows to more efficiently obtain mutations that both improve monomer state and complex
        :return:
        """

        protein = protein_input[protein_input["atom_name"] == "CA"]
        mask_ca = list(protein["design_mask"])
        design_mask = [True if m else False for m in mask_ca]
        sequence = get_sequence(protein)
        t, backbone = self.prepare_sampling_input(protein_input, chain_id_design)
        randn = torch.randn((1, backbone.n_residues), device=self.DEVICE)
        seq_design = "".join(
            [
                "_" if protein.iloc()[i]["design_mask"] else s
                for i, s in enumerate(sequence)
            ]
        )
        t = {n: torch.tensor(v, device=self.DEVICE) for n, v in t.items()}
        with torch.no_grad():
            if classic:
                ###
                ### This is a little modified decoder of the proteinmpnn
                ### in this case we provide input [features_monomer, features_complex]
                ### we calculate probabilities during decoding from both features of monomer
                ### and features of complex
                ### we provide output amino acid letter that satisfies both probabilities
                assert 1 == 2
                sample = self.model.double_sampler(
                    randn=randn,
                    **t,
                )
            else:
                ###
                ### Classical proteinmpnn decoder
                ###
                t["temperature"] = 0.001
                sample = self.model.sampler(
                    randn=randn,
                    **t,
                )

        mask_rec = t["mask"].clone()
        mask_rec[:, ~torch.tensor(design_mask)] *= 0
        scores = self.score_seq(sample, t)
        return sample["S"], scores



    def gen_connectivity_graph(self,
                               protein_input,
                               design_chain,
                               postfix=""):
        ###
        ### This function is implemented to reconstruct graph
        ### where each node is amino acid residues
        ### and each edge indicate possible epistatic effects when changing of the amino acid residues
        ### it is calculated by predicted the average changes in probabilities of  neighbor amino acid residues letters
        ### when that selected node is mutated

        model = self.model
        work_dir = os.path.join(self.work_dir, "screen_graphs/")
        Path(work_dir).mkdir(exist_ok=True)

        ### extract coordiantes in the PDB file
        xyz = protein_input[["x_coord", "y_coord", "z_coord"]].to_numpy()

        ### first calculate the graph based on distance between residues

        cd = distance.cdist(xyz, xyz)
        G_dist = nx.Graph()
        n = 0
        protein_input["CA_index"] = None
        for r, p in protein_input.groupby(["residue_number"]):
            protein_input.loc[p.index, "CA_index"] = n
            n += 1

        protein = protein_input[protein_input["atom_name"] == "CA"]
        seq = get_sequence(protein)

        ### if amino acid residue atoms within 6.0 A from each other
        ### we add them to the garaph
        ### take into account that 6.0 is empirical value and should be evaluated in the production run
        ### 6.0 ~ 2 hydrogen bonds distance

        chain_id = list(protein["chain_id_original"])
        for atom_1_i, atom_2_i in zip(*np.where(cd < 6.0)):
            r1_index = protein_input.iloc()[atom_1_i]["CA_index"]
            r2_index = protein_input.iloc()[atom_2_i]["CA_index"]
            if (r1_index, r2_index) not in G_dist.edges():
                G_dist.add_edge(r1_index, r2_index)

        ### prepare proteinmpnn input for scoring
        protein_input["design_mask"] = False
        inp = self.prepare_sampling_input(protein_input,
                                          design_chain,
                                          double_sampling=False)

        inp.pop("omit_AA_mask")
        inp["chain_M"] = inp.pop("chain_mask")
        inp["S"] = inp.pop("S_true")
        inp = {s: inp[s] for s in ["X", "S", "mask", "chain_M", "residue_idx", "chain_encoding_all"]}
        inp["chain_encoding_all"] = inp["chain_M"]

        G = nx.Graph()


        for i in range(len(seq)):
            ### iterate over each amino acid residue on the binding interface of proteins
            if not protein.iloc()[i]["interface_mask"]:
                continue

            ### find neighbors according to distance proximity
            ids = G_dist.neighbors(i)
            neighbors = [i_ for i_ in list(set(ids)) if i != i_]

            ### update the decoding order, so the reference/mutating amino acid residues i
            ### is mutated first, and residues that we want to evaluated the changes in the logits values/probs
            ### are decoded last
            mask = torch.zeros(len(seq))
            mask[neighbors] = 1000.0
            mask[i] = -1
            old_log_probs = None
            new_log_probs = []
            randn = torch.randn(1, len(seq))
            randn = (mask + 0.0001) * (torch.abs(randn))
            decoding_order = torch.argsort(randn)

            ### mutate the i residues to each amino acid and calculate changes in probabilities
            for aa in "ARNDCQEGHILKMFPSTWYV":
                print(protein.iloc()[i]["residue_number_original"], protein.iloc()[i]["residue_name"], aa)
                seq_upd = seq[:i] + aa + seq[i + 1:]
                inp["S"] = torch.tensor(AA_to_N(seq_upd).T)
                inp["decoding_order"] = decoding_order

                with torch.no_grad():
                    log_probs = model(randn=randn,
                                      use_input_decoding_order=True,
                                      **inp)
                if aa == seq[i]:
                    ### these are log probabilities that were in the wt- sequence
                    old_log_probs = log_probs[0, ...]
                else:
                    ### these are log probabilities that were in the mutant- sequence
                    new_log_probs.append(log_probs[0, ...])

            new_log_probs = torch.stack(new_log_probs, dim=0)
            new_log_probs = torch.mean(new_log_probs, dim=0)
            new_log_probs = torch.exp(new_log_probs)
            old_log_probs = torch.exp(old_log_probs)

            ### add edge within the graph i,j, is changes in average probabilities of residue j upon mutationg
            ### residue i were > 0.05
            ### 0.05 selected empirically
            ### in the production consider other cut offs

            for n in neighbors:
                delta = torch.sum(torch.abs(new_log_probs[n, :] - old_log_probs[n, :])).numpy()
                G.add_edge(i, n, weight=delta)

        ### save the epistatis graph in the work dir
        pickle.dump(G, open(os.path.join(work_dir, f"search_graph{postfix}.pkl"), 'wb'))


class ProteinmpnnddgWrapper:
    """
    Class that serves as wrapper to inference the model proteinmpnnddg model
    This is nice adjustment of the proteinmpnn model that significantly improves the inference time
    Currently it is only used to evaluate amino acid residues that significantly affect protein-protein interaction affinity
    """
    def __init__(self, config):
        self.model_name = (
            "v_48_020"  # @param ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
        )
        self.config = config
        self.work_dir = config.paths.temp_dir
        Path(self.work_dir).mkdir(exist_ok=True)

    def predict_mutants(self, pdb_df):
        """
        This function is currently never used, but keep for later
        :param pdb_df: PDB dataframe
        :return: set of mutants and their ddg
        """
        pdb_path = os.path.join(self.work_dir, "temp.pdb")
        save_pdb(pdb_df, pdb_path)
        chain_to_predict = pdb_df[pdb_df["design_mask"]].iloc()[0]["chain_id_original"]
        chains = [c for c in set(pdb_df["chain_id_original"]) if c != chain_to_predict]

        df = predict_logits_for_all_point_mutations_of_single_pdb(
            self.model_name,
            chains,
            pdb_path,
            nrepeat=1,  # nrepeats,
            seed=42,  # seed,
            chain_to_predict=chain_to_predict,
            pad_inputs=False,
            apply_ddG_correction=True,
        )
        design_mask = pdb_df[pdb_df["design_mask"]]["residue_number_original"]
        df = df[df["pos"].isin(design_mask)]
        #df = df[df["logit_difference_ddg"] >= -0.25]
        return df

    def get_binding_ddg(self, pdb_df):
        """
        """
        data = {}
        ### chain_ID of design chain
        binder_chain = [pdb_df[pdb_df["design_mask"]].iloc()[0]["chain_id_original"]]
        ### other chain IDS
        receptor_chains = [
            c for c in set(pdb_df["chain_id_original"]) if c not in binder_chain
        ]
        ### make temporary file to run inference of proteinmpnnddg
        ### not nice way , in future should be fixed
        pdb_path = os.path.join(self.work_dir, "temp.pdb")
        save_pdb(pdb_df, pdb_path)



        ### for each design residue predict effect of all substitutions
        ### both on monomer and multimer
        for source, chains_to_predict, context_chains in [
            ("unbound", binder_chain, []),
            ("bound", binder_chain, receptor_chains),
        ]:
            dfs = []
            for chain in chains_to_predict:
                df = predict_logits_for_all_point_mutations_of_single_pdb(
                    self.model_name,
                    chains_to_predict + context_chains,
                    pdb_path,
                    nrepeat=1,  # nrepeats,
                    seed=42,  # seed,
                    chain_to_predict=chain,
                    pad_inputs=False,
                    apply_ddG_correction=True,
                )
                df["chain"] = chain
                #df = df[df["pos"].isin(nodes_to_design)]
                dfs.append(df)

            df = pd.concat(dfs, axis=0)
            data[source] = df

        ### calculate ddg
        data_ddg = (
                data["bound"]["logit_difference_ddg"]
                - data["unbound"]["logit_difference_ddg"]
        )
        df["logit_difference_ddg_bound"] = data["bound"]["logit_difference_ddg"]
        df["logit_difference_ddg_unbound"] = data["unbound"]["logit_difference_ddg"]
        df["logit_difference_bound"] = data["bound"]["logit_difference"]
        df["logit_difference_unbound"] = data["unbound"]["logit_difference"]
        df.drop(columns=['logit_difference', 'logit_difference_ddg'], inplace=True)

        return df



    def filter_binder_affecting_residues(self, pdb_df, nodes_to_design_split):
        """
        Currently it is only used to evaluate amino acid residues that
        significantly affect protein-protein interaction affinity, e.g.
        predict hot spot residues and filter them out from design tasks
        :param pdb_df: PDB dataframe of complex
        :param nodes_to_design_split: list of lists of amino acid residues selected for design
        :return:
        """
        data = {}
        ### chain_ID of design chain
        binder_chain = [pdb_df[pdb_df["interface_mask"]].iloc()[0]["chain_id_original"]]
        ### other chain IDS
        receptor_chains = [
            c for c in set(pdb_df["chain_id_original"]) if c not in binder_chain
        ]
        ### make temporary file to run inference of proteinmpnnddg
        ### not nice way , in future should be fixed
        pdb_path = os.path.join(self.work_dir, "temp.pdb")
        save_pdb(pdb_df, pdb_path)

        ###r residues on the binidng intreface
        binder_interface_residues = list(
            pdb_df[pdb_df["interface"]]["residue_number_original"]
        )

        nodes_to_design = [
            item for sublist in nodes_to_design_split for item in sublist
        ]


        ### for each design residue predict effect of all substitutions
        ### both on monomer and multimer
        for source, chains_to_predict, context_chains in [
            ("unbound", binder_chain, []),
            ("bound", binder_chain, receptor_chains),
        ]:
            dfs = []
            for chain in chains_to_predict:
                df = predict_logits_for_all_point_mutations_of_single_pdb(
                    self.model_name,
                    chains_to_predict + context_chains,
                    pdb_path,
                    nrepeat=1,  # nrepeats,
                    seed=42,  # seed,
                    chain_to_predict=chain,
                    pad_inputs=False,
                    apply_ddG_correction=True,
                )
                df["chain"] = chain
                df = df[df["pos"].isin(nodes_to_design)]
                dfs.append(df)

            df = pd.concat(dfs, axis=0)
            data[source] = df

        ### calculate ddg
        data_ddg = (
            data["bound"]["logit_difference_ddg"]
            - data["unbound"]["logit_difference_ddg"]
        )
        data["bound"]["binding_ddg"] = data_ddg
        df = data["bound"]

        ### for our task we remove residues, which has > 15 mutations with destabilizing ddG
        ### it indicates that they are likely hot spot, and we don't want to touch them
        ###
        better_not_touch_residues = []
        for node in nodes_to_design:
            resi = df[df["pos"] == node]
            n_ok = resi[resi["binding_ddg"] > -1.0].shape[0]
            if n_ok < 15:
                better_not_touch_residues.append(node)
            if (
                node not in binder_interface_residues
            ):  # binder_interface["residue_number_original"]:
                continue
            n_not_ok = resi[resi["logit_difference_ddg"] < -2.0].shape[0]
            if n_not_ok > 15:
                better_not_touch_residues.append(node)

        nodes_to_design = [n for n in better_not_touch_residues]
        nodes_to_design_split = [
            [n_ for n_ in n if n_ not in better_not_touch_residues]
            for n in nodes_to_design_split
        ]
        nodes_to_design_split = [n for n in nodes_to_design_split if len(n) != 0]
        ### return list of clusters of residues to design without hot spot residues
        return nodes_to_design_split




def generate_search_space():
    config = OmegaConf.load("configs/test_ace2.yaml")
    design_chain = config.design_task.protein_design_chain

    protein_full = load_protein(config.design_task.protein_design)

    device = config.other_settings.device
    model = load_protein_mpnn_model(model_type="ca", device="cpu")
    add_interface_mask_column(protein_full, design_chain)

    exit(0)
    gen_connectivity_graph(
        protein_full,
        model=model,
        DEVICE=config.other_settings.device,
        work_dir=config.paths.work_dir,
        postfix="_design",
    )

    exit(0)

    protein_target = load_protein(config.design_task.protein_target)
    model = load_protein_mpnn_model(model_type="ca", device="cpu")
    add_interface_mask_column(protein_target, config.design_task.protein_target_chain)
    gen_connectivity_graph(
        protein_target,
        model=model,
        DEVICE=config.other_settings.device,
        work_dir=config.paths.work_dir,
        postfix="_target",
    )


def select_optimization_residue_sets():
    config = OmegaConf.load("configs/test_ace2.yaml")
    pr = ProteinMPNNWrapper(config)
    protein_design = load_protein(config.design_task.protein_design)
    protein_target = load_protein(config.design_task.protein_target)

    add_interface_mask_column(protein_target, config.design_task.protein_target_chain)
    add_interface_mask_column(protein_design, config.design_task.protein_design_chain)
    protein_design["interface"] = protein_design["interface_mask"]

    design_chain = config.design_task.protein_design_chain
    for residue in protein_target[protein_target["interface_mask"]].iloc():
        resi = residue["residue_number_original"]
        r = (protein_design["residue_number_original"] == resi) & (
            protein_design["chain_id_original"]
            == config.design_task.protein_design_chain
        )
        protein_design.loc[r, "interface_mask"] = True
    protein_design["b_factor"] = protein_design["interface_mask"]

    pr.gen_connectivity_graph(protein_design,
                              design_chain=design_chain,
                              postfix="design")

    graph_path = os.path.join(
        config.paths.work_dir, f"screen_graphs/search_graphdesign.pkl"
    )
    nodes_to_design = split_graph(
        protein_design,
        G=pickle.load(open(graph_path, "rb")),
        show_pymol=True,
        show_matplotlib=False,
        pymol_log_dir=config.paths.pymol_log_dir,
    )

    # print(nodes_to_design)
    pr_ddg = ProteinmpnnddgWrapper(config)

    ###
    ### Here we remove nodes that can't be mutated without affecting significantly the binding to antibody
    ###
    nodes_to_design = pr_ddg.filter_binder_affecting_residues(
        protein_design, nodes_to_design
    )

    json.dump(
        [[int(n_) for n_ in n] for n in nodes_to_design],
        open(
            os.path.join(config.paths.work_dir, "screen_graphs/design_residues.json"),
            "w",
        ),
    )


def plot_clusters():
    config = OmegaConf.load("configs/test_ace2.yaml")
    nodes_to_design = json.load(
        open(os.path.join(config.paths.work_dir, "screen_graphs/design_residues.json"))
    )
    # print(nodes_to_design)
    # exit(0)
    protein_design = load_protein(config.design_task.protein_design)
    add_interface_mask_column(protein_design, config.design_task.protein_design_chain)
    plot_design_nodes_in_pymol(protein_design, nodes_to_design, config)


def main():
    pass

if __name__ == "__main__":
    main()
