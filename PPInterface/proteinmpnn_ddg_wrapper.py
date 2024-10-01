import os

import proteinmpnn_ddg
from omegaconf import OmegaConf
from proteinmpnn_ddg import predict_logits_for_all_point_mutations_of_single_pdb
import numpy as np
import json
import pandas as pd
from pathlib import Path
import warnings, os, re

# from mh_mcmc import prepare_optimization_tasks
import pickle
from PPInterface.protein_utils import load_protein, save_pdb, get_sequence, mutate_protein
from typing import List
import numpy as np
import pandas as pd


def get_pdb(pdb_code=""):
    if pdb_code is None or pdb_code == "":
        upload_dict = files.upload()
        pdb_string = upload_dict[list(upload_dict.keys())[0]]
        with open("tmp.pdb", "wb") as out:
            out.write(pdb_string)
        return "tmp.pdb"
    elif os.path.isfile(pdb_code):
        return pdb_code
    elif len(pdb_code) == 4:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"
    else:
        os.system(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb"
        )
        return f"AF-{pdb_code}-F1-model_v3.pdb"


class ProteinmpnnddgWrapper:
    """
    Helper class to predict ddg using proteinmpnn_ddgg
    """

    def __init__(self, config):
        self.model_name = "v_48_020"
        self.config = config
        self.work_dir = config.paths.temp_dir
        Path(self.work_dir).mkdir(exist_ok=True)

    def predict_mutants(self, pdb_df):
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
        return df

        design_mask = pdb_df[pdb_df["design_mask"]]["residue_number_original"]

        df = df[df["pos"].isin(design_mask)]
        df = df[df["logit_difference_ddg"] >= -0.25]
        print(df)

    # exit(0)
    def filter_binder_affecting_residues(self, pdb_df, nodes_to_design_split):
        """
        Inference ProteinMPNN-ddG for protein-protein complex
        Detect residues that change ddG too much
        Filter them from residues for design
        :param pdb_df: input protein complex dataframe
        :param nodes_to_design_split: nodes that are considered for optimization
        :return:
        """
        data = {}
        binder_chain = [pdb_df[pdb_df["design_mask"]].iloc()[0]["chain_id_original"]]
        receptor_chains = [
            c for c in set(pdb_df["chain_id_original"]) if c not in binder_chain
        ]
        pdb_path = os.path.join(self.work_dir, "temp.pdb")
        save_pdb(pdb_df, pdb_path)

        # nodes_to_design_split = json.load(open("./run_1/screen_graphs/design_residues.json"))
        nodes_to_design = [
            item for sublist in nodes_to_design_split for item in sublist
        ]

        for source, chains_to_predict, context_chains in [
            ("unbound", binder_chain, []),
            ("bound", binder_chain, receptor_chains),
        ]:
            dfs = []
            for chain in chains_to_predict:
                df = predict_logits_for_all_point_mutations_of_single_pdb(
                    model_name,
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

        data_ddg = (
            data["bound"]["logit_difference_ddg"]
            - data["unbound"]["logit_difference_ddg"]
        )
        data["bound"]["binding_ddg"] = data_ddg
        df = data["bound"]

        return df

        better_not_touch_residues = []
        for node in nodes_to_design:
            resi = df[df["pos"] == node]
            n_ok = resi[resi["binding_ddg"] > -1.0].shape[0]
            if n_ok < 15:
                better_not_touch_residues.append(node)

        nodes_to_design = [n for n in better_not_touch_residues]
        nodes_to_design_split = [
            [n_ for n_ in n if n_ not in better_not_touch_residues]
            for n in nodes_to_design_split
        ]
        nodes_to_design_split = [n for n in nodes_to_design_split if len(n) != 0]
        return nodes_to_design_split


def main():
    pass
    #config = OmegaConf.load("configs/test_ace2.yaml")
    #protein_design, protein_target = prepare_optimization_tasks(config, 0)
    #pr = ProteinmpnnddgWrapper(config)


if __name__ == "__main__":
    main()
