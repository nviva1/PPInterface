# Pseudocode for Metropolis-Hastings MCMC with Simulated Annealing
import random
import math
import numpy as np
import os
from PPInterface.protein_utils import load_protein, save_pdb, make_dummy_protein, get_sequence, mutate_protein, aa_3_to_1, \
    aa_1_to_3, kalign, mutate_protein_from_list
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from typing import Callable
import torch
import json
from PPInterface.protein_utils import add_interface_mask_column
from PPInterface.rosetta_wrapper import RosettaWrapper
from PPInterface.proteinmpnn_wrapper import  AA_to_N, N_to_AA, ProteinMPNNWrapper
import pandas as pd
import numpy as np
import transformers
from pathlib import Path
from transformers import EsmTokenizer, EsmForMaskedLM
from PPInterface.openfold_wrapper import OpenFoldWraper
import logging

logger = logging.getLogger('ppi_logger')
logger.setLevel(logging.DEBUG)  # Set the logging level


def prepare_optimization_tasks(config,
                               design_id=0):

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
    nodes_to_design = map(int,config.design_task.protein_design_residues.split(","))
    protein_design["design_mask"] = (protein_design["residue_number_original"].isin(nodes_to_design)) & (
                protein_design["chain_id_original"] == design_chain)
    return protein_design


def test_openfold_wrapper(config, protein):
    """
    This function inference openfold and output pae for interacting subunits
    :param config:
    :param protein:
    :return:
    """
    weights_path = "/Users/ivanisenko/projects/ProteinAIDesign/sasha/weights/params_model_2_ptm.npz"
    ofr = OpenFoldWraper(weights_path=weights_path, device=config.other_settings.openfold_device)
    of_output, out_per_cycle, pdb_df_pred = ofr.inference_monomer(protein, n_recycle=1)
    save_pdb(pdb_df_pred, config.paths.work_dir+"/ofr_ref.pdb")
    logger.info('OpenFold inference success')
    scores = ofr.get_ppi_metrics(protein, of_output, chain_id=config.design_task.protein_design_chain)
    logger.info(f"Scores: {scores}")

def prepare_logger(config):
    file_handler = logging.FileHandler(os.path.join(config.paths.work_dir, 'logfile.log'))
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main():
    config_file = OmegaConf.load("example/config.yaml")
    Path(config_file.paths.work_dir).mkdir(exist_ok=True)
    prepare_logger(config_file)
    protein_design = prepare_optimization_tasks(config_file)
    test_openfold_wrapper(config_file, protein_design)

    #print(protein_design)
    pass


if __name__ == "__main__":
    main()
