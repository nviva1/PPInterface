# Pseudocode for Metropolis-Hastings MCMC with Simulated Annealing
import random
import math
import numpy as np
import os
from PPInterface.protein_utils import load_protein, save_pdb, prepare_optimization_input, get_sequence, mutate_protein, aa_3_to_1, \
    aa_1_to_3, kalign, mutate_protein_from_list
from omegaconf import OmegaConf
from abc import ABC, abstractmethod
from typing import Callable
import torch
import json
from PPInterface.protein_utils import add_interface_mask_column
#from PPInterface.rosetta_wrapper import RosettaWrapper
from PPInterface.proteinmpnn_wrapper import  AA_to_N, N_to_AA, ProteinMPNNWrapper
import pandas as pd
import numpy as np
import transformers
from pathlib import Path
from transformers import EsmTokenizer, EsmForMaskedLM
from PPInterface.openfold_wrapper import OpenFoldWraper, download_weights
import logging
from PPInterface.sampler import MCSampler, MCState

logger = logging.getLogger('ppi_logger')
logger.setLevel(logging.DEBUG)  # Set the logging level




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


def test_proteinmpnn_wrapper(config):
    pw = ProteinMPNNWrapper(config)
    protein_design = prepare_optimization_input(config)
    r_des = pw.sample_protein(protein_design,
                              chain_id_design=config.design_task.protein_design_chain)
    print(r_des)
    exit(0)


def sample_sequences():
    ### get alphafold weights
    download_weights()

    ### Log config file
    config_file = OmegaConf.load("example/config_complex.yaml")
    Path(config_file.paths.work_dir).mkdir(exist_ok=True)
    prepare_logger(config_file)

    ### Protein PDB dataframe with labeled interface and design residues
    protein_design = prepare_optimization_input(config_file)

    ### Simple Monte-Carlo sampler with ProteinMPNN probabilities and OpenFold metrics
    mc_sampler = MCSampler(config=config_file, logger=logger)
    state = MCState(protein_design)

    for i in range(10):
        output = mc_sampler.singe_iteration(state)
        print(output)


    pass

def main():
    sample_sequences()

if __name__ == "__main__":
    main()
