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


class MCState:
    """
    Object for the state of the protein complex for optimization algorithm
    state variable indicate current sequence of designable residues
    """
    def __init__(self,
                 protein,
                 pose=None):
        """
        :param protein: PDB dataframe of the designable protein.
        design_mask column indicate residues that should be optimized
        :param pose: rosetta pose
        """

        ### we remove sides chaines for the designable residues to avoid bias
        self.protein = protein.copy()
        c1 = (self.protein["atom_name"].isin(["N", "C", "CA", "O"])) & (self.protein["design_mask"])
        c2 = (~self.protein["design_mask"])
        self.protein = self.protein[c1 | c2]

        ### PDB dataframe but only with CA atoms
        self.protein_ca = self.protein[self.protein["atom_name"] == "CA"].reset_index(drop=True)
        self.protein_ca["aa_original"] = list(get_sequence(self.protein_ca))

        design_aa = "".join(list(self.protein_ca[self.protein_ca["design_mask"]]["aa_original"]))

        ### set state varible - current amino acid residues sequence
        self.state = design_aa
        self.pose = pose
        self.design_mask = self.protein_ca["design_mask"]


        ### set variable with chain id of subunit that can be designed
        sel = self.protein_ca["design_mask"]
        chains = list(set(self.protein_ca["chain_id_original"]))
        design_chain = self.protein_ca[sel]["chain_id_original"].iloc()[0]
        non_design_chains = ",".join([c for c in chains if c != design_chain])
        self.design_chains = design_chain

        ### set variable with chain id of constant subunits
        self.non_design_chains = non_design_chains

        self.ddg = None

        ### openfold refined protein
        self.openfold_protein = None

    def update_protein(self):
        ### update PDB dataframe in case we introduced mutations as defined by self.state variable
        sel = self.protein_ca["design_mask"]
        new_seq = self.state
        old_seq = self.protein_ca[sel]["aa_original"]
        resi = self.protein_ca[sel]["residue_number"]
        mutant_codes = []
        ### basicaly we just change residue names in the PDB dataframe
        ### because designable residues have only backbone atoms
        for residue_number, s_old, s_new in zip(list(resi), list(old_seq), list(new_seq)):
            mutant_codes.append([s_old, residue_number, s_new])
            con = self.protein["residue_number"] == residue_number
            self.protein.loc[con, 'residue_name'] = aa_1_to_3[s_new]
        pass

    def get_design_sequence(self):
        ### return full sequence of designable subunit
        pca = self.protein[self.protein["atom_name"] == "CA"]
        pca = pca[pca["chain_id_original"] == self.design_chains]
        return get_sequence(pca), list(pca["design_mask"])


    def copy(self):
        ### make copy of the object
        new_obj = MCState(self.protein, self.pose)
        new_obj.state = self.state
        return new_obj

class MonteCarloOptimizer():
    """
    Implementation of ProteinMPNN probabilities guided MC algorithm
    """

    def __init__(self, config, **kwargs):
        self.rw = None
        ### use pyrosetta to calculate metrics
        if config.metrics.rosetta:
            self.rw = RosettaWrapper()

        ### use openfold to calculate metrics
        if config.metrics.openfold:
            self.ow = OpenFoldWraper(weights_path=config.paths.alphafold_weights_path,
                                     device=config.other_settings.openfold_device)

        self.config = config
        self.chain_id = None
        self.pw = ProteinMPNNWrapper(config)

        ### implement ESM LLM with 650M parameters
        ### in the next version consider using larger models
        self.esm_device = "cpu"  # mps"
        self.esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm_model = EsmForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(self.esm_device)

    def pae_scores_from_openfold(self, protein, of_output):
        """
        Function to select confidence metrics from the openfold inference output
        we calculate average PAE only between interface residues
        we calculate average pLDDT only more mutant positions
        :param protein: PDB dataframe
        :param of_output: openfold output
        :return:
        """

        protein_ca = protein[protein["atom_name"] == "CA"].reset_index(drop=True)
        ids = list(protein_ca[protein_ca["interface_mask"]].index)
        pae = of_output["predicted_aligned_error"]
        chain_ref = protein_ca[protein_ca["interface_mask"]].iloc()[0]["chain_id_original"]
        other_chain_ids = list(protein_ca[protein_ca['chain_id_original'] != chain_ref].index)
        pae1 = np.average(pae[:, ids][other_chain_ids, :])
        pae2 = np.average(pae[ids, :][:, other_chain_ids])
        plddt = np.average(of_output["plddt"][ids])

        return {"pae": np.average([pae1, pae2]),
                "plddt": plddt}

    def openfold_score(self, state):
        """
        Inference of modified version of MSA-free only template OpenFold
        return confidence metrics PAE and PLDDT
        :param state: MCState object
        :return:
        """

        state.update_protein()
        of_output, out_per_cycle, pdb_df_pred1 = self.ow.inference_monomer(
            state.protein, ### PDB dataframe
            n_recycle=2 ### 2 is OK , but 1 cycle might be enough. further study should be carried out
        )
        state.openfold_protein = pdb_df_pred1
        ### todo - add estimate of plddt change of the monomer in the future versions

        return self.pae_scores_from_openfold(state.protein, of_output)

    def esm_score(self, state):  # sequence, design_mask):
        sequence, design_mask = state.get_design_sequence()
        """
        in this implementation we evaluate pseudoperplexity score of the designed sequence
        of the designed chain. 
        The better idea could be to use ESM2-trained model that evaluates protein melting temperature
        or ddG upon multiple mutations
        However within this implementation we use this score
        Code is borrowed from https://www.kaggle.com/code/daehunbae/esm-2-pseudo-perplexity-ranking
        """
        tensor_input = self.esm_tokenizer.encode(sequence, return_tensors='pt')
        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, self.esm_tokenizer.mask_token_id)
        labels = repeat_input.masked_fill(masked_input != self.esm_tokenizer.mask_token_id, -100)
        masked_input = masked_input[design_mask, :]
        labels = labels[design_mask, :]
        with torch.no_grad():
            loss = self.esm_model(masked_input.to(self.esm_device), labels=labels.to(self.esm_device)).loss
        return np.exp(loss.item())

    def update_pose(self, state):
        """
        Update Rosetta pose based on the current state
        return the pyrosetta ddG
        important - input proteins better to be minimzed using rosetta forcefield prior to use
        how much it affects the results should be investigated.
        :param state:
        :return:
        """

        state.update_protein()
        new_seq = state.state
        design_residue_mask = state.protein_ca["design_mask"]
        mutant_codes = {}
        n = 0
        design_chains = ",".join(list(set(state.protein_ca[design_residue_mask]["chain_id_original"])))

        non_design_chains = [c for c in list(set(state.protein_ca[~design_residue_mask]["chain_id_original"])) if
                             c not in design_chains]

        non_design_chains = ",".join(non_design_chains)

        temp_path = os.path.join(self.config.paths.temp_dir, "temp.pdb")
        save_pdb(state.protein, temp_path)
        state.pose = self.rw.load_pose(temp_path)

        for r, res in state.protein_ca[design_residue_mask].groupby(
                ["insertion", "residue_number_original", "chain_id_original"], sort=False):
            mutant_codes[r] = new_seq[n]
            n += 1

        ddg = self.rw.mutate_interface_and_repack(state.pose,
                                                  mutant_codes,
                                                  chains_1=design_chains,
                                                  chains_2=non_design_chains)
        state.ddg = ddg
        return ddg

    def update_state(self, state_design,
                     new_seq):
        """
        ### change sequence in the previous state to new_seq
        :param state_design:  previous state
        :param new_seq: new sequence
        :return:
        """

        s_new = state_design.copy()
        s_new.state = new_seq
        s_new.update_protein()
        return s_new

    def calculate_metrics(self,
                          state):
        """
            function to calculate the metrics by Rosetta / Openfold / ESM2
            :param state_design:
            :param state_target:
            :return:
        """
        state.ddg = {}
        if self.config.metrics.rosetta:
            ddg_new_design = self.update_pose(state)
            state.ddg = {"rosetta_ddg_complex": ddg_new_design[0],
                        "rosetta_ddg_monomer": ddg_new_design[1]}
        if self.config.metrics.openfold:
            state.ddg.update(self.openfold_score(state))
        state.ddg["esm2_loss"] = self.esm_score(state)


    def single_guided_iteration(self,
                            state_design,
                            state_target,
                            prefix=""):
        """
        single step in the MC search and calculate of the objective function
        """

        if state_design.ddg is None:
            self.calculate_metrics(state_design)
            self.calculate_metrics(state_target)

        ### predict new sequence
        new_state_design, new_state_target = self.proposal_distribution(state_design,
                                   state_target)

        ### calculate metrics of new sequence
        self.calculate_metrics(new_state_design)
        self.calculate_metrics(new_state_target)

        delta_pae = new_state_target.ddg["pae"]-state_target.ddg["pae"]
        delta_plddt = new_state_target.ddg["plddt"]-state_target.ddg["plddt"]
        objective_function_bad = delta_pae - delta_plddt

        delta_pae = new_state_design.ddg["pae"]-state_design.ddg["pae"]
        delta_plddt = new_state_design.ddg["plddt"]-state_target.ddg["plddt"]
        objective_function_good = delta_pae - delta_plddt

        design_residues = list(state_design.protein_ca[state_design.protein_ca["design_mask"]]["residue_number"])

        output = {"new_sequence":new_state_design.state,
                  "old_sequence":state_design.state,
                  "residues":design_residues,
                  "metrics_good_complex":new_state_design.ddg,
                  "metrics_bad_complex":new_state_target.ddg
                  }

        ### oops it should be swapped
        ### change next time
        self.save_pdb(state_design, prefix=prefix+"good_wt_")
        self.save_pdb(state_target, prefix=prefix+"bad_wt_")
        self.save_pdb(new_state_design, prefix=prefix+"good_mutant_")
        self.save_pdb(new_state_target, prefix=prefix+"bad_mutant_")

        return

    def proposal_distribution(self, state_design):  # current_state):
        """
        Function to conduct ProteinMPNN guided proposal of the sequence changes
        We calculate probabilities of amino acid residues within PPI complexes
        We want to have high probabilities of amino acid residues within the first complex - state_design
        We want to have high probabilities of amino acid residues within the monomer subunit
        We want to have low probabilities of amino acid residues within the second complex - state_target

        :param state_design: MCState object for the PPI complex that we want to keep
        :param state_target: MCState object for the PPI complex that we want to break
        :return:
        """


        ### calculate probabilities using proteinmpnn for the designable chain in the good complex
        ### we use the non-standard decoder scheme (see sample_protein function)
        r_des = self.pw.sample_protein(state_design.protein,
                                       state_design.design_chains)

        probs = np.exp(r_des["log_probs"][0, ...].numpy())

        ### probability of deletions should be zero
        probs[:, -1] *= 0
        q = 1 / np.sum(probs, axis=-1)[:, None]
        probs = probs * q

        ### within monte carlo simulation we randomly select the amino acid substitutions
        ### based on the probabilities

        cumulative_sum = np.cumsum(probs, axis=1)
        n_iter = 100
        old_esm_score = self.esm_score(state_design)

        ###
        ### Since we don't want to evaluate sequences that destabilizes the monomer
        ### We use fast ESM2- based score to filter out such cases
        ### AA shouldn't increase the pseudo-perplexity of the sequence
        ### The effect of this step for the RBD should be evaluate in the future studies
        ### For some proteins ESM2 perplexity is not very good predictor
        ###

        delta_esm_score = 1
        n = 0
        while delta_esm_score > 0:
            ### repeate the random selection according to the ProteinMPNN probabilities of new aa
            ### repeat until delta_esm_score > 0

            random_vector = np.random.rand(cumulative_sum.shape[0])
            new_state = state_design.copy()
            ids = np.array(
                [np.searchsorted(cumulative_sum[ii], random_vector[ii]) for ii in range(random_vector.shape[0])])

            new_seq = N_to_AA([ids])[0]
            new_state.state = new_seq
            new_state.update_protein()
            seq = new_state.get_design_sequence()
            old_seq = state_design.state

            delta_esm_score = self.esm_score(new_state) - old_esm_score

            n += 1
            print({"new_sequence": new_seq,
                   "old_sequence": old_seq,
                   "delta_esm": delta_esm_score})

            if n > n_iter:
                print("Can't find good esm score!")
                break

        new_seq = new_state.state

        return (self.update_state(state_design, new_seq))

    def save_pdb(self, state, prefix=""):
        """
        save the metrics and structures for each state
        :param state:
        :param prefix:
        :return:
        """

        save_dir = os.path.join(self.config.paths.work_dir,"mc_sample")
        Path(save_dir).mkdir(exist_ok=True)
        ### save openfold if calculated
        if state.openfold_protein is not None:
            save_pdb(state.openfold_protein, os.path.join(save_dir, prefix+"_afold.pdb"))

        ### save rosetta pdb if calculcated
        if self.config.metrics.rosetta:
            state.pose.dump_pdb(os.path.join(save_dir, prefix + "_rosetta.pdb"))

        d = {n:float(v) for n,v in state.ddg.items()}
        d["aa_mutants"]=state.state
        json.dump(d, open(os.path.join(save_dir, prefix+".json"),'w'))


def main():
    pass


if __name__ == "__main__":
    main()


def main():
    pass

if __name__ == "__main__":
    main()