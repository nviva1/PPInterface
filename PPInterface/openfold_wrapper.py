"""
Code for the Inference of OpenFold
in the MSA free / template mode
using the BioPandas dataframe as an input
"""

from pathlib import Path
import sys
import PPInterface.openfold.np.protein as protein
import PPInterface.openfold.np.residue_constants as residue_constants
from PPInterface.openfold.config import model_config
from PPInterface.openfold.data import feature_pipeline
from PPInterface.openfold.utils.script_utils import load_models_from_command_line, prep_output
from PPInterface.openfold.utils.tensor_utils import dict_multimap, tensor_tree_map
from PPInterface.openfold.np import residue_constants
import numpy as np
from PPInterface.protein_utils import (
    load_protein,
    pdb_str_to_dataframe,
    save_pdb,)
from functools import partial
import torch
from scipy.spatial import distance
import json


class OpenFoldWraper:
    def __init__(
        self,
        device="cuda:0",
        weights_path="params_model_2_ptm.npz",
    ):
        self.device = device
        ### load alphafold model
        self.init_alphafold(weights_path)

    def prepare_features(self, target_protein):
        """prepare protein features for AlphaFold calculation
        from protein BioPandas dataframe
        no MSA / template based
        """
        sequence = residue_constants.aatype_to_str_sequence(target_protein.aatype)
        features = {
            "template_all_atom_positions": target_protein.atom_positions[None, ...],
            "template_all_atom_mask": target_protein.atom_mask[None, ...],
            "template_sequence": [sequence],
            "template_aatype": target_protein.aatype[None, ...],
            "template_domain_names": [None],  # f''.encode()]
        }
        num_templates = features["template_aatype"].shape[0]
        """ look for more elegant way to calculate sequence features """
        sequence_features = {}
        num_res = len(sequence)
        sequence_features["aatype"] = residue_constants.sequence_to_onehot(
            sequence=sequence,
            mapping=residue_constants.restype_order_with_x,
            map_unknown_to_x=True,
        )
        sequence_features["between_segment_residues"] = np.zeros(
            (num_res,), dtype=np.int32
        )
        sequence_features["domain_name"] = np.array(
            ["input".encode("utf-8")], dtype=np.object_
        )
        sequence_features["residue_index"] = np.array(
            target_protein.residue_index, dtype=np.int32
        )
        sequence_features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
        sequence_features["sequence"] = np.array(
            [sequence.encode("utf-8")], dtype=np.object_
        )
        deletion_matrix = np.zeros(num_res)
        sequence_features["deletion_matrix_int"] = np.array(
            deletion_matrix, dtype=np.int32
        )[None, ...]
        int_msa = [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
        sequence_features["msa"] = np.array(int_msa, dtype=np.int32)[None, ...]
        sequence_features["num_alignments"] = np.array(np.ones(num_res), dtype=np.int32)
        sequence_features["msa_species_identifiers"] = np.array(["".encode()])
        feature_dict = {**sequence_features, **features}
        return feature_dict

    def init_alphafold(
        self,
        weights_path="/Users/ivanisenko/projects/ProteinAIDesign/CGMover/alphafold/params_model_2_ptm.npz",
    ):
        """get config, prepare feature_processor and load model"""
        self.config = model_config("model_2_ptm", low_prec=(True))
        self.feature_processor = feature_pipeline.FeaturePipeline(self.config.data)
        model_generator = load_models_from_command_line(
            self.config, self.device, None, weights_path, None
        )
        for model, output_directory in model_generator:
            self.model = model
            output_directory = output_directory
        self.model.to(self.device)

    def get_batch_features(self, af_output_batch, af_input_features, batch_id):
        """
        Input: output of alphafold
               features for alphafold inference
               batch_id
        Output:
               output and features for batch_id
        """
        out = {"sm": {}}
        for k, out_ in af_output_batch.items():
            if k == "sm":
                for name in out_:
                    if out_[name].shape[0] == 1:
                        out["sm"][name] = out_[name][batch_id, ...]
                    if out_[name].shape[0] == 8:
                        out["sm"][name] = out_[name][:, batch_id, ...]
                continue
            if len(out_.shape) == 0:
                out[k] = out_
                continue
            out[k] = out_[batch_id, ...]
        features = tensor_tree_map(
            lambda x: np.array(x[batch_id, ..., -1].cpu()), af_input_features
        )

        return out, features

    def get_ppi_metrics(self, protein, of_output, chain_id):
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

    def inference_monomer(self, pdb_df, n_recycle=2):#, template_mask=True, side_chain_mask=False, return_all_cycles=False):
        """inference openfold with pdb dataframe input
         0 cycle with template features
         1 cycle masked template features
         2 cycle masked template
         ...
         n_recycle
        ...
        n cycle masked template features

        return alphafold output and predicted pdb structure
        return_all_cycles - return embeddings for each cycle
        """
        pdb = protein.from_pdb_df(pdb_df)
        resi = pdb.residue_index
        features = self.prepare_features(pdb)
        pdb_df_ca = pdb_df[pdb_df["atom_name"] == "CA"].reset_index()
        processed_feature_dict = self.feature_processor.process_features(
            features,
            mode="predict",
        )
        """ add recycling features with masked template features """
        processed_feature_dict_list = [
            processed_feature_dict
        ]
        for i in range(n_recycle):
            processed_feature_dict_list.append(
                {k: p.detach().clone() for k, p in processed_feature_dict.items()}
            )
            processed_feature_dict_list[-1]["template_mask"] *= 0

        cat_fn = partial(torch.cat, dim=-1)
        processed_feature_dict = dict_multimap(cat_fn, processed_feature_dict_list)

        for c, p in processed_feature_dict.items():
            if p.dtype == torch.float64:
                processed_feature_dict[c] = torch.as_tensor(
                    p, dtype=torch.float32, device=self.device
                )
            else:
                processed_feature_dict[c] = torch.as_tensor(p, device=self.device)
            processed_feature_dict[c] = processed_feature_dict[c][None, ...]

        """ load alphafold model """
        """ the only modification in the openfold is return per cycle"""
        with torch.no_grad():
            out_batch, out_per_cycle = self.model(processed_feature_dict)

        for i in range(len(out_per_cycle)):
            out_per_cycle[i] = tensor_tree_map(
                lambda x: np.array(x[...].detach().cpu()), out_per_cycle[i]
            )
            out_per_cycle[i], ifd = self.get_batch_features(
                out_per_cycle[i], processed_feature_dict, 0
            )
            unrelaxed_protein = prep_output(
                out_per_cycle[i],
                ifd,
                ifd,
                self.feature_processor,
                "model_2_ptm",
                200,
                False,
            )
            pdb_str = protein.to_pdb(unrelaxed_protein)
            pdb_df_pred = pdb_str_to_dataframe(pdb_str, pdb_df)
            out_per_cycle[i]["pdb"] = pdb_df_pred

        out_batch_ = tensor_tree_map(
            lambda x: np.array(x[...].detach().cpu()), out_batch
        )
        out, ifd = self.get_batch_features(out_batch_, processed_feature_dict, 0)
        unrelaxed_protein = prep_output(
            out, ifd, ifd, self.feature_processor, "model_2_ptm", 200, False
        )
        pdb_str = protein.to_pdb(unrelaxed_protein)
        pdb_df_pred = pdb_str_to_dataframe(pdb_str, pdb_df)
        return out, out_per_cycle, pdb_df_pred



if __name__ == "__main__":
    pdb_df = load_protein("../tests/test_dimer.pdb")
    ofr = OpenFoldWraper()
    of_output, pdb_df_pred = ofr.inference_monomer(pdb_df)
    print(ofr.get_structure_metrics(of_output))
    print(
        ofr.get_structure_metrics_by_interface(of_output, pdb_df, interface_chain="A")
    )
    save_pdb(pdb_df_pred, "../tests/test.pdb")
