"""
Code includes class for the run of pyrosetta for different PPI tasks
"""
import sys
import pickle
import shutil
from pathlib import Path
#import pyrosetta
import itertools
import numpy as np

"""
ex3/ex4 might be added to improve performance
flip_HNQ - is arguable parameter, better to investigate how it changes the perofrmance
"""
#pyrosetta.init("-ignore_unrecognized_res 1 -ex1 -ex2 -flip_HNQ")

### very bad standart to import *
### should be fixed in later editions
from pyrosetta.teaching import *
from pyrosetta import *
import os, sys, urllib
from rosetta.protocols.minimization_packing import *
from pyrosetta.rosetta.core.chemical import aa_from_oneletter_code
from PPInterface.protein_utils import load_protein, save_pdb, get_sequence, mutate_protein
from scipy.spatial import distance


class RosettaWrapper:
    def __init__(self):
        """
        define rosetta scoring function
        """
        self.scorefxn = pyrosetta.create_score_function("ref2015_cart")
        pass

    def load_pose(self, path):
        """
        Load rosetta pose from the path
        :param path:
        :return:
        """
        return pyrosetta.pose_from_pdb(path)

    def relax_pose(self, input_pdb_path, output_pdb_path, max_iter=100, constrain_relax_to_native_coords=True):
        """
        run pose relaxation
        :param input_pdb_path:
        :param output_pdb_path:
        :param max_iter:
        :return:
        """

        pose = pyrosetta.pose_from_pdb(input_pdb_path)
        movemap = pyrosetta.rosetta.core.kinematics.MoveMap()
        movemap.set_bb(True)
        movemap.set_chi(True)

        relax = pyrosetta.rosetta.protocols.relax.FastRelax()
        ### add constraints to native coords because we don't want to go far from X-rayy
        relax.constrain_relax_to_native_coords(constrain_relax_to_native_coords)
        ### e.g. we work with PPI and incorrect tree can be an issue, so we do cartesian optimization
        relax.cartesian(True)
        relax.set_scorefxn(self.scorefxn)
        relax.set_movemap(movemap)
        relax.max_iter(max_iter)
        relax.apply(pose)
        pose.dump_pdb(output_pdb_path)

    def repack_pose(self, input_pdb_path, output_pdb_path, max_iter=100):
        """
        Function to repack side chains, with minimzation afterwards
        :param input_pdb_path:
        :param output_pdb_path:
        :param max_iter:
        :return:
        """
        pose = pyrosetta.pose_from_pdb(input_pdb_path)
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(
            pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())  # should be operation.PreventRepacking()?
        pack_task = tf.create_task_and_apply_taskoperations(pose)
        packer = PackRotamersMover(self.scorefxn, pack_task)
        packer.apply(pose)

        mm = MoveMap()
        mm.set_bb(True)
        mm.set_chi(True)
        min_mover = rosetta.protocols.minimization_packing.MinMover()
        min_mover.movemap(mm)
        min_mover.score_function(self.scorefxn)
        min_mover.min_type("lbfgs_armijo")
        min_mover.tolerance(1e-6)
        min_mover.apply(pose)

        pose.dump_pdb(output_pdb_path)
        return pose

    def mutate_task_interface(self, pose, mutant_codes, chains_1="A", chains_2="B", R=10.0):
        """
        Function to mutate and repack amino acid residues on the PPI interface

        :param pose:
        :param mutant_codes: mutants in format {(aa_before,residue_number,chain_id):aa_after ...
        e.g. {"R",222,"QNK"} would run fastdesign to mutate to Q or N or K
        :param chains_1:
        :param chains_2:
        :param R: cut-off for PPI residues
        :return:
        """
        prevent_in = pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT()
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.PreventRepacking())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

        ### select interacting chains
        chain_A = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(chains_1)
        chain_B = pyrosetta.rosetta.core.select.residue_selector.ChainSelector(chains_2)

        packer_task_design = tf.create_task_and_apply_taskoperations(pose)
        ### select residues close to chain A
        interface_selector_1 = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(chain_A, R,
                                                                                                          True)
        ### select residues close to chain B
        interface_selector_2 = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector(chain_B, R,
                                                                                                          True)
        ### select only interface
        ia = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(chain_A)
        ib = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(chain_B)
        ia = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(interface_selector_1, ia)
        ib = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(interface_selector_2, ib)
        interface_selector = pyrosetta.rosetta.core.select.residue_selector.OrResidueSelector(ia, ib)

        ### only repack non design residues (later)
        prevent_in = pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT()
        ### not repack distant residues (later)
        prevent_off = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()

        ### select designable interface residues
        not_interface = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(interface_selector)
        design_resi = [f"{r}{chain}" for _, r, chain in mutant_codes]  # [:-1]
        design_resi = rosetta.core.select.residue_selector.ResidueIndexSelector(",".join(design_resi))
        designable_interface = design_resi

        ### select тщт-designable interface residues
        not_designable_interface = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(design_resi)
        not_designable_interface = pyrosetta.rosetta.core.select.residue_selector.AndResidueSelector(interface_selector,
                                                                                                     not_designable_interface)

        ### apply repack tasks for non designable interface
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_in,
                                                                                       not_designable_interface,
                                                                                       False))

        ### prevent repacking interface not at the interface
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_off,
                                                                                       not_interface,
                                                                                       False))

        ###apply these tasks to the pose
        packer_task_design = tf.create_task_and_apply_taskoperations(pose)

        ###set mutations tasks according to mutant_codes
        pdb2pose_resi = pyrosetta.rosetta.core.pose.get_pdb2pose_numbering_as_stdmap(pose)
        for (_, resi, chain), aa in mutant_codes.items():
            pose_resi = pdb2pose_resi[chain + str(resi) + "."]
            if aa == "*":
                continue
            aa = list(aa)
            mutant_aa = [int(aa_from_oneletter_code(aa_)) for aa_ in aa]
            aa_bool = pyrosetta.Vector1([aa_ in mutant_aa for aa_ in range(1, 21)])
            packer_task_design.nonconst_residue_task(pose_resi).restrict_absent_canonical_aas(aa_bool)
        return packer_task_design

    def mutate_interface_and_repack(self, pose, mutant_codes, chains_1="C", chains_2="A,B", R=12.0):
        """
        mutate interface residues and repack side chains / no minimization
        :param self:
        :param pose:
        :param mutant_codes: mutants in format {(aa_before,residue_number,chain_id):aa_after ...
        e.g. {"R",222,"QNK"} would run fastdesign to mutate to Q or N or K
        :param chains_1:
        :param chains_2:
        :param R:
        :return:
        """
        m_task = self.mutate_task_interface(pose, mutant_codes, chains_1=chains_1, chains_2=chains_2, R=R)
        packer = PackRotamersMover(self.scorefxn, m_task)
        packer.apply(pose)
        ddg = self.calc_ddg(pose, chain_id=chains_1)
        return ddg

    def calc_monomer(self, pose):
        """
        calculate ddg of chain_id for the PPI
        :param self:
        :param pose:
        :param chain_id:
        :return:
        """
        ### score complex and subunits and calculate ddg
        m_task = self.mutate_task_interface(pose, mutant_codes, chains_1=chains_1, chains_2=chains_2, R=R)
        packer = PackRotamersMover(self.scorefxn, m_task)
        packer.apply(pose)
        ea = self.scorefxn(pose_a)  ### change in scores of monomer
        ddg = self.scorefxn(pose) - ea - self.scorefxn(pose_b)  ###change in score of PPI
        return ddg, ea

    def calc_ddg(self, pose, chain_id="A"):
        """
        calculate ddg of chain_id for the PPI
        :param self:
        :param pose:
        :param chain_id:
        :return:
        """
        pose_a = None
        pose_b = None

        ### extract list of chains within the pose
        chains = []
        for i in range(1, pose.size() + 1):
            c = pose.pdb_info().chain(i)
            if c in chains:
                continue
            chains.append(c)

        c = [0] + list(pyrosetta.rosetta.core.pose.chain_end_res(pose))

        ### split PPI pose complex to two poses with chain_id and without
        pose_chains = []
        for i in range(1, len(c)):
            pose_chains.append(pyrosetta.rosetta.protocols.grafting.return_region(pose, c[i - 1] + 1, c[i]))

        for i in range(len(pose_chains)):
            if chains[i] == chain_id:
                pose_a = pose_chains[i]
            else:
                if pose_b is None:
                    pose_b = pose_chains[i]
                else:
                    pose_b.append_pose_by_jump(pose_chains[i], pose_b.num_jump() + 1)

        ### score complex and subunits and calculate ddg
        ea = self.scorefxn(pose_a) ### change in scores of monomer
        ddg = self.scorefxn(pose) - ea - self.scorefxn(pose_b) ###change in score of PPI
        return ddg, ea

def main():
    pass

if __name__ == "__main__":
    main()
