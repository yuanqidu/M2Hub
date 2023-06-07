"""
Creates LMDB files with extracted graph features.
"""

import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
import sys

import ase.io
import lmdb
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

from m2models.preprocessing import AtomsToGraphs, AtomsToPeriodicGraphs

from torch_geometric.data import Data

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from rdkit import Chem
from sklearn.utils import shuffle



def write_images_to_lmdb_oc(mp_arg):
    a2g, db_path, samples, sampled_ids, idx, pid, args = mp_arg
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=5000 * len(samples),
        position=pid,
        desc="Preprocessing data into LMDBs",
    )
    for sample in samples:
        if args.task in ["s2ef", 'is2re']:
            traj_logs = open(sample, "r").read().splitlines()
            xyz_idx = os.path.splitext(os.path.basename(sample))[0]
            traj_path = os.path.join(args.data_path, f"{xyz_idx}.extxyz")
            traj_frames = ase.io.read(traj_path, ":")

            for i, frame in enumerate(traj_frames):
                frame_log = traj_logs[i].split(",")
                sid = int(frame_log[0].split("random")[1])
                fid = int(frame_log[1].split("frame")[1])
                data_object = a2g.convert(frame)
                # add atom tags
                data_object.tags = torch.LongTensor(frame.get_tags())
                data_object.sid = sid
                data_object.fid = fid
                # subtract off reference energy
                if args.ref_energy and not args.test_data:
                    ref_energy = float(frame_log[2])
                    data_object.y -= ref_energy

                txn = db.begin(write=True)
                txn.put(
                    f"{idx}".encode("ascii"),
                    pickle.dumps(data_object, protocol=-1),
                )
                txn.commit()
                idx += 1
                sampled_ids.append(",".join(frame_log[:2]) + "\n")
                pbar.update(1)


    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(idx, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return sampled_ids, idx

def write_images_to_lmdb(a2g, db_path, args, x, y, indices):
    db = lmdb.open(
        db_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    pbar = tqdm(
        total=len(indices),
        desc="Preprocessing data into LMDBs",
    )

    for i, idx in enumerate(indices):
        frame = x[idx]
        data_object = frame
        if len(y) != 0:
            label = y[idx]
            data_object.y = label
        txn = db.begin(write=True)
        txn.put(
            f"{i}".encode("ascii"),
            pickle.dumps(data_object, protocol=-1),
        )
        txn.commit()
        pbar.update(1)


    # Save count of objects in lmdb.
    txn = db.begin(write=True)
    txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
    txn.commit()

    db.sync()
    db.close()

    return True


def main(args):
    if args.task in ["s2ef", 'is2re']:
        xyz_logs = glob.glob(os.path.join(args.data_path, "*.txt"))
        if not xyz_logs:
            raise RuntimeError("No *.txt files found. Did you uncompress?")
        if args.num_workers > len(xyz_logs):
            args.num_workers = len(xyz_logs)

        # Initialize feature extractor.
        a2g = AtomsToPeriodicGraphs(
            max_neigh=50,
            radius=6,
            r_energy=not args.test_data,
            r_forces=not args.test_data,
            r_fixed=True,
            r_distances=False,
            r_edges=args.get_edges,
        )

        # Create output directory if it doesn't exist.
        os.makedirs(os.path.join(args.out_path), exist_ok=True)

        # Initialize lmdb paths
        db_paths = [
            os.path.join(args.out_path, "data.%04d.lmdb" % i)
            for i in range(args.num_workers)
        ]

        # Chunk the trajectories into args.num_workers splits
        chunked_txt_files = np.array_split(xyz_logs, args.num_workers)

        # Extract features
        sampled_ids, idx = [[]] * args.num_workers, [0] * args.num_workers

        pool = mp.Pool(args.num_workers)
        mp_args = [
            (
                a2g,
                db_paths[i],
                chunked_txt_files[i],
                sampled_ids[i],
                idx[i],
                i,
                args,
            )
            for i in range(args.num_workers)
        ]
        op = list(zip(*pool.imap(write_images_to_lmdb_oc, mp_args)))
        sampled_ids, idx = list(op[0]), list(op[1])

        # Log sampled image, trajectory trace
        for j, i in enumerate(range(args.num_workers)):
            ids_log = open(
                os.path.join(args.out_path, "data_log.%04d.txt" % i), "w"
            )
            ids_log.writelines(sampled_ids[j])
    else:
        if args.task not in ['qm9', 'tmqm']:
            a2g = AtomsToPeriodicGraphs(
                max_neigh=50,
                radius=6,
                r_energy=False,
                r_forces=False,
                r_fixed=False,
                r_distances=False,
                r_edges=args.get_edges,
            )
        else:
            a2g = AtomsToGraphs(
                radius=6,
            )
        system_indices, ele_count = {}, {}
        if args.task == 'mp':
            raw_data = pd.read_csv(args.data_path)
            
            x, y = [], []
            ids = []
            idx = 0
            for i in tqdm(range(raw_data.shape[0])):
                frame = raw_data.iloc[i]
                ids.append(raw_data.iloc[i]['material_id']) 
                if frame[args.property] == '':
                    continue
                else:
                    sample = Structure.from_str(frame['cif'], fmt='cif')
                    sample = AseAtomsAdaptor.get_atoms(sample)
                    chemform = sample.get_chemical_formula(mode='hill')
                    chemsys = ''.join([char for char in chemform if not char.isdigit()])
                    ele_ct = 0
                    for char in chemform:
                        if char.isupper():
                            ele_ct += 1
                    if ele_ct in ele_count:
                        if chemsys not in ele_count[ele_ct]:
                            ele_count[ele_ct].append(chemsys)
                    else:
                        ele_count[ele_ct] = [chemsys]
                    if chemsys in system_indices:
                        if chemform in system_indices[chemsys]:
                            system_indices[chemsys][chemform].append(idx)
                        else:
                            system_indices[chemsys][chemform] = [idx]
                    else:
                        system_indices[chemsys] = {chemform: [idx]}
                    x.append(a2g.convert(sample))
                    y.append(frame[args.property])
                    idx += 1

        elif args.task == 'omdb':
            raw_data = read(args.data_path, index=':')
            label = pd.read_csv(args.data_path.replace('xyz', 'csv'), header=None)
            x, y = [], []
            for idx in range(len(raw_data)):
                sample = raw_data[idx]
                chemform = sample.get_chemical_formula(mode='hill')
                chemsys = ''.join([char for char in chemform if not char.isdigit()])
                ele_ct = 0
                for char in chemform:
                    if char.isupper():
                        ele_ct += 1
                if ele_ct in ele_count:
                    if chemsys not in ele_count[ele_ct]:
                        ele_count[ele_ct].append(chemsys)
                else:
                    ele_count[ele_ct] = [chemsys]
                if chemsys in system_indices:
                    if chemform in system_indices[chemsys]:
                        system_indices[chemsys][chemform].append(idx)
                    else:
                        system_indices[chemsys][chemform] = [idx]
                else:
                    system_indices[chemsys] = {chemform: [idx]}
                x.append(a2g.convert(sample))
                y.append(label.iloc[idx][0])

            total_len = 0
            for name in system_indices:
                for form in system_indices[name]:
                    total_len += len(system_indices[name][form])

            total_len = 0
            for ele_ct in ele_count:
                total_len += len(ele_count[ele_ct])
        
        elif args.task == 'matbench':
            from matbench.bench import MatbenchBenchmark
            mb = MatbenchBenchmark(autoload=True)
            dataset = mb.tasks_map[args.property]
            x, y = [], []
            raw_data = dataset.df['structure']
            label = dataset.df.iloc[:, -1]

            for idx, (sample, label) in enumerate(zip(raw_data, label)):
                sample = Structure.from_sites(sample)
                sample = AseAtomsAdaptor.get_atoms(sample)
                chemform = sample.get_chemical_formula(mode='hill')
                chemsys = ''.join([char for char in chemform if not char.isdigit()])
                ele_ct = 0
                for char in chemform:
                    if char.isupper():
                        ele_ct += 1
                if ele_ct in ele_count:
                    if chemsys not in ele_count[ele_ct]:
                        ele_count[ele_ct].append(chemsys)
                else:
                    ele_count[ele_ct] = [chemsys]
                if chemsys in system_indices:
                    if chemform in system_indices[chemsys]:
                        system_indices[chemsys][chemform].append(idx)
                    else:
                        system_indices[chemsys][chemform] = [idx]
                else:
                    system_indices[chemsys] = {chemform: [idx]}
                x.append(a2g.convert(sample))
                y.append(label)
            
            total_len = 0
            for name in system_indices:
                for form in system_indices[name]:
                    total_len += len(system_indices[name][form])

            total_len = 0
            for ele_ct in ele_count:
                total_len += len(ele_count[ele_ct])

        elif args.task == 'jarvis':
            from jarvis.db.figshare import data as jdata
            from jarvis.core.atoms import Atoms

            subset, property = args.property.split(':')

            raw_data = jdata(subset)
            x, y = [], []
            idx = 0
            for sample in enumerate(raw_data):
                sample = sample[1]
                atoms = Atoms.from_dict(sample["atoms"])
                target = sample[property]
                if target != "na":
                    atoms.write_cif(filename="atoms.cif")
                    sample = Structure.from_file("atoms.cif")
                    sample = AseAtomsAdaptor.get_atoms(sample)
                    chemform = sample.get_chemical_formula(mode='hill')
                    chemsys = ''.join([char for char in chemform if not char.isdigit()])
                    ele_ct = 0
                    for char in chemform:
                        if char.isupper():
                            ele_ct += 1
                    if ele_ct in ele_count:
                        if chemsys not in ele_count[ele_ct]:
                            ele_count[ele_ct].append(chemsys)
                    else:
                        ele_count[ele_ct] = [chemsys]
                    if chemsys in system_indices:
                        if chemform in system_indices[chemsys]:
                            system_indices[chemsys][chemform].append(idx)
                        else:
                            system_indices[chemsys][chemform] = [idx]
                    else:
                        system_indices[chemsys] = {chemform: [idx]}
                    x.append(a2g.convert(sample))
                    y.append(torch.tensor(target).unsqueeze(dim=0))
                    idx += 1
            
            total_len = 0
            for name in system_indices:
                for form in system_indices[name]:
                    total_len += len(system_indices[name][form])

            total_len = 0
            for ele_ct in ele_count:
                total_len += len(ele_count[ele_ct])

        # molecules w/o peridocitiy
        elif args.task == 'tmqm':
            if args.split != 'random':
                raise ValueError(
                    f"{args.split} split not supported for {args.task} dataset"
                )
            try:
                import openbabel 
            except:
                raise ValueError(
                    f"openbabel needs to be installed to process {args.task}"
                )

            
            raw_datain = open(args.data_path, 'r')
            lines = []
            raw_data = []
            for line in raw_datain:
                lines.append(line)
                if line == '\n':
                    raw_data.append("".join(lines))
                    lines = []
            mols = []
            for raw_sample in raw_data:
                raw_dataout = open('temp.xyz', 'w+')
                raw_dataout.write(raw_sample)
                raw_dataout.close()
                os.system('obabel temp.xyz -O temp.sdf')
                mol = Chem.SDMolSupplier('temp.sdf', sanitize=False)[0]
                mols.append(mol)
            raw_data = mols
            label = pd.read_csv(args.data_path.replace('xyz', 'csv'), delimiter=';')
            label = label[args.property]
            x, y = [], []
            for idx in range(label.shape[0]):
                sample = raw_data[idx]
                x.append(a2g.convert(sample))
                y.append(label.iloc[idx])
        
        # molecules w/o peridocitiy
        elif args.task == 'qm9':
            raw_data = np.load(os.path.join(args.data_path))
            train_frac, val_frac = 110000, 10000
            total_len = len(raw_data['N'])
            indices = shuffle(range(total_len), random_state=42)

            label = raw_data[args.property]

            R = raw_data['R']
            Z = raw_data['Z']
            N = raw_data['N']
            split = np.cumsum(N)
            R_qm9 = np.split(R, split)
            Z_qm9 = np.split(Z,split)

            x, y = [], []
            for i in tqdm(range(len(N))):
                R_i = torch.tensor(R_qm9[i],dtype=torch.float32)
                z_i = torch.tensor(Z_qm9[i],dtype=torch.int64)
                data = Data(pos=R_i, z=z_i)

                x.append(data)
                y.append(label[i])

            train_idx, val_idx, test_idx = (
                indices[: train_frac],
                indices[train_frac : train_frac + val_frac],
                indices[train_frac + val_frac :],
            )
        
        elif args.task == 'perov5' or args.task == 'carbon24':
            train_path = args.data_path.replace(f'{args.task}.csv', f'{args.task}_train.csv')
            val_path = args.data_path.replace(f'{args.task}.csv', f'{args.task}_val.csv')
            test_path = args.data_path.replace(f'{args.task}.csv', f'{args.task}_test.csv')
            train_data = pd.read_csv(train_path)
            val_data = pd.read_csv(val_path)
            test_data = pd.read_csv(test_path)

            idx = 0
            x, y = [], []
            train_idx, val_idx, test_idx = [], [], []
            for i in tqdm(range(train_data.shape[0])):
                frame = train_data.iloc[i]
                sample = Structure.from_str(frame['cif'], fmt='cif')
                sample = AseAtomsAdaptor.get_atoms(sample)
                x.append(a2g.convert(sample))
                train_idx.append(idx)
                idx += 1
            for i in tqdm(range(val_data.shape[0])):
                frame = val_data.iloc[i]
                sample = Structure.from_str(frame['cif'], fmt='cif')
                sample = AseAtomsAdaptor.get_atoms(sample)
                x.append(a2g.convert(sample))
                val_idx.append(idx)
                idx += 1
            for i in tqdm(range(test_data.shape[0])):
                frame = test_data.iloc[i]
                sample = Structure.from_str(frame['cif'], fmt='cif')
                sample = AseAtomsAdaptor.get_atoms(sample)
                x.append(a2g.convert(sample))
                test_idx.append(idx)
                idx += 1

        else:
            raise ValueError(
                    f"{args.task} task not supported"
                )


        if args.split == 'random':
            if args.task not in ['qm9', 'carbon24', 'perov5']:
                train_frac, val_frac, test_frac = args.data_frac
                total_len = len(x)
                indices = np.random.permutation(total_len)
                train_idx, val_idx, test_idx = (
                    indices[: int(total_len * train_frac)],
                    indices[int(total_len * train_frac) : int(total_len * (train_frac + val_frac))],
                    indices[int(total_len * (train_frac + val_frac)) :],
                )
        
        elif args.split == 'composition':
            if args.task not in ['qm9', 'carbon24', 'perov5']:
                train_frac, val_frac, test_frac = args.data_frac
                train_idx, val_idx, test_idx = [], [], []
                for name in system_indices:
                    length = len(system_indices[name])
                    # smaller than 3, all in train
                    if length < 3:
                        chemform = list(system_indices[name].keys())
                        for chemname in chemform:
                            train_idx.extend(system_indices[name][chemname])
                    # 3 or 4, 1 in test, 1 in valid, rest in train
                    elif length >= 3 and length < 5:
                        chemform = list(system_indices[name].keys())
                        indices = np.random.permutation(len(chemform))
                        train_indices, val_indices, test_indices = (
                            indices[2:],
                            indices[1],
                            indices[0],
                        )
                        for idx, chemname in enumerate(chemform):
                            if idx in train_indices:
                                train_idx.extend(system_indices[name][chemname])
                            elif idx == val_indices:
                                val_idx.extend(system_indices[name][chemname])
                            elif idx == test_indices:
                                test_idx.extend(system_indices[name][chemname])
                    elif length >= 5:
                        chemform = list(system_indices[name].keys())
                        indices = np.random.permutation(len(chemform))
                        total_len = len(indices)
                        train_indices, val_indices, test_indices = (
                            indices[: int(total_len * train_frac)],
                            indices[int(total_len * train_frac) : int(total_len * (train_frac + val_frac))],
                            indices[int(total_len * (train_frac + val_frac)) :],
                        )
                        for idx, chemname in enumerate(chemform):
                            if idx in train_indices:
                                train_idx.extend(system_indices[name][chemname])
                            elif idx in val_indices:
                                val_idx.extend(system_indices[name][chemname])
                            elif idx in test_indices:
                                test_idx.extend(system_indices[name][chemname])

        elif args.split == 'system':
            if args.task not in ['qm9', 'carbon24', 'perov5']:
                train_frac, val_frac, test_frac = args.data_frac
                system_size = list(ele_count.keys())
                system_size.sort(reverse=False)
                total_len = len(x)
                train_idx, val_idx, test_idx = [], [], []
                n_total_train = train_frac * total_len
                n_total_valid = (val_frac + train_frac) * total_len
                is_train, is_valid = True, True
                for count_size in system_size:
                    sys_name = ele_count[count_size]
                    print (count_size, is_train, is_valid)
                    for chemname in sys_name:
                        subcount = 0
                        for sys_formula in system_indices[chemname]:
                            subcount += len(system_indices[chemname][sys_formula])
                        if len(train_idx) + subcount > n_total_train and is_train:
                            is_train = False 
                        if len(val_idx) + subcount > n_total_valid - n_total_train and is_valid:
                            is_valid = False 
                        if len(train_idx) < n_total_train and is_train:
                            for sys_formula in system_indices[chemname]:
                                train_idx.extend(system_indices[chemname][sys_formula])
                        elif len(val_idx) < n_total_valid - n_total_train and is_valid:
                            for sys_formula in system_indices[chemname]:
                                val_idx.extend(system_indices[chemname][sys_formula])
                        else:
                            for sys_formula in system_indices[chemname]:
                                test_idx.extend(system_indices[chemname][sys_formula])


        elif args.split == 'time':
            if args.task != 'mp':
                raise ValueError(
                        f"only mp is supported for {args.split} split"
                    )
            else:
                # year already sorted
                train_frac, val_frac, test_frac = args.data_frac
                total_len = len(x)
                indices = [idx for idx in range(total_len)]
                train_idx, val_idx, test_idx = (
                    indices[: int(total_len * train_frac)],
                    indices[int(total_len * train_frac) : int(total_len * (train_frac + val_frac))],
                    indices[int(total_len * (train_frac + val_frac)) :],
                )

        else:
            raise ValueError(
                    f"{args.split} split not supported"
                )
        
        # Create output directory if it doesn't exist.
        os.makedirs(os.path.join(args.out_path, f'{args.split}_train'), exist_ok=True)

        # Initialize lmdb paths
        db_path = os.path.join(args.out_path, f'{args.split}_train', "data.lmdb")

        write_images_to_lmdb(a2g, db_path, args, x, y, train_idx)

        # Create output directory if it doesn't exist.
        os.makedirs(os.path.join(args.out_path, f'{args.split}_valid'), exist_ok=True)

        # Initialize lmdb paths
        db_path = os.path.join(args.out_path, f'{args.split}_valid', "data.lmdb")

        write_images_to_lmdb(a2g, db_path, args, x, y, val_idx)

        # Create output directory if it doesn't exist.
        os.makedirs(os.path.join(args.out_path, f'{args.split}_test'), exist_ok=True)

        # Initialize lmdb paths
        db_path = os.path.join(args.out_path, f'{args.split}_test', "data.lmdb")

        write_images_to_lmdb(a2g, db_path, args, x, y, test_idx)
            



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        help="Path to dir containing *.extxyz and *.txt files",
    )
    parser.add_argument(
        "--out-path",
        help="Directory to save extracted features. Will create if doesn't exist",
    )
    parser.add_argument(
        "--get-edges",
        action="store_true",
        help="Store edge indices in LMDB, ~10x storage requirement. Default: compute edge indices on-the-fly.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="No. of feature-extracting processes or no. of dataset chunks",
    )
    parser.add_argument(
        "--ref-energy", action="store_true", help="Subtract reference energies"
    )
    parser.add_argument(
        "--test-data",
        action="store_true",
        help="Is data being processed test data?",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
