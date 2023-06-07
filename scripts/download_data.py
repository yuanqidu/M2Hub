import argparse
import glob
import logging
import os
import numpy as np

import m2models

from torch_geometric.data import download_url

"""
This script provides users with an automated way to download, preprocess (where
applicable), and organize data to readily be used by the existing config files.
"""

DOWNLOAD_LINKS = {
    "s2ef": {
        "200k": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar",
        "2M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_2M.tar",
        "20M": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_20M.tar",
        "all": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_all.tar",
        "val_id": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar",
        "val_ood_ads": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_ads.tar",
        "val_ood_cat": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_cat.tar",
        "val_ood_both": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_ood_both.tar",
        "test": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_test_lmdbs.tar.gz",
        "rattled": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_rattled.tar",
        "md": "https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_md.tar",
    },
    "is2re": "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz",
    "mp": "https://figshare.com/ndownloader/files/40771040",
    "omdb": "https://figshare.com/ndownloader/files/40064083",
    "omdb_label": "https://figshare.com/ndownloader/files/40064086",
    "tmqm": "https://figshare.com/ndownloader/files/40222432",
    "tmqm_label": "https://figshare.com/ndownloader/files/40104436",
    "qm9": "https://github.com/klicperajo/dimenet/raw/master/data/qm9_eV.npz",
    "carbon24_train": "https://figshare.com/ndownloader/files/40323517",
    "carbon24_val": "https://figshare.com/ndownloader/files/40323520",
    "carbon24_test": "https://figshare.com/ndownloader/files/40323514",
    "perov5_train": "https://figshare.com/ndownloader/files/40323508",
    "perov5_val": "https://figshare.com/ndownloader/files/40323511",
    "perov5_test": "https://figshare.com/ndownloader/files/40323505",
}

TASK_TYPE = {
    'mp': 'csv',
    'omdb': 'xyz',
    'omdb_label': 'csv',
    'tmqm': 'xyz',
    'tmqm_label': 'csv',
    'qm9': 'npz',
    'perov5': 'csv',
    'carbon24': 'csv',
}

S2EF_COUNTS = {
    "s2ef": {
        "200k": 200000,
        "2M": 2000000,
        "20M": 20000000,
        "all": 133934018,
        "val_id": 999866,
        "val_ood_ads": 999838,
        "val_ood_cat": 999809,
        "val_ood_both": 999944,
        "rattled": 16677031,
        "md": 38315405,
    },
}


def get_data(datadir, task, property, split, seed, data_frac, del_intmd_files):
    os.makedirs(datadir, exist_ok=True)
    os.makedirs(f'{datadir}/{task}', exist_ok=True)

    if task == "s2ef" and split is None:
        raise NotImplementedError("S2EF requires a split to be defined.")

    if task == "s2ef":
        assert (
            split in DOWNLOAD_LINKS[task]
        ), f'S2EF "{split}" split not defined, please specify one of the following: {list(DOWNLOAD_LINKS["s2ef"].keys())}'
        download_link = DOWNLOAD_LINKS[task][split]

    elif task == "is2re":
        download_link = DOWNLOAD_LINKS[task]

    elif task == 'mp':
        download_link = DOWNLOAD_LINKS[task]

    elif task == 'omdb':
        download_link = DOWNLOAD_LINKS[task]
        label_download_link = DOWNLOAD_LINKS[f'{task}_label']

    elif task == 'tmqm':
        download_link = DOWNLOAD_LINKS[task]
        label_download_link = DOWNLOAD_LINKS[f'{task}_label']

    elif task == 'qm9':
        download_link = DOWNLOAD_LINKS[task]

    elif task == 'carbon24':
        download_link_train = DOWNLOAD_LINKS[f'{task}_train']
        download_link_val = DOWNLOAD_LINKS[f'{task}_val']
        download_link_test = DOWNLOAD_LINKS[f'{task}_test']
    
    elif task == 'perov5':
        download_link_train = DOWNLOAD_LINKS[f'{task}_train']
        download_link_val = DOWNLOAD_LINKS[f'{task}_val']
        download_link_test = DOWNLOAD_LINKS[f'{task}_test']

    if task in ["s2ef", 'is2re']:
        os.system(f"wget {download_link} -P {datadir}")
        filename = os.path.join(datadir, os.path.basename(download_link))
        logging.info("Extracting contents...")
        os.system(f"tar -xvf {filename} -C {datadir}")
        dirname = os.path.join(
            datadir,
            os.path.basename(filename).split(".")[0],
        )
        if task == "s2ef" and split != "test":
            compressed_dir = os.path.join(dirname, os.path.basename(dirname))
            if split in ["200k", "2M", "20M", "all", "rattled", "md"]:
                output_path = os.path.join(datadir, task, split, "train")
            else:
                output_path = os.path.join(datadir, task, "all", split)
            uncompressed_dir = uncompress_data(compressed_dir)
            preprocess_data(uncompressed_dir, output_path, task)

            verify_count(output_path, task, split)
        if task == "s2ef" and split == "test":
            os.system(f"mv {dirname}/test_data/s2ef/all/test_* {datadir}/s2ef/all")
        elif task == "is2re":
            os.system(f"mv {dirname}/data/is2re {datadir}")

        if del_intmd_files:
            cleanup(filename, dirname)
    
    elif task == 'mp':
        os.system(f"wget {download_link} -P {datadir}/{task} -O {datadir}/{task}/{task}_{property}.{TASK_TYPE[task]}")
        data_path = os.path.join(datadir, task, f"{task}_{property}.{TASK_TYPE[task]}")
        output_path = os.path.join(datadir, task, property)
        preprocess_data(data_path, output_path, task, property, split, seed, data_frac)
    
    elif task == 'jarvis':
        output_path = os.path.join(datadir, task, property)
        preprocess_data(None, output_path, task, property, split, seed, data_frac)

    elif task == 'matbench':
        output_path = os.path.join(datadir, task, property)
        preprocess_data(None, output_path, task, property, split, seed, data_frac)
        

    elif task == 'omdb':
        property = 'band_gap'
        os.system(f"wget {download_link} -P {datadir}/{task} -O {datadir}/{task}/{task}_{property}.{TASK_TYPE[task]}")
        label = f'{task}_label'
        os.system(f"wget {label_download_link} -P {datadir}/{task} -O {datadir}/{task}/{task}_{property}.{TASK_TYPE[label]}")
        data_path = os.path.join(datadir, task, f"{task}_{property}.{TASK_TYPE[task]}")
        output_path = os.path.join(datadir, task, property)
        preprocess_data(data_path, output_path, task, property, split, seed, data_frac) 

    elif task == 'tmqm':
        os.system(f"wget {download_link} -P {datadir}/{task} -O {datadir}/{task}/{task}_{property}.{TASK_TYPE[task]}")
        label = f'{task}_label'
        os.system(f"wget {label_download_link} -P {datadir}/{task} -O {datadir}/{task}/{task}_{property}.{TASK_TYPE[label]}")
        data_path = os.path.join(datadir, task, f"{task}_{property}.{TASK_TYPE[task]}")
        output_path = os.path.join(datadir, task, property)
        preprocess_data(data_path, output_path, task, property, split, seed, data_frac) 

    elif task == 'qm9':
        # os.system(f"wget {download_link} -P {datadir}/{task} -O {datadir}/{task}/{task}_{property}.{TASK_TYPE[task]}")
        data_path = os.path.join(datadir, task)
        download_url(download_link, data_path)
        data_path = os.path.join(datadir, task, f"{task}_eV.{TASK_TYPE[task]}")
        output_path = os.path.join(datadir, task, property)
        preprocess_data(data_path, output_path, task, property, split, seed, data_frac) 

    elif task == 'carbon24' or task == 'perov5':
        os.system(f"wget {download_link_train} -P {datadir}/{task} -O {datadir}/{task}/{task}_train.{TASK_TYPE[task]}")
        os.system(f"wget {download_link_val} -P {datadir}/{task} -O {datadir}/{task}/{task}_val.{TASK_TYPE[task]}")
        os.system(f"wget {download_link_test} -P {datadir}/{task} -O {datadir}/{task}/{task}_test.{TASK_TYPE[task]}")
        data_path = os.path.join(datadir, task, f"{task}.{TASK_TYPE[task]}")
        output_path = os.path.join(datadir, task)
        preprocess_data(data_path, output_path, task, property, split, seed, data_frac) 

    else: 
        raise ValueError(
                f"{args.task} task not supported"
            )


def uncompress_data(compressed_dir):
    from scripts import uncompress

    parser = uncompress.get_parser()
    args, _ = parser.parse_known_args()
    args.ipdir = compressed_dir
    args.opdir = os.path.dirname(compressed_dir) + "_uncompressed"
    uncompress.main(args)
    return args.opdir


def preprocess_data(uncompressed_dir, output_path, task, property=None, split=None, seed=None, data_frac=[0.8, 0.1, 0.1]):
    import scripts.preprocess_data as preprocess

    parser = preprocess.get_parser()
    args, _ = parser.parse_known_args()
    args.data_path = uncompressed_dir
    args.out_path = output_path
    args.task = task
    if property is not None:
        args.property = property
    if split is not None:
        args.split = split 
        args.seed = seed
        args.data_frac = data_frac
    preprocess.main(args)


def verify_count(output_path, task, split):
    paths = glob.glob(os.path.join(output_path, "*.txt"))
    count = 0
    for path in paths:
        lines = open(path, "r").read().splitlines()
        count += len(lines)
    assert (
        count == S2EF_COUNTS[task][split]
    ), f"S2EF {split} count incorrect, verify preprocessing has completed successfully."


def cleanup(filename, dirname):
    import shutil

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    if os.path.exists(dirname + "_uncompressed"):
        shutil.rmtree(dirname + "_uncompressed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task to download")
    parser.add_argument("--property", type=str, help="Property to predict")
    parser.add_argument(
        "--split", type=str, help="Corresponding data split to download"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed to split"
    )
    parser.add_argument(
        "--data_frac", type=list, default=[0.6, 0.2, 0.2], help="Data frac to split"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep intermediate directories and files upon data retrieval/processing",
    )
    # Flags for S2EF train/val set preprocessing:
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
        "--data-path",
        type=str,
        default=os.path.join(os.path.dirname(m2models.__path__[0]), "data"),
        help="Specify path to save dataset. Defaults to 'm2models/data'",
    )

    args, _ = parser.parse_known_args()
    np.random.seed(args.seed)
    get_data(
        datadir=args.data_path,
        property=args.property,
        task=args.task,
        split=args.split,
        seed=args.seed,
        data_frac=args.data_frac,
        del_intmd_files=not args.keep,
    )
