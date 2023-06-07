
import os
import io
import sys
import itertools
import warnings
import requests
import json
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

from automatminer import MatPipe
from automatminer.automl.adaptors import SinglePipelineAdaptor
from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import DataCleaner, FeatureReducer
from matminer.featurizers.structure import SineCoulombMatrix

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from pymatgen import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.analysis.substrate_analyzer import SubstrateAnalyzer
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matbench.bench import MatbenchBenchmark

from fuzzywuzzy import fuzz
from joblib import dump, load
from tqdm import tqdm
from operator import itemgetter


task_list = ['matbench_dielectric',
            'matbench_expt_gap',
            'matbench_expt_is_metal',
            'matbench_glass',
            'matbench_jdft2d',
            'matbench_log_gvrh',
            'matbench_log_kvrh',
            'matbench_mp_e_form',
            'matbench_mp_gap',
            'matbench_mp_is_metal',
            'matbench_perovskites',
            'matbench_phonons',
            'matbench_steels']

task_file_id_dict = {
    'matbench_phonons': 41053898,
    'matbench_perovskites': 41053895,
    'matbench_log_kvrh': 41053889,
    'matbench_log_gvrh': 41053886,
    'matbench_mp_gap': 41053820,
    'matbench_mp_e_form': 41053460,
    'matbench_mp_is_metal': 41052791
}

def print_sys(s):
    """system print

    Args:
        s (str): the string to print
    """
    print(s, flush=True, file=sys.stderr)


def get_closet_match(predefined_tokens, test_token, threshold=0.2):
    """Get the closest match by Levenshtein Distance.

    Args:
        predefined_tokens (list): Predefined string tokens.
        test_token (str): User input that needs matching to existing tokens.
        threshold (float, optional): The lowest match score to raise errors, defaults to 0.8

    Returns:
        str: the exact token with highest matching prob
            float: probability

    Raises:
        ValueError: no name is matched
    """
    prob_list = []

    for token in predefined_tokens:
        prob_list.append(fuzz.ratio(str(token).lower(), str(test_token).lower()))

    assert len(prob_list) == len(predefined_tokens)

    prob_max = np.nanmax(prob_list)
    token_max = predefined_tokens[np.nanargmax(prob_list)]

    # match similarity is low
    if prob_max / 100 < threshold:
        print_sys(predefined_tokens)
        raise ValueError(
            test_token, "does not match to available values. " "Please double check."
        )
    return token_max, prob_max / 100


def fuzzy_search(name, dataset_names):
    """fuzzy matching between the real dataset name and the input name

    Args:
        name (str): input dataset name given by users
        dataset_names (str): the exact dataset name in TDC

    Returns:
        s: the real dataset name

    Raises:
        ValueError: the wrong task name, no name is matched
    """
    name = name.lower()
    if name in dataset_names:
        s = name
    else:
        s = get_closet_match(dataset_names, name)[0]
    if s in dataset_names:
        return s
    else:
        raise ValueError(
            s + " does not belong to this task, please refer to the correct task name!"
        )


def groupby_itemkey(iterable, item):
    """groupby keyed on (and pre-sorted by) itemgetter(item)."""
    itemkey = itemgetter(item)
    return itertools.groupby(sorted(iterable, key=itemkey), itemkey)


class Oracle:
    def __init__(self, task_name = 'matbench_dielectric'):
        self.task_name = task_name
        if self.task_name in ['substrate_match']:
            self.substrate_analyzer = SubstrateAnalyzer()
        else:
            self.mb = MatbenchBenchmark(autoload=False)

            learner = SinglePipelineAdaptor(
                        regressor=RandomForestRegressor(n_estimators=500),
                        classifier=RandomForestClassifier(n_estimators=500),
                    )

            pipe_config = {
                        "learner": learner,
                        "reducer": FeatureReducer(reducers=[]),
                        "cleaner": DataCleaner(feature_na_method="mean", max_na_frac=0.01, na_method_fit="drop", na_method_transform="mean"),
                        "autofeaturizer": AutoFeaturizer(n_jobs=10, preset="debug"),
                    }

            self.pipe = MatPipe(**pipe_config)

            self.task_name = fuzzy_search(self.task_name, task_list)

            print('Run the training task:', self.task_name)
            if self.task_name == "matbench_dielectric":
                self.task = self.mb.matbench_dielectric
            elif self.task_name == "matbench_expt_gap":
                self.task = self.mb.matbench_expt_gap
            elif self.task_name == "matbench_expt_is_metal":
                self.task = self.mb.matbench_expt_is_metal
            elif self.task_name == "matbench_glass":
                self.task = self.mb.matbench_glass
            elif self.task_name == "matbench_jdft2d":
                self.task = self.mb.matbench_jdft2d
            elif self.task_name == "matbench_log_gvrh":
                self.task = self.mb.matbench_log_gvrh
            elif self.task_name == "matbench_log_kvrh":
                self.task = self.mb.matbench_log_kvrh
            elif self.task_name == "matbench_mp_e_form":
                self.task = self.mb.matbench_mp_e_form
            elif self.task_name == "matbench_mp_gap":
                self.task = self.mb.matbench_mp_gap
            elif self.task_name == "matbench_mp_is_metal":
                self.task = self.mb.matbench_mp_is_metal
            elif self.task_name == "matbench_perovskites":
                self.task = self.mb.matbench_perovskites
            elif self.task_name == "matbench_phonons":
                self.task = self.mb.matbench_phonons
            elif self.task_name == "matbench_steels":
                self.task = self.mb.matbench_steels
            self.task.load()

            model_path = f"{self.task_name}_model.joblib"
            if os.path.exists(model_path):
                self.pipe = load(model_path)
            elif self.task_name in list(task_file_id_dict.keys()):
                file_id = task_file_id_dict[self.task_name]
                url = f"https://figshare.com/ndownloader/files/{file_id}"
                response = requests.get(url, stream=True)
                total_size_in_bytes= int(response.headers.get('content-length', 0))
                block_size = 1024 #1 Kibibyte
                progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
                if response.status_code == 200:
                    content = bytes()
                    for data in response.iter_content(block_size):
                        progress_bar.update(len(data))
                        content += data
                    progress_bar.close()
                    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                        print("ERROR, downloading failed.")
                    with zipfile.ZipFile(io.BytesIO(content)) as zip_ref:
                        zip_ref.extractall()
                    self.pipe = load(model_path)
            else:
                self.pipe.fit(self.task.df, self.task.metadata.target)
                dump(self.pipe, model_path)

    def predict(self, cif_files):
        structures = [Structure.from_file(cif) for cif in cif_files]
        compositions = [structure.composition.reduced_formula for structure in structures]
        if self.task_name in ['matbench_dielectric', 'matbench_jdft2d', 'matbench_log_gvrh', 'matbench_log_kvrh', 
                            'matbench_mp_e_form', 'matbench_mp_gap', 'matbench_mp_is_metal', 'matbench_perovskites', 'matbench_phonons']:
            df_material = pd.DataFrame({'structure': structures})
        elif self.task_name in ['matbench_expt_gap', 'matbench_expt_is_metal', 'matbench_glass']:
            df_material = pd.DataFrame({'composition': compositions})
        elif self.task_name in ['matbench_steels']:
            compositions = [structure.composition for structure in structures]
            compositions = [{element: proportion / sum(composition.values()) for element, proportion in composition.items()} for composition in compositions]
            compositions = ["".join([f"{key}{value}" for key, value in composition.items()]) for composition in compositions]
            df_material = pd.DataFrame({'composition': compositions})
        
        predictions = self.pipe.predict(df_material)[f"{self.task.metadata.target} predicted"]

        df_results = pd.DataFrame({
            'Materials': compositions,
            'Prediction': predictions
        })

        df_results.to_csv(f'{self.task_name}_predictions.csv', index=False)

        return predictions.tolist()

    def substrate_match(self, film, substrates=['Si.cif']):
        film = Structure.from_file(film)
        substrates = [Structure.from_file(sub) for sub in substrates]
        all_matches = []

        for substrate in tqdm(substrates):
            matches = groupby_itemkey(self.substrate_analyzer.calculate(film, substrate, lowest=True),"sub_miller")
            lowest_matches = [min(g, key=itemgetter("match_area")) for k, g in matches]

            for match in lowest_matches:
                db_entry = {
                    "sub_formula": substrate.composition.reduced_formula,
                    "orient": " ".join(map(str, match["sub_miller"])),
                    "film_orient": " ".join(map(str, match["film_miller"])),
                    "area": match["match_area"],
                }

                if "elastic_energy" in match:
                    db_entry["energy"] = match["elastic_energy"]
                    db_entry["strain"] = match["strain"]

                all_matches.append(db_entry)

        df_matches = pd.DataFrame(all_matches)
        df_matches.sort_values("area")
        df_matches.to_csv(f'{self.task_name}_results.csv', index=False)

        return all_matches

