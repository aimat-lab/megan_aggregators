import os
import json
import csv
import random
import time 
import joblib
import shutil
import types
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rdkit.Chem as Chem
import xgboost as xgb
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from rich import print as pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

from cats import cats_descriptor
from megan_aggregators.utils import EXPERIMENTS_PATH
from megan_aggregators.utils import handle_keyboard_interrupt
from megan_aggregators.utils import IntegerOutputClassifier

# == SOURCE PARAMETERS ==
DATASET_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'aggregators_new.csv')
TEST_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'aggregators_new__indices_test.json')
NUM_TEST: int = 1000
NUM_VAL: int = 1000
NUM_TRAIN: int = None
BALANCE_TRAIN_SET: bool = True
EXTERNAL_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'external.csv')

# == FEATURE SELECTION PARAMETERS ==
FEATURE_INDICES_PATH: str = os.path.join(EXPERIMENTS_PATH, 'assets', 'yang_feature_indices.json')
DO_FEATURE_SELECTION: bool = False

# == PROCESSING PARAMETERS ==
FINGERPRINT_SIZE: int = 1024
FINGERPRINT_LENGTH: int = 4

# == MODEL PARAMETERS ==
NUM_ESTIMATORS: int = 500
MAX_DEPTH: int = None
CRITERION: str = 'entropy'
CLASS_WEIGHT: str = None # 'balanced'

__DEBUG__ = True
__TESTING__ = True

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment) -> dict:
    """
    This hook is supposed to implement the loading of the raw data from the CSV file into a index_data_map 
    structure which is a dictionary that maps the integer indices to the data dicts that represent each of
    individiual molecule in the dataset.
    """
    # We'll be using this dict structure here to store the raw datasets where the keys will be the integer 
    # indices and the values will be the dictionary structures that contain the information about the single 
    # data points including the labels and the smiles string for example.
    index_data_map: dict[int, dict] = {}
    
    # Here we just want to read all the data in the CSV file that was given as a parameter
    e.log('loading data...')
    with open(e.DATASET_PATH, 'r') as file:
        dict_reader = csv.DictReader(file, fieldnames=['index', 'smiles', 'aggregator', 'non-aggregator'])
        for c, row in enumerate(dict_reader):
            if row['index'].isdigit():
                index = int(row['index'])
                row['index'] = index
                row['aggregator'] = float(row['aggregator'])
                row['non-aggregator'] = float(row['non-aggregator'])
                index_data_map[index] = row
    
    return index_data_map


@experiment.hook('process_dataset', default=True, replace=False)
def process_dataset(e: Experiment, 
                    index_data_map: dict
                    ) -> dict:
    """
    This hook is supposed to implement the processing of the raw dataset that was loaded from the CSV file.
    Each element in the dataset has to be converted into a mol object to extract the necessary numeric features 
    on which the models will be based on - such as the fingerprint representations.
    """
    e.log('processing elements to mol objects and fingerprints...')
    for c, (index, data) in enumerate(index_data_map.items()):
        mol = Chem.MolFromSmiles(data['smiles'])
        data['mol'] = mol
        
        if not mol:
            e.log(f' ! molecule processing error: {data["smiles"]}')
            continue
            
        # rdkit fingerprint
        fp = Chem.RDKFingerprint(mol, e.FINGERPRINT_LENGTH, fpSize=e.FINGERPRINT_SIZE)
        data['fp'] = np.array(fp).astype(float)
        
        # morgan fingerprint
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, e.FINGERPRINT_LENGTH, nBits=e.FINGERPRINT_SIZE)
        data['mfp'] = np.array(mfp).astype(float)
        
        # cats descriptor
        cats = np.array(cats_descriptor([mol]))[0, 1:]
        data['cats'] = np.array(cats).astype(float)
        
        # maccs
        maccs = MACCSkeys.GenMACCSKeys(mol)
        data['maccs'] = np.array(maccs).astype(float)
        
        if c % 10_000 == 0:
            e.log(f' * processed {c}/{len(index_data_map)} elements')


@experiment.hook('get_attributes', default=True, replace=False)
def get_attributes(e: Experiment, data: dict) -> np.ndarray:
    """
    This hook is supposed to turn the information given in the ``data`` dict for each individual input 
    molecule into single numeric numpy array that represents the feature vector.
    
    This default implementation simply concatenates the four different fingerprint vectors that were
    calculated for each molecule.
    """
    if 'feature_indices' in e.data:
        features = []
        if 'fp' in e['feature_indices']:
            features.append(data['fp'][e['feature_indices']['fp']])
        if 'mfp' in e['feature_indices']:
            features.append(data['mfp'][e['feature_indices']['mfp']])
        if 'cats' in e['feature_indices']:
            features.append(data['cats'][e['feature_indices']['cats']])
        if 'maccs' in e['feature_indices']:
            features.append(data['maccs'][e['feature_indices']['maccs']])    
        return np.concatenate(features)
        
    else:
        return np.concatenate([
            data['fp'],
            data['mfp'],
            data['cats'],
            data['maccs']
        ])


@experiment.hook('feature_selection', default=True, replace=False)
def feature_selection(e: Experiment,
                      index_data_map: dict,
                      train_indices: list[int],
                      test_indices: list[int],
                      val_indices: list[int],
                      num_features_max: int = 20,
                      ) -> None:
    
    for feature in ['fp', 'mfp', 'cats', 'maccs']:
        
        with handle_keyboard_interrupt():
            
            e.log(f' * feature selection for feature: {feature}')
            
            x_val = np.array([index_data_map[index][feature] for index in val_indices])
            y_val = np.array([index_data_map[index]['aggregator'] for index in val_indices])
            
            x_test = np.array([index_data_map[index][feature] for index in test_indices])
            y_test = np.array([index_data_map[index]['aggregator'] for index in test_indices])
            
            # ~ training
            # first we train the classifier model on the training data
            x_train = np.array([index_data_map[index][feature] for index in train_indices])
            y_train = np.array([index_data_map[index]['aggregator'] for index in train_indices])
            
            model_rf = RandomForestClassifier(
                n_estimators=e.NUM_ESTIMATORS,
                max_depth=e.MAX_DEPTH,
                criterion=e.CRITERION,
                class_weight=e.CLASS_WEIGHT,
            )
            model_rf.fit(x_train, y_train)
            
            # ~ importances
            # Then we can actually get the feature importances from the random forest model.
            importances = model_rf.feature_importances_
            importance_order = np.argsort(importances)[::-1]
            
            e.log('   plotting the importances...')
            fig, (ax_raw, ax_sort) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
            ax_raw.bar(range(len(importances)), importances)
            ax_sort.bar(range(len(importances)), importances[importance_order], color='red')
            e.commit_fig(f'importances__{feature}.png', fig)
            
            # ~ selection
            # Now we want to select only a subset of the most important features. Instead of just 
            # trusting the feature importances we will actually train a separate model for different 
            # subsets of those features (in the order of importance) and then select the subset with 
            # the highest validation performance.
            
            num_feature_evaluation_map: dict[int, dict] = {}
            
            for num_features in range(2, num_features_max, 1):
            
                # ~ training
                # again, first we train the classifier model on the training data
                x_train_features = x_train[:, importance_order[:num_features]]
                x_val_features = x_val[:, importance_order[:num_features]]
                x_test_features = x_test[:, importance_order[:num_features]]
                
                model_rf = RandomForestClassifier(
                    n_estimators=e.NUM_ESTIMATORS,
                    max_depth=e.MAX_DEPTH,
                    criterion=e.CRITERION,
                    class_weight=e.CLASS_WEIGHT,
                )
                model_rf.fit(x_train_features, y_train)

                # ~ evaluation
                # afterwards we are going to evaluate the model
                y_pred_val = model_rf.predict(x_val_features)
                acc_value_val = accuracy_score(y_val, y_pred_val)
                f1_value_val = f1_score(y_val, y_pred_val)
                
                y_pred_test = model_rf.predict(x_test_features)
                acc_value_test = accuracy_score(y_test, y_pred_test)
                f1_value_test = f1_score(y_test, y_pred_test)
                
                num_feature_evaluation_map[num_features] = {
                    'f1': f1_value_val,
                    'acc': acc_value_val,
                }
                
                e.log(f'   ({num_features:4d}/{num_features_max}) features'
                    f' - val acc: {acc_value_val:.3f} - val f1: {f1_value_val:.3f}'
                    f' - test acc: {acc_value_test:.3f} - test f1: {f1_value_test:.3f}')
                
        # all of this will be exectued regardless of the keyboard interrupt - using whatever 
        # version of the evaluation map that exists at that point.
                
        # ~ selection
        # Finally we select the best subset of features as the one that has the highest 
        # validation F1 performance. We then save the corresponding feature indices into 
        # the experiment data storage so it can be accessed be the featurization hook later 
        # on.
        num_features_best = max(num_feature_evaluation_map, key=lambda k: num_feature_evaluation_map[k]['f1'])
        feature_indices = importance_order[:num_features_best]
        e[f'feature_indices/{feature}'] = feature_indices
        e.log(f' * using {num_features_best} features: {feature_indices}')
                
        # ~ plotting
        # We want to plot the progression of the performance over the different numbers of 
        # features so that we get a good estimate here.
        e.log('plotting num features vs evaluation performance...')
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.plot(
            list(num_feature_evaluation_map.keys()),
            [v['f1'] for v in num_feature_evaluation_map.values()],
        )
        ax.scatter(num_features_best, num_feature_evaluation_map[num_features_best]['f1'], color='red')
        ax.set_title(f'Feature Selection - {feature}\n'
                     f'Feature Indices: {feature_indices}')
        ax.set_xlabel('num features')
        ax.set_ylabel('f1 performance')
        e.commit_fig(f'num_features_performance__{feature}.png', fig)


@experiment.hook('evaluate_model', default=True, replace=False)
def evaluate_model(e: Experiment,
                   index_data_map: dict,
                   indices: list[int],
                   model: RandomForestClassifier,
                   key: str = 'default',
                   ) -> None:
    """
    This hook is supposed to evaluate the ``model`` that was previously trained on the dataset that is 
    defined by the given ``indices`` related to the ``index_data_map``.
    """
    e.log(f'evaluating model "{key}"...')
    # The "get_attributes" hook will turn the raw information in the dict representation for each input 
    # molecule into a single numeric np array feature vector.
    x_test = np.array([
        e.apply_hook('get_attributes', data=index_data_map[index]) 
        for index in indices
    ])
    y_test = np.array([index_data_map[index]['aggregator'] for index in indices])
    
    # Calculating flat metrics
    y_pred = model.predict(x_test)
    acc_value = accuracy_score(y_test, y_pred)
    prec_value = precision_score(y_test, y_pred)
    rec_value = recall_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred)
    e.log(f' * accuracy: {acc_value:.3f}'
          f' * precision: {prec_value:.3f}'
          f' * recall: {rec_value:.3f}'
          f' * f1: {f1_value:.3f}')

    # creating the confusion matrix
    e.log('creating confusion matrix...')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    labels = ['Non-Aggregator', 'Aggregator']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title(f'Confusion Matrix - {key}\n'
                 f'Accuracy: {acc_value:.3f} - F1: {f1_value:.3f}')
    e.commit_fig(f'confusion_matrix__{key}.png', fig)


@experiment.hook('evaluate_model', default=False, replace=False)
def evaluate_model_external(e: Experiment,
                            index_data_map: dict,
                            indices: list[int],
                            model: RandomForestClassifier,
                            key: str = 'default',
                            **kwargs,
                            ) -> None:
    
    e.log('evaluating model on external dataset...')
    
    # Instead of using the externally supplied elements, we'll be using the external dataset which we need 
    # to load here first. Actually if we want to use this external dataset multiple times we cache it in the 
    # experiment data storage.
    e.log('loading external dataset...')
    
    if '_external' in e.data:
        index_data_map = e['_external']
        
    else:
        index_data_map: dict[int, dict] = {}
        with open(e.EXTERNAL_PATH, 'r') as file:
            dict_reader = csv.DictReader(file, fieldnames=['smiles', 'agg'])
            for c, row in enumerate(dict_reader):
                index = c
                row['index'] = index
                
                try:
                    target = float(bool(int(row['agg'])))
                    row['aggregator'] = target
                    row['non-aggregator'] = 1 - target
                    index_data_map[index] = row
                except:
                    continue
                
        e.apply_hook(
            'process_dataset',
            index_data_map=index_data_map,
        )
        
        # Then we cache it so we dont have to do this the next time.
        e['_external'] = index_data_map
                
    e.log(f'loaded external dataset with {len(index_data_map)} elements')
    indices = list(index_data_map.keys())

    e.log('evaluating the model the model...')
    x_test = np.array([
        e.apply_hook('get_attributes', data=index_data_map[index]) 
        for index in indices
    ])
    y_test = np.array([index_data_map[index]['aggregator'] for index in indices])
    y_pred = model.predict(x_test)
    
    # Calculating flat metrics
    y_pred = model.predict(x_test)
    acc_value = accuracy_score(y_test, y_pred)
    prec_value = precision_score(y_test, y_pred)
    rec_value = recall_score(y_test, y_pred)
    f1_value = f1_score(y_test, y_pred)
    e.log(f' * accuracy: {acc_value:.3f}'
          f' * precision: {prec_value:.3f}'
          f' * recall: {rec_value:.3f}'
          f' * f1: {f1_value:.3f}')

    # creating the confusion matrix
    e.log('creating confusion matrix...')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    labels = ['Non-Aggregator', 'Aggregator']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_title(f'Confusion Matrix - {key}\n'
                 f'Accuracy: {acc_value:.3f} - F1: {f1_value:.3f}')
    e.commit_fig(f'external__confusion_matrix__{key}.png', fig)
    


@experiment
def experiment(e: Experiment):
    
    e.log('starting experiment...')
    np.random.seed(0)
    
    e.log('copy cats dependencies...')
    shutil.copy(os.path.join(EXPERIMENTS_PATH, 'cats.py'), e.path)
    shutil.copy(os.path.join(EXPERIMENTS_PATH, 'features.py'), e.path)
    
    # == DATA LOADING ==
    # The "load_dataset" hook will load all the elements from the raw CSV file and provide them in the 
    # format of an index_data_map whose keys are the integer unique indices and the values are 
    # dicts themselves that contain the necessary information about the elements including the 
    # smiles string and the labe.
    index_data_map: dict[int, dict] = e.apply_hook('load_dataset')
            
    indices = list(index_data_map.keys())
    e.log(f'loaded {len(index_data_map)} elements')
    first_element: dict = next(iter(index_data_map.values()))
    pprint(first_element)
    
    # If the test indices are externally determined with a indices file, then we are going to use those, 
    # otherwise we will do a random split using the NUM_TEST parameter.
    e.log('creating train-test-val split...')
    if os.path.exists(e.TEST_INDICES_PATH):
        e.log(f'loading test indices from file @ {e.TEST_INDICES_PATH}')
        with open(e.TEST_INDICES_PATH, 'r') as file:
            content = file.read()
            test_indices = json.loads(content)   
    
    else:
        e.log('randomly selecting test indices...')
        test_indices = random.sample(indices, e.NUM_TEST)
    
    target_index_map: dict[int, list[int]] = defaultdict(list)
    for index in index_data_map.keys():
        data = index_data_map[index]
        target = int(data['aggregator']) 
        target_index_map[target].append(index)
        
    val_indices = (
        random.sample(list(set(target_index_map[0]) - set(test_indices)), k=e.NUM_VAL // 2) + 
        random.sample(list(set(target_index_map[1]) - set(test_indices)), k=e.NUM_VAL // 2)
    )
            
    train_indices = list(set(indices) - set(test_indices) - set(val_indices))
    # If given, the NUM_TRAIN parameter can determine a sub-sampling of the train set. This is 
    # because the whole train set can be very large and it might not make sense the use the 
    # entire train set when testing for example.
    if e.NUM_TRAIN is not None:
        e.log('sub-sampling train indices for efficiency...')
        num_train = min(e.NUM_TRAIN, len(train_indices))
        train_indices = random.sample(train_indices, num_train)
        
    # The BALANCE_TRAIN_SET flag determines whether the train set should be artificially balanced
    # by oversampling the minority class.
    if e.BALANCE_TRAIN_SET:
        e.log('balancing train set by oversampling minority class...')
            
        target_index_map_train = {
            target: list(set(idxs) & set(train_indices))
            for target, idxs in target_index_map.items()
        }
        minorty_target = min(target_index_map_train, key=lambda k: len(target_index_map_train[k]))
        oversampling_factor = max([len(idxs) for idxs in target_index_map_train.values()]) // len(target_index_map_train[minorty_target])
        e.log(f' * minority target: {minorty_target} - oversampling factor: {oversampling_factor}')
        
        train_indices = train_indices + target_index_map_train[minorty_target] * (oversampling_factor - 1)
    
    e.log(f'using {len(train_indices)} train elements, {len(test_indices)} test elements, {len(test_indices)} val elements')
    e['train_indices'] = train_indices
    e['test_indices'] = test_indices
    e['val_indices'] = val_indices
    
    e.commit_json('train_indices.json', train_indices)
    e.commit_json('test_indices.json', test_indices)

    # We actually want to remove the elements that are neither in the train nor the test set 
    # such that they won't unnecessarily be processed.
    unused_indices = list(set(indices) - set(train_indices) - set(test_indices) - set(val_indices))
    e.log(f'{len(unused_indices)} unused elements')
    for index in unused_indices:
        index_data_map.pop(index)
    
    # == MOL OBJECTS AND FINGERPRINTS ==
    # For each element in the dataset we use the SMILES string to create the RDKit molecule object and then 
    # create various features from that including certain global properties and fingerprint vectors.
    
    # :hook process_dataset:
    #       This hook is supposed to implement the processing of the raw data that was loaded from the 
    #       csv file to extract meaningful features. In this case to create the RDKit molecule objects
    #       and the fingerprint vectors.
    e.apply_hook('process_dataset', index_data_map=index_data_map)
    
    e.log('done processing elements')
    pprint(first_element)
    
    # == FEATURE SELECTION ==
    # Prior to the training of the actual models we might want to perform a feature selection to increase 
    # the generalization of the models.

    # If this path is given it means that there exists a json file in which we store an already existing 
    # feature selection. Then we can just load the information from that JSON file.
    if e.FEATURE_INDICES_PATH is not None:
        
        e.log(f'loading feature indices from path @ {e.FEATURE_INDICES_PATH}')
        with open(e.FEATURE_INDICES_PATH, 'r') as file:
            content = file.read()
            feature_indices = json.loads(content)
            e['feature_indices'] = feature_indices

    # If the feature indices are not given we can compute them in the "feature_selection" hook.
    if e.DO_FEATURE_SELECTION:
        
        e.log('starting feature selection...')
        e.apply_hook(
            'feature_selection', 
            index_data_map=index_data_map,
            train_indices=train_indices,
            test_indices=test_indices,
            val_indices=val_indices,    
        )
        
        # We also want to save the information about the feature indices so that we can 
        # possibly just load it in subsequent runs.
        e.commit_json('feature_indices.json', e['feature_indices'])
    
    e.log('creating data features...')
    # The "get_attributes" hook implements the function which turns the raw information about each 
    # input molecule into the actual numeric feature vector which can then be used to train the machine 
    # learning model. The function returns a single numpy array that represents the features.
    x_train = np.array([
        e.apply_hook('get_attributes', data=index_data_map[index])
        for index in train_indices
    ])
    y_train = np.array([index_data_map[index]['aggregator'] for index in train_indices])
    pprint(x_train[0])
    
    x_val = np.array([
        e.apply_hook('get_attributes', data=index_data_map[index])
        for index in val_indices
    ])
    y_val = np.array([index_data_map[index]['aggregator'] for index in val_indices])
    
    # == RANDOM FOREST ==
    # The first model that we want to train is a random forest model.
    
    model_rf = RandomForestClassifier(
        n_estimators=e.NUM_ESTIMATORS,
        max_depth=e.MAX_DEPTH,
        criterion=e.CRITERION,
        class_weight=e.CLASS_WEIGHT,
    )
    
    e.log('fitting random forest model...')
    time_start = time.time()
    model_rf.fit(x_train, y_train)
    time_end = time.time()
    e.log(f' * fitted model in {time_end - time_start:.2f}s')
    
    # The "evaluate_model" hook implements the evaluation of the model that was trained in the previous 
    # step. This may include the calculation of accuracy metrics as well as evaluation plots such as 
    # confusion matrix etc.
    e.apply_hook(
        'evaluate_model',
        index_data_map=index_data_map,
        indices=test_indices,
        model=model_rf,
        key='random_forest',
    )
    
    e.log('saving model...')
    joblib.dump(model_rf, os.path.join(e.path, 'model_rf.joblib'))
    
    # == XGBOOST ==
    # The second model of the ensemble is an XGBoost model.
    
    model_xgb = xgb.XGBClassifier(
        seed=0,
        n_estimators=e.NUM_ESTIMATORS,
        max_depth=8,
        learning_rate=0.1,
    )
    e.log('fitting xgboost model...')
    model_xgb.fit(x_train, y_train)
    
    e.log('saving model...')
    joblib.dump(model_xgb, os.path.join(e.path, 'model_xgb.joblib'))
    
    e.apply_hook(
        'evaluate_model',
        index_data_map=index_data_map,
        indices=val_indices,
        model=model_xgb,
        key='xgboost',
    )
    
    # == GRADIENT BOOSTING ==
    # The final model of the ensemble is a gradient boosting model
    
    model_gb = GradientBoostingClassifier(
        n_estimators=e.NUM_ESTIMATORS,
        max_depth=8,
        learning_rate=0.1,
    )
    e.log('fitting gradient boosting model...')
    model_gb.fit(x_train, y_train)
    
    e.apply_hook(
        'evaluate_model',
        index_data_map=index_data_map,
        indices=val_indices,
        model=model_gb,
        key='gradient_boosting',
    )
    
    e.log('saving model...')
    joblib.dump(model_gb, os.path.join(e.path, 'model_gb.joblib'))
    
    # == MODEL ENSEMBLE ==
    # Finally, we stitch the three models together into a single ensemble model and then 
    # evaluate this ensemble model on the test set as well.
    
    # The "IntegerOutputClassifier" is a simple wrapper that converts the output of the predict 
    # method into actual integer type arrays, which is required to avoid a bug in the VotingClassifier 
    # predict implementation.
    model_rf = IntegerOutputClassifier(model_rf)
    model_gb = IntegerOutputClassifier(model_gb)
    model_xgb = IntegerOutputClassifier(model_xgb)
    
    model_ens = VotingClassifier(
        [('rf', model_rf), ('xgb', model_xgb), ('gb', model_gb)],
        voting='hard',
    )
    model_ens.estimators_ = [model_rf, model_xgb, model_gb]
    model_ens.classes_ = [0, 1]
    label_encoder = LabelEncoder()
    label_encoder.fit(model_ens.classes_)
    model_ens.le_ = label_encoder
    
    e.apply_hook(
        'evaluate_model',
        index_data_map=index_data_map,
        indices=test_indices,
        model=model_ens,
        key='ensemble',
    )
    
    # == MODEL STACK ==
    # As an alternative to the vanilla model ensemble we are also going to try a model stacking 
    # approach. In a model stack we have an additional model at the end which is trained using 
    # the outputs of the individual models as the input.
    
    model_stack = StackingClassifier(
        [
            ('rf', model_rf),
            ('xgb', model_xgb),
            ('gb', model_gb),
        ],
        final_estimator=RandomForestClassifier(),
        cv='prefit',
        passthrough=True,
    )
    # The final prediction model we are going to calibrate so to say on the unseen validation set.
    model_stack.fit(x_val, y_val)

    e.apply_hook(
        'evaluate_model',
        index_data_map=index_data_map,
        indices=test_indices,
        model=model_stack,
        key='stack',
    )

    
experiment.run_if_main()