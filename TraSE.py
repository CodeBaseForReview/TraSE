# Submitted with paper on github for review: CodeBaseForReview/TraSE. DO NOT MODIFY, DISTRIBUTE OR REUSE.

# python imports
import itertools
import json
import multiprocessing as mp
import os
from random import shuffle
import pandas as pd
from shutil import rmtree
import re
import string
from xml.etree import ElementTree

# external libraries
from matplotlib import pyplot as plt
import flair
import numpy as np
import torch
from flair.data import Sentence
from flair.embeddings import TransformerWordEmbeddings
from nltk import word_tokenize, pos_tag
from nltk.tokenize import sent_tokenize
from scipy.spatial.distance import cosine
from scipy.special import kl_div
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import preprocessor  # tweet-preprocessor 0.6.0

# Global variables
SEPARATOR = "$$"
n_CROSS_VALIDATE = 4  # Need to fix get_data() to support n_CROSS_VALIDATE != 4
MAX_TOPIC_WORD_COUNT = 20
n_DIM_TraSE = 180

# Assign compute device if necessary
SELECTED_COMPUTE_DEVICE = "cpu"
flair.device = torch.device(SELECTED_COMPUTE_DEVICE)


class TraSEFeatureExtractor:
    """
    Utilizes BERT language model to extract TraSE feature representation from text file.
    """

    def __init__(self, topic_word_counts=MAX_TOPIC_WORD_COUNT, word_embed_model="bert-base-uncased"):
        """
        :param topic_word_counts: Max limit on the number of topic words (NOUNS) extracted from text sample. Default: 20
        :param word_embed_model: Embedding model fetched from Huggingface.co through flair. Default: "bert-base-uncased"
        """
        self.topic_word_count = topic_word_counts
        self.embedding_model = TransformerWordEmbeddings(word_embed_model)

    def _get_likely_topic_words(self, text):
        """
        Extract most frequent NOUNS from text.
        :param text: Raw text string
        :return: Topic word list
        """
        word_dict = dict()
        for word, tag in pos_tag(word_tokenize(text), tagset="universal"):
            if tag == "NOUN":
                try:
                    freq = word_dict[word]
                    word_dict[word] = freq + 1
                except KeyError:
                    word_dict[word] = 1
        try:
            _, likely_topic_words = zip(*sorted([(word_dict[x], x) for x in word_dict.keys()], reverse=True))
            likely_topic_words = list(likely_topic_words)
            if len(likely_topic_words):
                likely_topic_words = likely_topic_words[:self.topic_word_count]
            else:
                likely_topic_words = []
        except ValueError:
            likely_topic_words = []
        return likely_topic_words

    def _embed_text(self, raw_text):
        """
        Embeds raw text sample using the chosen embedding model
        :param raw_text: Raw text string
        :return: Embedding dictionary
        """
        text_sample_embeddings, embedding_entry_counter = dict(), 0
        likely_topic_words = self._get_likely_topic_words(raw_text)
        for raw_sentence in sent_tokenize(raw_text):
            try:
                sentence = Sentence(raw_sentence)
                self.embedding_model.embed(sentence)
                embedding_buffer = []
                for i in range(0, len(sentence.tokens)):
                    embedding_buffer.append(np.array(sentence[i].embedding.cpu()))
                    if sentence.tokens[i].text in likely_topic_words:
                        if len(embedding_buffer) > 2:
                            text_sample_embeddings[embedding_entry_counter] = embedding_buffer
                            embedding_buffer = [np.array(sentence[i].embedding.cpu())]
                            embedding_entry_counter += 1
                        else:
                            embedding_buffer = [np.array(sentence[i].embedding.cpu())]
                if len(embedding_buffer) > 2:
                    text_sample_embeddings[embedding_entry_counter] = embedding_buffer
            except IndexError:
                print("Invalid sentence found: ", raw_sentence)
        return text_sample_embeddings

    @staticmethod
    def _recursive_resultant_evolution(text_embed_dict):
        """
        Recursive resultant evolution algorithm for transforming embedding dictionary to TraSE feature representation
        :param text_embed_dict: Embedding dictionary
        :return: TraSE feature vector
        """
        direction = np.zeros(n_DIM_TraSE, dtype=int)
        for key in text_embed_dict.keys():
            all_entries = text_embed_dict[key]
            initial_dir_vector = np.subtract(all_entries[1], all_entries[0])
            resultant = np.copy(initial_dir_vector)
            try:
                for i in range(2, len(all_entries)):
                    new_dir_vec = np.subtract(all_entries[i], all_entries[i - 1])
                    angle = int(round(np.rad2deg(np.arccos(1 - cosine(resultant, new_dir_vec)))))
                    resultant = np.add(new_dir_vec, initial_dir_vector)
                    if 0 <= angle < n_DIM_TraSE:
                        direction[angle] += 1
            except ValueError:
                pass
        return direction

    def get_feature_vector(self, raw_text):
        """
        Extract TraSE feature vector (array-like) from raw text sample string(string-like)
        :param raw_text: Raw text string
        :return: TraSE feature vector
        """
        return self._recursive_resultant_evolution(self._embed_text(raw_text))


class TraSEEval:
    """
    Performs authorship attribution experiments on a chosen corpus using TraSE feature representation.
    Expects the following organization: corpus folder -> author folder -> sample files
    Expects the naming convention: corpus folder -> corpus-name,
                                   author folder -> author-id
                                   sample_file -> author-id_otherTags.txt
    """

    def __init__(self, dataset_path, working_directory_path, use_multiprocessing=0):
        """
        :param dataset_path: Path to the corpus folder
        :param working_directory_path: Path to working directory to store feature vector, other intermediate working
        files and classification results
        :param use_multiprocessing: Core count for multiprocessing. Set 0 or 1 for sequential processing.
        """
        self.dataset_path = dataset_path
        self.work_dir_path = working_directory_path

        self.core_count = 0
        if use_multiprocessing > 2:
            self.core_count = use_multiprocessing

        if len(working_directory_path):
            if not os.path.isdir(self.work_dir_path):
                os.mkdir(self.work_dir_path)

        self.overwrite_flag = False

    def _generate_cross_validation_config_json(self, fold_count=n_CROSS_VALIDATE):
        """
        Generates cross-validation configuration.
        :param fold_count: Fold count for cross-validation. Currently only supports 4-fold CV.
        :return:
        """
        folds = dict()
        for author_id in os.listdir(self.dataset_path):
            file_list = os.listdir(os.path.join(self.dataset_path, author_id))
            if len(file_list) >= fold_count:
                splits = np.array_split(file_list, fold_count)
                folds[author_id] = [splits[_x].tolist() for _x in range(0, fold_count)]
        with open(os.path.join(self.work_dir_path, "crossValFolds.json"), "w") as f_json_dump:
            json.dump(folds, f_json_dump)

    @staticmethod
    def _sparse_encode_feature_vector(feature_vector):
        """
        Encode TraSE feature representation for storage
        :param feature_vector: array-like, TraSE feature vector
        :return: string-like, encoded TraSE feature vector
        """
        return [str(index) + "-" + str(feature_vector[index]) for index in range(0, len(feature_vector)) if
                feature_vector[index]]

    @staticmethod
    def _sparse_decode_feature_vector(encoded_text_list):
        """
        Decode encoded TraSE feature representation from storage
        :param encoded_text_list: string-like, Encoded TraSE feature vector
        :return: array-like, TraSE feature vector. None if feature vector is corrupted or unavailable.
        """
        feature_vector = np.zeros(n_DIM_TraSE)
        for element in encoded_text_list:
            index, freq = [int(x) for x in element.split("-")]
            feature_vector[index] = freq
        decoded_feature_vector = None
        if sum(feature_vector):
            decoded_feature_vector = np.divide(feature_vector, sum(feature_vector))
        return decoded_feature_vector

    def _extract_features_for_author_wrapper(self, author_id):
        """
        Wrapper function for parallel processing. Extracts feature vectors and stores in working directory from a
        chosen author-id.
        :param author_id: string-like, author-id
        :return: None
        """
        print("Currently processing: ", author_id)
        _trase_obj = TraSEFeatureExtractor()
        author_path = os.path.join(self.dataset_path, author_id)
        output_path = os.path.join(self.work_dir_path, author_id + ".txt")

        if self.overwrite_flag:
            if os.path.isfile(output_path):
                os.remove(output_path)
            file_names_to_process = os.listdir(author_path)
        else:
            if os.path.isfile(output_path):
                existing_sample_file_names = []
                with open(output_path) as f_data:
                    for entry in f_data.readlines():
                        existing_sample_file_names.append(entry.split(SEPARATOR)[0])

                file_names_to_process = []
                for file_name in os.listdir(author_path):
                    if file_name not in existing_sample_file_names:
                        file_names_to_process.append(file_name)
            else:
                file_names_to_process = os.listdir(author_path)

        for file_name in file_names_to_process:
            with open(os.path.join(author_path, file_name), encoding="utf-8") as f_read_sample:
                feature_vector = _trase_obj.get_feature_vector(f_read_sample.read())
                encoded_feature_vector = self._sparse_encode_feature_vector(feature_vector)
                with open(output_path, "a") as f_author_output:
                    f_author_output.write(SEPARATOR.join([str(file_name)] + encoded_feature_vector) + "\n")

    def extract_features(self, overwrite_existing_data=False):
        """
        Extracts features from the target corpus and saves the encoded TraSE feature vectors in the working directory
        :param overwrite_existing_data: if True, overwrites existing data in the working directory. If False, checks
        for missing/unprocessed files and appends to existing files
        :return: None
        """
        print("Extracting features.....")
        self.overwrite_flag = overwrite_existing_data

        author_list = os.listdir(self.dataset_path)
        if self.core_count > 1:
            with mp.Pool(self.core_count) as p:
                p.map(self._extract_features_for_author_wrapper, author_list)
        else:
            for author_id in author_list:
                self._extract_features_for_author_wrapper(author_id)
        print("Feature extraction complete!")

    @staticmethod
    def _correct_sample(sample, author_profile):
        """
        Corrects sample feature vector using the author's profile
        :param sample: array-like, TraSE feature representation for sample
        :param author_profile: array-like, mean TraSE feature representation for the author
        :return: array-like, sample corrected for sparsity and missing data using author's profile
        """
        for x in range(0, len(sample)):
            if np.isnan(sample[x]) or np.isinf(sample[x]):
                sample[x] = author_profile[x]
            else:
                sample[x] = abs(author_profile[x] - sample[x])
        return sample

    @staticmethod
    def _get_author_profile(train_data):
        """
        Mean TraSE feature representation from training data for the author
        :param train_data: array-like, all train TraSE feature vectors
        :return: array-like, author profile
        """
        mean_profile_ = np.zeros((2, n_DIM_TraSE))
        for train_sample in train_data:
            if train_sample is not None:
                for index, x in enumerate(train_sample):
                    x_ = float(x)
                    if not np.isnan(x_) and not np.isinf(x_):
                        mean_profile_[0, index] += x_
                        mean_profile_[1, index] += 1
        author_profile = [mean_profile_[0, y] / max(1, mean_profile_[1, y]) for y in range(0, n_DIM_TraSE)]
        return author_profile

    @staticmethod
    def generate_classification_report(true_labels, predicted_labels):
        """
        Results on classification test. Includes accuracy, macro-F1, precision and recall
        :param true_labels: array-like, Ground-truth labels
        :param predicted_labels: array-like, Predicted labels from the trained model
        :return: accuracy (float), macro-F1 (float), precision (float), recall (float), log_text (string)
        """
        log_text = ""
        acc = round(accuracy_score(true_labels, predicted_labels), 2)
        f1 = round(f1_score(true_labels, predicted_labels, average="macro"), 2)
        precision = round(precision_score(true_labels, predicted_labels, average="macro"), 2)
        recall = round(recall_score(true_labels, predicted_labels, average="macro"), 2)

        print("--------Classification Report--------")
        print("Accuracy: ", acc)
        print("F1 score: ", f1)
        print("Precision: ", precision)
        print("Recall: ", recall)

        log_text += "--------Classification Report--------\n"
        log_text += "Accuracy: " + str(acc) + "\n"
        log_text += "F1 score: " + str(f1) + "\n"
        log_text += "Precision: " + str(precision) + "\n"
        log_text += "Recall: " + str(recall) + "\n"
        return acc, f1, precision, recall, log_text

    def get_data(self, cv_config_json, split, fold):
        """
        Loads data for classification test
        :param cv_config_json: JSON dict, author-id:{[fold1-files], [fold2-files] ..... [foldn-files]}
        :param split: float, train-test split ratio. Supports: 0.25, 0.50, 0.75
        :param fold: int, cross-validation fold index
        :return: train_files, test_files, train_file_names, test_file_names
        train_files, test_files: list of tuples, [(sample_TraSE_feature_vector, label)]
        train_file_names, test_file_names: list of strings
        """
        print("Loading data.....")
        train_splits = {
            0.75: [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 3]],
            0.50: [[0, 1], [1, 2], [2, 3], [3, 0]],
            0.25: [[0], [1], [2], [3]]
        }
        author_list = cv_config_json.keys()
        _train_files, test_files, test_file_names = [], [], []
        for label, author in enumerate(author_list):
            file = author + ".txt"
            with open(os.path.join(self.work_dir_path, file), "r") as f:
                data = f.read().splitlines()

            train_fold_indices = train_splits[split][fold]
            auth_samples = [x.split(SEPARATOR)[0] for x in data]
            train_samples = list(itertools.chain.from_iterable([cv_config_json[author][i] for i in train_fold_indices]))
            test_samples = [i for i in auth_samples if i not in train_samples]
            train_data = [self._sparse_decode_feature_vector(x.split(SEPARATOR)[1:]) for x in data if
                          x.split(SEPARATOR)[0] in train_samples]
            mean_profile = self._get_author_profile(train_data)

            for entry in data:
                file_name = entry.split(SEPARATOR)[0]
                sample = self._sparse_decode_feature_vector(entry.split(SEPARATOR)[1:])
                if sample is not None:
                    corrected_sample = self._correct_sample(sample, mean_profile)
                    if file_name in test_samples:
                        test_file_names.append(file_name)
                        test_files.append((sample, label))
                    if file_name in train_samples:
                        _train_files.append(((corrected_sample, label), file_name))
        shuffle(_train_files)
        train_files, train_file_names = [list(_x) for _x in zip(*_train_files)]
        print("Data load complete !")
        return train_files, test_files, train_file_names, test_file_names

    def train_and_evaluate_on_dataset(self, train_test_split_ratio, fold_select=-1, cv_config_path=None,
                                      save_tree_stats=True):
        """
        Classification test on chosen corpus using TraSE feature representation. Results are displayed and saved in
        working directory
        :param train_test_split_ratio: float, supports 0.25, 0.5 and 0.75
        :param fold_select: int, -1 for all-folds or fold index for a particular fold
        :param cv_config_path: string-like, path to JSON-dict CV configuration. Checks working directory for existing
        version else generates new one.
        :param save_tree_stats: bool, if True (default), saves decision tree statistics/structure in JSON-dict for
        further analysis
        :return: None
        """

        if cv_config_path is None:
            default_cv_config_path = os.path.join(self.work_dir_path, "crossValFolds.json")
            if not os.path.isfile(default_cv_config_path):
                self._generate_cross_validation_config_json()
            with open(default_cv_config_path, "r") as fp:
                cv_folds = json.load(fp)
        else:
            with open(cv_config_path, "r") as fp:
                cv_folds = json.load(fp)

        print("Processing training data ratio: ", train_test_split_ratio)
        if fold_select == -1:
            folds = range(0, n_CROSS_VALIDATE)
        else:
            folds = [fold_select]

        for fold in folds:
            print("Fold: ", fold)
            train_files, test_files, train_file_names, test_file_names = self.get_data(cv_folds, train_test_split_ratio,
                                                                                       fold)
            train_data, train_labels = zip(*train_files)
            test_data, test_labels = zip(*test_files)
            print("Data splits....")
            print("Train files: ", len(train_labels), " Test files: ", len(test_labels))
            print("Training model")
            model = DecisionTreeClassifier(random_state=0).fit(list(train_data), list(train_labels))
            print("Making predictions")
            predictions = model.predict(list(test_data))
            acc, f1, precision, recall, log_text = self.generate_classification_report(test_labels, predictions)

            output_file_name = str(train_test_split_ratio)[2:] + "_" + str(fold)

            print("Saving classification report")
            with open(os.path.join(self.work_dir_path, output_file_name + "_report.txt"), "w") as f_logger:
                f_logger.write(log_text)

            if save_tree_stats:
                print("Saving classifier data")
                self._save_decision_tree_stats(model, output_file_name, cv_folds, train_files, test_files,
                                               train_file_names, test_file_names, predictions)

    def _save_decision_tree_stats(self, trained_model, output_file_name, cv_config_json, train_files, test_files,
                                  train_file_names, test_file_names, predicted_labels):
        """
        Saves essential decision tree structure/statistics. Produces three JSON-dicts
        outputFileName_treeData.json - node_id: decision_dimension_in_TraSE, decision_threshold, child(left), child(right)
        outputFileName_trainData.json - train_data: decision path with train file names
        outputFileName_testData.json - test_data: test file name with decision path, ground truth author, predicted author
        :param trained_model: trained scikit-learn decision tree model
        :param output_file_name: generated name showing evaluation configuration
        :param cv_config_json: JSON-dict
        :param train_files: list of tuples, Training data [(sample_TraSE_feature_vector, label)]
        :param test_files: list of tuples, Testing data [(sample_TraSE_feature_vector, label)]
        :param train_file_names: list of string, Train file names
        :param test_file_names: list of string, Test file names
        :param predicted_labels: list, prediction from the trained model on the test data
        :return: None
        """

        # Dump all processed data
        tree_data_dump_path = os.path.join(self.work_dir_path, "tree_data")
        if not os.path.isdir(tree_data_dump_path):
            os.mkdir(tree_data_dump_path)

        # node_id: decision_dimension_in_TraSE, decision_threshold, child(left), child(right)
        n_nodes = trained_model.tree_.node_count
        children_left = trained_model.tree_.children_left
        children_right = trained_model.tree_.children_right
        feature = trained_model.tree_.feature
        threshold = trained_model.tree_.threshold
        tree_structure_dict = {node_id: (int(feature[node_id]), float(threshold[node_id]), int(children_left[node_id]),
                                         int(children_right[node_id])) for node_id in range(n_nodes)}
        with open(os.path.join(tree_data_dump_path, output_file_name + "_treeData.json"), "w") as f_json:
            json.dump(tree_structure_dict, f_json)
        tree_structure_dict = None

        # train_data: decision path with train file names
        train_data, _ = zip(*train_files)
        all_train_decision_paths = trained_model.decision_path(list(train_data)).todense().tolist()
        train_decision_data = dict()
        for file_index, (current_sample, label) in enumerate(train_files):
            current_file_name = train_file_names[file_index]
            current_sample_path = all_train_decision_paths[file_index]
            decision_path_str = SEPARATOR.join([str(_decision) for _decision in current_sample_path])
            try:
                existing_file_names = train_decision_data[decision_path_str]
                existing_file_names.append(current_file_name)
                train_decision_data[decision_path_str] = existing_file_names
            except KeyError:
                train_decision_data[decision_path_str] = [current_file_name]
        with open(os.path.join(tree_data_dump_path, output_file_name + "_trainData.json"), "w") as f_json:
            json.dump(train_decision_data, f_json)
        train_decision_data = None

        # test_data: test file name with decision path, ground truth author, predicted author
        test_decision_data = dict()
        author_id_list = list(cv_config_json.keys())
        test_data, _ = zip(*test_files)
        all_test_decision_paths = trained_model.decision_path(list(test_data)).todense().tolist()
        for file_index, (current_sample, label) in enumerate(test_files):
            current_file_name = test_file_names[file_index]
            predicted_author_id = author_id_list[predicted_labels[file_index]]
            current_sample_path = all_test_decision_paths[file_index]
            decision_path_str = SEPARATOR.join([str(_decision) for _decision in current_sample_path])
            test_decision_data[current_file_name] = (decision_path_str, author_id_list[label], predicted_author_id)
        with open(os.path.join(tree_data_dump_path, output_file_name + "_testData.json"), "w") as f_json:
            json.dump(test_decision_data, f_json)
        test_decision_data = None


class StandardizePANdataset:
    """
    Interface for PAN datasets. Transforms PAN datasets into template datasets (organization and naming convention) for
    processing through TraSEEval.
    """

    def __init__(self, original_PAN_dataset_path, working_directory_path, use_multiprocessing=0,
                 options="ground-truth"):
        """
        Transforms typical PAN  authorship attribution dataset into PAN_mod dataset. Extracts features and evaluates on
        the PAN_mod dataset.
        :param original_PAN_dataset_path: string-like
        :param working_directory_path: string-like
        :param use_multiprocessing: int, use 0 or 1 for sequential processing
        :param options: Handles issues with ground truth keys in "ground-truth.json". Uses "ground-truth" for
        PAN11/PAN12  and "ground_truth" for PAN18/PAN19
        """
        self.original_path = original_PAN_dataset_path
        self.work_dir_path = working_directory_path
        self.core_count = use_multiprocessing
        self.processed_dataset_path = original_PAN_dataset_path + "_mod"
        self.options = options

        self._fixed_split, self._fixed_fold = 0.75, 0

        if not os.path.isfile(os.path.join(self.work_dir_path, "crossValFolds.json")):
            if os.path.isdir(self.processed_dataset_path):
                rmtree(self.processed_dataset_path)
            if os.path.isdir(self.work_dir_path):
                rmtree(self.work_dir_path)

        if not os.path.isdir(self.work_dir_path):
            os.mkdir(self.work_dir_path)

        if not os.path.isdir(self.processed_dataset_path):
            os.mkdir(self.processed_dataset_path)

        if not os.path.isfile(os.path.join(self.work_dir_path, "crossValFolds.json")):
            self._generate_template_corpus()

        self._trase_eval_obj = TraSEEval(dataset_path=self.processed_dataset_path,
                                         working_directory_path=self.work_dir_path,
                                         use_multiprocessing=self.core_count)

    def _read_gt_data_for_testing_files(self):
        """
        Reads ground-truth labels from "ground-truth.json" in PAN dataset
        :return: dict, unknown-label: ground_truth-label
        """
        label_dict = dict()
        gt_json_path = os.path.join(self.original_path, "ground-truth.json")
        with open(gt_json_path) as f_json:
            for _label_entry in json.load(f_json)[self.options]:
                if _label_entry["true-author"] != "<UNK>":
                    label_dict[_label_entry["unknown-text"]] = _label_entry["true-author"]
        return label_dict

    def _generate_template_corpus(self, fold_count=n_CROSS_VALIDATE):
        """
        Transforms original PAN dataset into PAN_mod template dataset.
        :param fold_count: int, cross-validation fold count
        :return: None
        """

        author_train_files, author_test_files = dict(), dict()
        author_file_counter = dict()
        for _author_id in os.listdir(self.original_path):
            if "candidate" in _author_id:
                author_train_files[_author_id], author_file_counter[_author_id] = [], 0
                author_test_files[_author_id] = []

                template_author_path = os.path.join(self.processed_dataset_path, _author_id)
                if not os.path.isdir(template_author_path):
                    os.mkdir(template_author_path)

                author_path = os.path.join(self.original_path, _author_id)
                for _file_id, _file_name in enumerate(os.listdir(author_path)):
                    new_file_name = _author_id + "_" + str(_file_id) + ".txt"
                    author_train_files[_author_id].append(new_file_name)
                    with open(os.path.join(author_path, _file_name), encoding="utf-8") as f_text_read:
                        with open(os.path.join(template_author_path, new_file_name), "w",
                                  encoding="utf-8") as f_text_dump:
                            author_file_counter[_author_id] = _file_id
                            f_text_dump.write(f_text_read.read())

        _gt_label_dict = self._read_gt_data_for_testing_files()
        test_path = os.path.join(self.original_path, "unknown")
        for _file_name in os.listdir(test_path):
            try:
                _author_id = _gt_label_dict[_file_name]
                author_file_counter[_author_id] += 1
                out_file_name = _author_id + "_" + str(author_file_counter[_author_id]) + ".txt"
                author_test_files[_author_id].append(out_file_name)

                with open(os.path.join(test_path, _file_name), encoding="utf-8") as f_text_read:
                    with open(os.path.join(self.processed_dataset_path, _author_id, out_file_name), "w",
                              encoding="utf-8") as f_text_dump:
                        f_text_dump.write(f_text_read.read())
            except KeyError:
                pass

        folds = dict()
        for _author_id in author_train_files.keys():
            file_list = author_train_files[_author_id]
            splits = np.array_split(file_list, fold_count - 1)
            folds[_author_id] = [splits[_x].tolist() for _x in range(0, fold_count - 1)] + [
                author_test_files[_author_id]]
        with open(os.path.join(self.work_dir_path, "crossValFolds.json"), "w") as f_json_dump:
            json.dump(folds, f_json_dump)

    def extract_features(self):
        """
        Extract TraSE features and store them for evaluation
        :return: None
        """
        self._trase_eval_obj.extract_features(overwrite_existing_data=False)

    def train_and_evaluate_on_dataset(self):
        """
        Evaluate PAN dataset. Training and testing data are fixed according to PAN dataset.
        Classification report is displayed and stored by default.
        :return: None
        """
        print("REMINDER: PAN PROTOCOL. FIXED TRAIN-TEST DATA. NO CROSS-VALIDATION AVAILABLE. ONLY FOLD 0.")
        self._trase_eval_obj.train_and_evaluate_on_dataset(train_test_split_ratio=self._fixed_split,
                                                           fold_select=self._fixed_fold)


class TraSEExperiments(TraSEEval):
    def __init__(self, dataset_path, working_directory_path, use_multiprocessing=0):
        """
        Experiments reported in the source paper for supporting TraSE
        :param dataset_path: empty string or path
        :param working_directory_path:
        """
        super().__init__(dataset_path=dataset_path, working_directory_path=working_directory_path,
                         use_multiprocessing=use_multiprocessing)

        if len(dataset_path) and len(working_directory_path):
            default_cv_config_path = os.path.join(self.work_dir_path, "crossValFolds.json")
            if not os.path.isfile(default_cv_config_path):
                self._generate_cross_validation_config_json()

            with open(default_cv_config_path, "r") as fp:
                cv_folds = json.load(fp)
            self.author_id_list = list(cv_folds.keys())

            self.tree_data_path = os.path.join(self.work_dir_path, "tree_data")
            self.experiment_path = os.path.join(self.work_dir_path, "experiment_data")
            if not os.path.isdir(self.experiment_path):
                os.mkdir(self.experiment_path)

    def _run_multi_collinearity_test(self):
        """
        Test to capture the VIF factor for each dimension in TraSE for each author
        :return: None
        """
        print("Running VIF multi-collinearity test....")
        _multi_collinearity_data_path = os.path.join(self.experiment_path, "vif_multicollinearity_test")
        _failed_author_debug_path = os.path.join(_multi_collinearity_data_path, "failed_authors.txt")
        if not os.path.isdir(_multi_collinearity_data_path):
            os.mkdir(_multi_collinearity_data_path)

        _validity_check_failed_authors = []
        if os.path.isfile(_failed_author_debug_path):
            with open(_failed_author_debug_path) as f_read:
                _validity_check_failed_authors += f_read.read().splitlines()

        for author_id in self.author_id_list:
            _dump_path = os.path.join(_multi_collinearity_data_path, author_id + ".json")
            if not os.path.isfile(_dump_path):
                if author_id not in _validity_check_failed_authors:
                    with open(os.path.join(self.work_dir_path, author_id + ".txt"), "r") as f:
                        data = f.read().splitlines()

                    valid_feature_vectors = []
                    for entry in data:
                        sample = self._sparse_decode_feature_vector(entry.split(SEPARATOR)[1:])
                        if sample is not None:
                            valid_feature_vectors.append(sample)

                    data_frame_dict_buf = {_x: np.array(valid_feature_vectors)[:, _x] for _x in range(0, n_DIM_TraSE) if
                                           sum(np.array(valid_feature_vectors)[:, _x])}
                    data_frame = pd.DataFrame(data_frame_dict_buf)

                    _VIF_interpretation = {"not_correlated": [], "moderately_correlated": [], "highly_correlated": []}
                    valid_dim_count = len(data_frame_dict_buf.keys())
                    for _i in range(0, valid_dim_count):
                        try:
                            current_dim = list(data_frame_dict_buf.keys())[_i]
                            vif_score = variance_inflation_factor(data_frame, _i)
                            if vif_score == 1:
                                _VIF_interpretation["not_correlated"].append(current_dim)
                            elif 1 < vif_score <= 5:
                                _VIF_interpretation["moderately_correlated"].append(current_dim)
                            else:
                                _VIF_interpretation["highly_correlated"].append(current_dim)
                        except OverflowError:
                            pass
                    _data_validity_check = sum(
                        [len(_VIF_interpretation[_x]) for _x in _VIF_interpretation.keys()]) == valid_dim_count
                    if _data_validity_check:
                        with open(_dump_path, "w") as f_json:
                            json.dump(_VIF_interpretation, f_json)
                    else:
                        _validity_check_failed_authors.append(author_id)

        with open(_failed_author_debug_path, "w") as f_write:
            f_write.write("\n".join(_validity_check_failed_authors) + "\n")
        print("VIF multi-collinearity test complete")

    def multi_collinearity_analysis(self, train_test_split_ratio, fold, show_plots=True):
        """
        Used to assert the need for decision trees over other classification approaches to attribute over style. Analyze
        VIF coefficients for each dimension in TraSE with its relevance in establishing author identity.
        :param train_test_split_ratio: float, supports 0.25, 0.50 and 0.75
        :param fold: int, selected validation fold
        :param show_plots: bool, displays results from analysis as plots (shown in paper)
        :return: dict,
        """
        self._run_multi_collinearity_test()

        print("Analyzing results from VIF test.....")

        # run correlation between vif interpretation and decision node dims for the author
        _data_file_name = str(train_test_split_ratio)[2:] + "_" + str(fold)
        with open(os.path.join(self.tree_data_path, _data_file_name + "_trainData.json")) as f_json_load:
            train_path_dict = json.load(f_json_load)

        # authors and their decision paths in training
        author_to_decision_paths_dict = {_x: [] for _x in self.author_id_list}
        for _decision_path in train_path_dict.keys():
            authors_in_entry = set([_x.split("_")[0] for _x in train_path_dict[_decision_path]])
            decision_path = [int(_x) for _x in _decision_path.split(SEPARATOR)]
            for author_id in authors_in_entry:
                author_to_decision_paths_dict[author_id].append(decision_path)

        # TraSE feature dims required for identifying author in decision tree
        with open(os.path.join(self.tree_data_path, _data_file_name + "_treeData.json")) as f_json_load:
            tree_data_dict = json.load(f_json_load)
        author_to_decision_dims_dict = {_x: [] for _x in self.author_id_list}
        for author_id in self.author_id_list:
            for path in author_to_decision_paths_dict[author_id]:
                for node_id in list(np.where(np.array(path) == 1)[0]):
                    _dim = tree_data_dict[str(node_id)][0]
                    if _dim >= 0:
                        author_to_decision_dims_dict[author_id].append(_dim)

        # Load VIF interpretable dims for an author
        VIF_on_corpus_results = {"not_correlated": [], "moderately_correlated": [], "highly_correlated": []}
        for author_id in self.author_id_list:
            vif_path = os.path.join(self.experiment_path, "vif_multicollinearity_test", author_id + ".json")
            if os.path.isfile(vif_path):
                _VIF_on_author_results = {"not_correlated": 0, "moderately_correlated": 0, "highly_correlated": 0}
                with open(vif_path) as f_json_load:
                    vif_interpretation = json.load(f_json_load)
                for _dim in list(set(author_to_decision_dims_dict[author_id])):
                    for corr_type in vif_interpretation.keys():
                        if _dim in vif_interpretation[corr_type]:
                            _VIF_on_author_results[corr_type] += 1
                total_dims = sum([_VIF_on_author_results[_x] for _x in _VIF_on_author_results.keys()])
                if total_dims:
                    for corr_type in _VIF_on_author_results.keys():
                        VIF_on_corpus_results[corr_type].append(
                            round(_VIF_on_author_results[corr_type] / total_dims, 2))

        if show_plots:
            plt.figure()
            _labels = {"not_correlated": "Not correlated",
                       "moderately_correlated": "Moderately correlated",
                       "highly_correlated": "Highly correlated"}
            for corr_type in VIF_on_corpus_results.keys():
                a, b = np.unique(VIF_on_corpus_results[corr_type], return_counts=True)
                plt.plot(a, b)
            plt.xlabel("VIF-type of dimensions used in attribution (in %)")
            plt.ylabel("No. of authors")
            plt.legend([_labels[_x] for _x in VIF_on_corpus_results.keys()])
            plt.show()

        return VIF_on_corpus_results

    def tree_overfit_test(self, train_test_split_ratio, fold, show_plots=True):
        """
        Test to demonstrate that features captured by decision trees are stylistic. Prominent decision paths capture
        most of the training and testing data.
        :param train_test_split_ratio: float, supports 0.25, 0.50 and 0.75
        :param fold: int, selected validation fold
        :param show_plots: bool, displays results from analysis as plots (shown in paper)
        :return: test_output_data, author_best_path_dict
        test_output_data: tuple, plot data
        author_best_path_dict: dict, stores most prominent decision path, author-id: decision-path
        """
        print("Running tree overfit test....")
        _data_file_name = str(train_test_split_ratio)[2:] + "_" + str(fold)

        with open(os.path.join(self.tree_data_path, _data_file_name + "_trainData.json")) as f_json_load:
            train_path_dict = json.load(f_json_load)

        # authors with their decision paths and its frequency
        author_decision_path_and_freq_dict = {_x: [] for _x in self.author_id_list}
        for _decision_path in train_path_dict.keys():
            no_of_samples = len(train_path_dict[_decision_path])
            author_id = train_path_dict[_decision_path][0].split("_")[0]
            author_decision_path_and_freq_dict[author_id].append((_decision_path, no_of_samples))

        # test_file_name: (decision_path_str, true_author_id, predicted_author_id)
        author_correct_path_dict = {_x: [] for _x in self.author_id_list}
        with open(os.path.join(self.tree_data_path, _data_file_name + "_testData.json")) as f_json_load:
            test_data_dict = json.load(f_json_load)

        for test_file_name in test_data_dict.keys():
            decision_path_str, gt_label, predicted_label = test_data_dict[test_file_name]
            author_id = test_file_name.split("_")[0]
            if gt_label == predicted_label:
                author_correct_path_dict[author_id].append(decision_path_str)

        author_tree_fitness_stat_dict = {_x: [] for _x in self.author_id_list}
        test_output_data, author_best_path_dict = [], dict()
        for _author_id in author_decision_path_and_freq_dict.keys():
            for _decision_path_train, no_of_train_samples in author_decision_path_and_freq_dict[_author_id]:
                corresponding_no_of_test_samples = author_correct_path_dict[_author_id].count(_decision_path_train)
                author_tree_fitness_stat_dict[_author_id].append(
                    (no_of_train_samples, corresponding_no_of_test_samples, _decision_path_train))

            train_sample_counts, test_sample_counts, paths = [list(_x) for _x in
                                                              zip(*sorted(author_tree_fitness_stat_dict[_author_id],
                                                                          reverse=True))]
            if sum(train_sample_counts) and sum(test_sample_counts):
                prominent_path_capture_ratio_for_train = round(train_sample_counts[0] / sum(train_sample_counts), 2)
                prominent_path_capture_ratio_for_test = round(test_sample_counts[0] / sum(test_sample_counts), 2)
                test_output_data.append((prominent_path_capture_ratio_for_train, prominent_path_capture_ratio_for_test))
                author_best_path_dict[_author_id] = [int(_x) for _x in paths[0].split(SEPARATOR)]
        print("Tree overfit test complete")

        if show_plots:
            plt.figure()
            for _train_capture_ratio, _test_capture_ratio in test_output_data:
                plt.scatter(_train_capture_ratio, _test_capture_ratio)
            plt.xlabel("Ratio captured by prominent decision path in training")
            plt.ylabel("Ratio captured by prominent decision path in testing")
            plt.show()

        return test_output_data, author_best_path_dict

    def cross_fold_decision_tree_similarity(self, train_test_split_ratio, show_plots=True):
        """
        Analyzes relationship between decision trees generated in various folds. Records KL-divergence between number of
        times a dimension of TraSE is used in deciding identity to the mean number of times the same dimensions
        is used in deciding identity across folds.
        :param train_test_split_ratio: float, supports 0.25, 0.50 and 0.75
        :param show_plots: bool, displays results from analysis as plots (shown in paper)
        :return:
        """
        valid_dims_in_folds = {_fold: [] for _fold in range(0, n_CROSS_VALIDATE)}
        for _fold in range(0, n_CROSS_VALIDATE):
            _data_file_name = str(train_test_split_ratio)[2:] + "_" + str(_fold)
            with open(os.path.join(self.tree_data_path, _data_file_name + "_treeData.json")) as f_json_load:
                tree_data_dict = json.load(f_json_load)
            for _node_id in tree_data_dict.keys():
                _dim = tree_data_dict[_node_id][0]
                if _dim >= 0:
                    valid_dims_in_folds[_fold].append(_dim)
        cum_sum = np.zeros(n_DIM_TraSE)
        fold_counters = []
        for _fold in range(0, n_CROSS_VALIDATE):
            fold_counter = np.zeros(n_DIM_TraSE)
            for _i in range(0, n_DIM_TraSE):
                fold_counter[_i] = valid_dims_in_folds[_fold].count(_i)
                cum_sum[_i] += fold_counter[_i]
            fold_counters.append(fold_counter)

        fold_level_divergence = []
        average_fold_counter = [int(_x / n_CROSS_VALIDATE) for _x in cum_sum]
        norm_average_fold_counter = np.divide(average_fold_counter, sum(average_fold_counter))
        for _fold_counter in fold_counters:
            norm_fold_counter = np.divide(_fold_counter, sum(_fold_counter))
            kl_divergence = sum([_x for _x in kl_div(norm_fold_counter, norm_average_fold_counter) if _x != np.inf])
            fold_level_divergence.append(round(kl_divergence, 3))

        if show_plots:
            plt.figure()
            plt.bar(x=range(0, n_CROSS_VALIDATE), height=fold_level_divergence, align="center")
            plt.xlabel("Cross-validation fold")
            plt.ylabel("KL-divergence (precision=3)")
            plt.show()

        return fold_level_divergence

    @staticmethod
    def _extract_features_from_text(key):
        """
        Custom wrapper for parallel processing. Extracts, encodes and stores TraSE feature vectors.
        :param key: (author-id, parsed-data-entry, output-directory-path)
        :return:
        """
        author_id, parsed_data_entry, output_dir_path = key
        print("Currently processing: ", author_id)
        output_file_path = os.path.join(output_dir_path, author_id + ".txt")

        existing_sample_file_names = []
        if os.path.isfile(output_file_path):
            with open(output_file_path) as f_data:
                for entry in f_data.readlines():
                    existing_sample_file_names.append(entry.split(SEPARATOR)[0])

        _trase_obj_NLM = TraSEFeatureExtractor()
        _trase_eval_NLM = TraSEEval(dataset_path="", working_directory_path="")

        for sample_name, text_data in parsed_data_entry:
            if sample_name not in existing_sample_file_names:
                feature_vector = _trase_obj_NLM.get_feature_vector(text_data)
                encoded_feature_vector = _trase_eval_NLM._sparse_encode_feature_vector(feature_vector)
                with open(output_file_path, "a") as f_author_output:
                    f_author_output.write(SEPARATOR.join([str(sample_name)] + encoded_feature_vector) + "\n")

    def parse_PAN13_AP(self, dataset_path):
        """
        Parse and pre-process PAN13 author profiling dataset
        :param dataset_path: string-like
        :return: dict, author-id_age-label: [(sample-id, text),....,(sample-id, text)]
        """
        parsed_data = dict()
        proxy_label = {"10s": 10, "20s": 20, "30s": 30}
        for index_counter, file_name in enumerate(os.listdir(dataset_path)):
            author_id = file_name.split("_")[0]
            tree = ElementTree.parse(os.path.join(dataset_path, file_name))
            root = tree.getroot()
            label = proxy_label[root.attrib['age_group']]
            text_list = []
            for child in root:
                for grandchildren in child:
                    sample_id = author_id + "_" + grandchildren.attrib['id'] + ".txt"
                    sample = grandchildren.text
                    try:
                        sample_ = str(sample.split("<a href")[0].replace("<br />;", "")).replace("\n", "").replace("\t",
                                                                                                                   "")
                        text_list.append((sample_id, sample_))
                    except AttributeError:
                        pass
            if len(text_list):
                parsed_data[author_id + "_" + str(label)] = text_list
        return parsed_data

    def parse_PAN14_AP(self, dataset_path, corpus_type):
        """
        Parse and pre-process PAN14 author profiling dataset. Supports reviews and socialmedia only
        :param dataset_path: string-like
        :param corpus_type: string-like, "reviews" or "socialmedia"
        :return: dict, author-id_age-label: [(sample-id, text),....,(sample-id, text)]
        """
        parsed_data, age_label_dict = dict(), dict()

        with open(os.path.join(dataset_path, "truth.txt")) as f_labels:
            for entry in f_labels.read().split("\n")[:-1]:
                author_id, _, age_range = entry.split(":::")
                age_bin_min, age_bin_max = age_range.split("-")
                if age_bin_max == "xx" or age_bin_max == "XX":
                    quantized_age = int(round(int(age_bin_min) / 5) * 5)
                else:
                    mean_age_value = (int(age_bin_min) + int(age_bin_max)) / 2
                    quantized_age = int(round(mean_age_value / 5) * 5)
                age_label_dict[author_id] = quantized_age

        if corpus_type == "reviews":
            for file_name in os.listdir(dataset_path):
                author_id, extension = file_name.split(".")
                if extension == "xml":
                    label = age_label_dict[author_id]
                    tree = ElementTree.parse(os.path.join(dataset_path, file_name))
                    root = tree.getroot()
                    text_list = []
                    for child in root:
                        for grandchildren in child:
                            sample_id = author_id + "_" + grandchildren.attrib['id'] + ".txt"
                            sample = grandchildren.text.replace("\n", "").replace("\t", "")
                            text_list.append((sample_id, sample))
                    if len(text_list):
                        parsed_data[author_id + "_" + str(label)] = text_list

        elif corpus_type == "socialmedia":
            for file_name in os.listdir(dataset_path):
                author_id, extension = file_name.split(".")
                if extension == "xml":
                    label = age_label_dict[author_id]
                    tree = ElementTree.parse(os.path.join(dataset_path, file_name))
                    root = tree.getroot()
                    text_list = []
                    for child in root:
                        for grandchildren in child:
                            sample_id = author_id + "_" + grandchildren.attrib['id'] + ".txt"
                            sample = grandchildren.text.replace("\n", "").replace("\t", "")
                            sample_ = ""
                            flag = True
                            for char in sample:
                                if char == "<":
                                    flag = False
                                if flag:
                                    sample_ += str(char)
                                if char == ">":
                                    flag = True
                                    sample_ += " "
                            text_list.append((sample_id, sample_))
                    if len(text_list):
                        parsed_data[author_id + "_" + str(label)] = text_list
        return parsed_data

    def parse_PAN15_AP(self, dataset_path):
        """
        Parse and pre-process PAN15 author profiling dataset
        :param dataset_path: string-like
        :return: dict, author-id_age-label: [(sample-id, text),....,(sample-id, text)]
        """
        parsed_data, age_label_dict = dict(), dict()

        with open(os.path.join(dataset_path, "truth.txt")) as f_labels:
            for entry in f_labels.read().splitlines():
                all_info = entry.split(":::")
                author_id, age_range = all_info[0], all_info[2]
                age_bin_min, age_bin_max = age_range.split("-")
                if age_bin_max == "xx" or age_bin_max == "XX":
                    quantized_age = int(round(int(age_bin_min) / 5) * 5)
                else:
                    mean_age_value = (int(age_bin_min) + int(age_bin_max)) / 2
                    quantized_age = int(round(mean_age_value / 5) * 5)
                age_label_dict[author_id] = quantized_age

        for file_name in os.listdir(dataset_path):
            author_id, extension = file_name.split(".")
            if extension == "xml":
                label = age_label_dict[author_id]
                tree = ElementTree.parse(os.path.join(dataset_path, file_name))
                root = tree.getroot()

                all_text = []
                for index, child in enumerate(root):
                    text_ = preprocessor.clean(child.text)
                    processed_text = text_.translate(str.maketrans('', '', string.punctuation))
                    if len(processed_text):
                        all_text.append((author_id + "_" + str(index) + ".txt", processed_text))

                if len(all_text):
                    parsed_data[author_id + "_" + str(label)] = all_text
        return parsed_data

    def age_vs_style_experiment(self, dataset_path_keys, working_directory_path, core_count, data_integrity_check=True,
                                show_plots=True):
        """
        Test relationship between style and age of human subjects.
        :param dataset_path_keys: tuple, (dataset-path, dataset-name) supports PAN13AP, PAN14AP-socialmedia,
        PAN14AP-reviews, PAN15AP and BAC
        :param working_directory_path: string-like,
        :param core_count: int,
        :param data_integrity_check: bool, if True check if all feature vectors are present. If False, skips check.
        :return: None
        """
        print("Running style variation with age experiment.......")
        if data_integrity_check:
            for dataset_path, corpus_name in dataset_path_keys:
                print("Parsing :", corpus_name)
                output_path = os.path.join(working_directory_path, "TraSE_" + corpus_name)
                if not os.path.isdir(output_path):
                    os.mkdir(output_path)

                if corpus_name == "PAN13AP":
                    keys = []
                    parsed_data = self.parse_PAN13_AP(dataset_path=dataset_path)
                    print("Parse complete. Checking for features......")
                    for author_id in parsed_data.keys():
                        keys.append((author_id, parsed_data[author_id], output_path))
                    with mp.Pool(core_count) as p:
                        p.map(self._extract_features_from_text, keys)
                    print("Feature check complete")

                elif corpus_name == "PAN14AP-socialmedia" or corpus_name == "PAN14AP-reviews":
                    keys = []
                    if corpus_name == "PAN14AP-socialmedia":
                        parsed_data = self.parse_PAN14_AP(dataset_path=dataset_path, corpus_type="socialmedia")
                    else:
                        parsed_data = self.parse_PAN14_AP(dataset_path=dataset_path, corpus_type="reviews")
                    print("Parse complete. Checking for features......")
                    for author_id in parsed_data.keys():
                        keys.append((author_id, parsed_data[author_id], output_path))
                    with mp.Pool(core_count) as p:
                        p.map(self._extract_features_from_text, keys)
                    print("Feature check complete")

                elif corpus_name == "PAN15AP":
                    keys = []
                    parsed_data = self.parse_PAN15_AP(dataset_path=dataset_path)
                    print("Parse complete. Checking for features......")
                    for author_id in parsed_data.keys():
                        keys.append((author_id, parsed_data[author_id], output_path))
                    with mp.Pool(core_count) as p:
                        p.map(self._extract_features_from_text, keys)
                    print("Feature check complete")

                else:
                    print("NOTE: Assumes BAC features are available. If not, extract features separately. "
                          "This function only handles PAN-AP 2013, 2014 (reviews and social media) and 2015.")

        def process_BAC_label(label):
            return int(round(int(label.split(".")[2]) / 5) * 5)

        def process_PAN_label(label):
            return int(label[:-4].split("_")[-1])

        cmp = plt.figure()
        size = 6
        plt.rc('font', size=size)
        plt.rc('axes', titlesize=size)
        plt.rc('axes', labelsize=size)
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        cmp.set_size_inches(2.5, 1.25)
        _, plot_legend = zip(*dataset_path_keys)
        for dataset_path, corpus_name in dataset_path_keys:
            print("Analyzing :", corpus_name)
            age_axis = {quantized_age_value: [] for quantized_age_value in range(5, 71, 5)}
            feature_path = os.path.join(working_directory_path, "TraSE_" + corpus_name)
            if corpus_name == "BAC":
                for author_id in os.listdir(dataset_path):
                    age_value = process_BAC_label(author_id)
                    if os.path.isfile(os.path.join(feature_path, author_id + ".txt")):
                        with open(os.path.join(feature_path, author_id + ".txt")) as f_feature_load:
                            for entry in f_feature_load.readlines():
                                sample = self._sparse_decode_feature_vector(entry.split(SEPARATOR)[1:])
                                if sample is not None:
                                    age_axis[age_value].append(sample)
            else:
                for file_name in os.listdir(feature_path):
                    age_value = process_PAN_label(file_name)
                    with open(os.path.join(feature_path, file_name)) as f_feature_load:
                        for entry in f_feature_load.readlines():
                            sample = self._sparse_decode_feature_vector(entry.split(SEPARATOR)[1:])
                            if sample is not None:
                                age_axis[age_value].append(sample)

            x_values, y_values = [], []
            for _age in sorted(age_axis.keys()):
                if len(age_axis[_age]):
                    x_values.append(_age)
                    cov_matrix = np.abs(np.cov(np.transpose(age_axis[_age])))
                    total_sum = np.sum(cov_matrix)
                    trace = np.sum(cov_matrix.diagonal())
                    off_diagonal_sum = total_sum - trace
                    y_values.append(off_diagonal_sum / total_sum)

            if show_plots:
                plt.plot(x_values, y_values)

        if show_plots:
            plt.savefig(os.path.join(working_directory_path, "small_version.png"), dpi=500)
            cmp.set_size_inches(5, 5)
            plt.xlabel("Age")
            plt.ylabel("Relative covariance density (in %)")
            plt.legend(list(plot_legend))
            plt.savefig(os.path.join(working_directory_path, "big_version.png"), dpi=500)
            plt.show()

    def evaluate_on_cross_topic_protocol(self, dataset_path, feature_directory_path, output_path, split=0.75,
                                         fold_select=-1):
        """
        Evaluates on cross-topic protocol. Display and save classification results.
        :param dataset_path: string-like
        :param feature_directory_path: string-like
        :param output_path: string-like
        :param split: float, train-test split ratio supports 0.25, 0.5 and 0.75
        :param fold_select: -1 for cross validation or int for specific fold
        :return: None
        """
        print("Running cross-topic evaluation protocol.....")
        print("Train-test split ratio: " + str(split))

        if fold_select == -1:
            folds = range(0, n_CROSS_VALIDATE)
        else:
            folds = [fold_select]

        # generate cross-topic standalone topics cross-validation json file
        _cv_config_dict = dict()
        for author_id in os.listdir(dataset_path):
            file_names = os.listdir(os.path.join(dataset_path, author_id))
            unique_topic_list = list(set([_x.split("_")[1] for _x in file_names]))
            if len(unique_topic_list) >= n_CROSS_VALIDATE:
                _cv_config_dict[author_id] = []
                shuffle(unique_topic_list)
                topic_splits = np.array_split(unique_topic_list, n_CROSS_VALIDATE)
                for selected_topics in topic_splits:
                    split_file_names = [file_name for file_name in file_names if
                                        file_name.split("_")[1] in selected_topics]
                    _cv_config_dict[author_id].append(split_file_names)

        for fold in folds:
            print("Fold: ", fold)

            _trase_eval_obj = TraSEEval(dataset_path="", working_directory_path=feature_directory_path)
            train_files, test_files, _, _ = _trase_eval_obj.get_data(cv_config_json=_cv_config_dict, split=split,
                                                                     fold=fold)
            train_data, train_labels = zip(*train_files)
            test_data, test_labels = zip(*test_files)
            print("Data splits....")
            print("Train files: ", len(train_labels), " Test files: ", len(test_labels))
            print("Training model")
            trained_model = DecisionTreeClassifier(random_state=0).fit(list(train_data), list(train_labels))
            print("Making predictions")
            predictions = trained_model.predict(list(test_data))
            acc, f1, precision, recall, log_text = self.generate_classification_report(test_labels, predictions)
            print("Saving classification report")
            output_file_name = "cross-topic-protocol_" + str(split)[2:] + "_" + str(fold)
            with open(os.path.join(output_path, output_file_name + "_report.txt"), "w") as f_logger:
                f_logger.write(log_text)

    def evaluate_on_cross_genre_protocol(self, dataset_path, feature_directory_path, corpus_name="Guardian10"):
        """
        Evaluates on cross-genre protocol. Displays classification results.
        :param dataset_path: string-like,
        :param feature_directory_path: string-like,
        :param corpus_name: Only supports Guardian10 for now. Train genre: Newspaper articles, Test genre: Books
        :return: None
        """
        if corpus_name == "Guardian10":
            _cv_config_dict = dict()
            for author_id in os.listdir(dataset_path):
                train_files_names, test_files_names = [], []
                for file_name in os.listdir(os.path.join(dataset_path, author_id)):
                    _, genre, sample_id = file_name[:-4].split("_")
                    if genre == "Books":
                        test_files_names.append(file_name)
                    else:
                        train_files_names.append(file_name)
                splits = np.array_split(train_files_names, n_CROSS_VALIDATE - 1)
                _cv_config_dict[author_id] = [splits[_x].tolist() for _x in range(0, n_CROSS_VALIDATE - 1)] + [
                    test_files_names]
            _trase_eval_obj = TraSEEval(dataset_path="", working_directory_path=feature_directory_path)
            train_files, test_files, _, _ = _trase_eval_obj.get_data(cv_config_json=_cv_config_dict, split=0.75, fold=0)
            train_data, train_labels = zip(*train_files)
            test_data, test_labels = zip(*test_files)
            print("Data splits....")
            print("Train files: ", len(train_labels), " Test files: ", len(test_labels))
            print("Training model")
            trained_model = DecisionTreeClassifier(random_state=0).fit(list(train_data), list(train_labels))
            print("Making predictions")
            predictions = trained_model.predict(list(test_data))
            acc, f1, precision, recall, log_text = self.generate_classification_report(test_labels, predictions)
        else:
            print("Function not available")

    def evaluate_on_all_inclusive_protocol(self, feature_directory_path, output_path, corpus_list, split,
                                           fold_select=-1):
        """
        All-inclusive protocol. Trains and evaluates after aggregating authors and samples from multiple corpora.
        Displays classification results and logs essential decision tree statistics.
        :param feature_directory_path: string-like
        :param output_path: string-like
        :param corpus_list: list, names of corpora used in earlier experiments
        :param split: train-test split ratio supports 0.25, 0.5 and 0.75
        :param fold_select: -1 for all folds and int for specific fold
        :return: None
        """
        print("Running all-inclusive test protocol.....")
        print("Train-test split ratio: " + str(split))

        if fold_select == -1:
            folds = range(0, n_CROSS_VALIDATE)
        else:
            folds = [fold_select]

        for fold in folds:
            print("Fold: ", fold)
            label_counter, train_files, test_files, test_file_names = 0, [], [], []
            proxy_label_to_author_id_dict, proxy_label_to_corpus_name_dict = dict(), dict()
            for corpus_name in corpus_list:
                print("Current corpus: ", corpus_name)
                feature_path = os.path.join(feature_directory_path, "TraSE_" + corpus_name)

                _trase_eval_obj = TraSEEval(dataset_path="", working_directory_path=feature_path)

                with open(os.path.join(feature_path, "crossValFolds.json")) as f_json_load:
                    _cv_config_dict = json.load(f_json_load)

                label_to_proxy_label = dict()
                for label, author_id in enumerate(_cv_config_dict.keys()):
                    label_to_proxy_label[label] = label_counter
                    proxy_label_to_author_id_dict[label_counter] = author_id
                    proxy_label_to_corpus_name_dict[label_counter] = corpus_name
                    label_counter += 1

                _train, _test, _, _test_file_names = _trase_eval_obj.get_data(cv_config_json=_cv_config_dict,
                                                                              split=split, fold=fold)
                for sample, label in _train:
                    train_files.append((sample, label_to_proxy_label[label]))
                for sample, label in _test:
                    test_files.append((sample, label_to_proxy_label[label]))
                test_file_names += _test_file_names
            train_data, train_labels = zip(*train_files)
            test_data, test_labels = zip(*test_files)
            print("Data splits....")
            print("Train files: ", len(train_labels), " Test files: ", len(test_labels))
            print("Training model")
            trained_model = DecisionTreeClassifier(random_state=0).fit(list(train_data), list(train_labels))
            print("Making predictions")
            predictions = trained_model.predict(list(test_data))
            acc, f1, precision, recall, log_text = self.generate_classification_report(test_labels, predictions)

            classifier_test_log = []
            for _i in range(0, len(test_file_names)):
                current_corpus_name = proxy_label_to_corpus_name_dict[test_labels[_i]]
                current_test_file = test_file_names[_i]
                gt_label, gt_author = str(test_labels[_i]), proxy_label_to_author_id_dict[test_labels[_i]]
                predicted_label, predicted_author = str(predictions[_i]), proxy_label_to_author_id_dict[predictions[_i]]
                test_entry = SEPARATOR.join([current_corpus_name, current_test_file, gt_author, gt_label,
                                             predicted_author, predicted_label])
                classifier_test_log.append(test_entry)

            n_nodes = trained_model.tree_.node_count
            children_left = trained_model.tree_.children_left
            children_right = trained_model.tree_.children_right
            feature = trained_model.tree_.feature
            threshold = trained_model.tree_.threshold
            tree_structure_dict = {node_id: (int(feature[node_id]), float(threshold[node_id]),
                                             int(children_left[node_id]), int(children_right[node_id])) for node_id in
                                   range(n_nodes)}

            print("Saving classification report")
            output_file_name = "all-inclusive_" + str(split)[2:] + "_" + str(fold)
            with open(os.path.join(output_path, output_file_name + "_report.txt"), "w") as f_logger:
                f_logger.write(log_text)

            with open(os.path.join(output_path, output_file_name + "_treeData.json"), "w") as f_json:
                json.dump(tree_structure_dict, f_json)

            with open(os.path.join(output_path, output_file_name + "_testLog.txt"), "w") as f_logger:
                f_logger.write("\n".join(classifier_test_log) + "\n")


class TraSEUtils:
    def __init__(self):
        """
        Support utlities for TraSE
        """
        pass

    @staticmethod
    def aggregate_feature_vectors_for_transfer(working_directory_path, corpus_name_list, output_path):
        """
        Aggregate all feature vectors for faster transfer
        :param working_directory_path: string-like
        :param corpus_name_list: list, corpus names
        :param output_path: string-like,
        :return: None
        """

        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        else:
            rmtree(output_path)
            os.mkdir(output_path)

        for corpus_name in corpus_name_list:
            print("Aggregating :", corpus_name)
            feature_directory_path = os.path.join(working_directory_path, "TraSE_" + corpus_name)
            with open(os.path.join(output_path, corpus_name + "_allFeaturesCompressed.txt"), "a") as f_text_dump:
                for file_name in os.listdir(feature_directory_path):
                    if "_" not in file_name and file_name.split(".")[-1] == "txt":
                        with open(os.path.join(feature_directory_path, file_name)) as f_text_load:
                            f_text_dump.write(f_text_load.read())

            with open(
                    os.path.join(working_directory_path, "TraSE_" + corpus_name, "crossValFolds.json")) as f_json_load:
                data = json.load(f_json_load)

            with open(os.path.join(output_path, corpus_name + ".json"), "w") as f_json_dump:
                json.dump(data, f_json_dump)
            print("Done")

    @staticmethod
    def restructure_feature_vectors_after_transfer(working_directory, output_directory):
        """
        Reorganizes aggregated files after transfer into template form for evaluation and running experiments
        :param working_directory: string-like, aggregated data folder path
        :param output_directory: string-like, data restructured at this path
        :return: None
        """
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

        for possible_compressed_corpus_file_name in os.listdir(working_directory):
            if "allFeaturesCompressed" in possible_compressed_corpus_file_name:
                corpus_name = possible_compressed_corpus_file_name.split("_")[0]
                print("Restructuring: ", corpus_name)
                decompress_path = os.path.join(output_directory, "TraSE_" + corpus_name)
                if not os.path.isdir(decompress_path):
                    os.mkdir(decompress_path)
                _data_buffer = dict()
                with open(os.path.join(working_directory, possible_compressed_corpus_file_name)) as f_load:
                    for entry in f_load.readlines():
                        data = entry.split(SEPARATOR)
                        author_id = data[0].split("_")[0]
                        try:
                            _data_buffer[author_id].append(entry)
                        except KeyError:
                            _data_buffer[author_id] = [entry]
                for author_id in _data_buffer.keys():
                    with open(os.path.join(decompress_path, author_id + ".txt"), "a") as f_dump:
                        for entry in list(set(_data_buffer[author_id])):
                            f_dump.write(entry)

                with open(os.path.join(working_directory, corpus_name + ".json")) as f_json_load:
                    data = json.load(f_json_load)

                with open(os.path.join(decompress_path, "crossValFolds.json"), "w") as f_json_dump:
                    json.dump(data, f_json_dump)
                print("Done")

    @staticmethod
    def _setup_plot_format_for_paper():
        """
        Setup plot format for paper
        :return: matplotlib figure
        """
        cmp = plt.figure()
        size = 6
        plt.rc('font', size=size)
        plt.rc('axes', titlesize=size)
        plt.rc('axes', labelsize=size)
        plt.rc('xtick', labelsize=size)
        plt.rc('ytick', labelsize=size)
        return cmp

    def plot_multi_collinearity_analysis_for_multiple_corpus(self, keys, train_test_ratio, target_fold, save_path):
        """
        Combine and plot VIF test results for multiple corpus
        :param keys: tuple, (dataset-path, working-dir-path) datasets to plot results for
        :param train_test_ratio: float, supports 0.25, 0.5 and 0.75
        :param target_fold: int, targeted fold in cross-validation
        :param save_path: string, path to save plot
        :return: None
        """
        VIF_on_all_corpora_results = {"not_correlated": [], "moderately_correlated": [], "highly_correlated": []}
        for _data_path, _result_path, _corpus_name in keys:
            print("Currently processing: ", _corpus_name)
            VIF_on_corpus_results = TraSEExperiments(dataset_path=_data_path,
                                                     working_directory_path=_result_path).multi_collinearity_analysis(
                train_test_split_ratio=train_test_ratio, fold=target_fold, show_plots=False)

            for corr_type in VIF_on_all_corpora_results.keys():
                VIF_on_all_corpora_results[corr_type] += VIF_on_corpus_results[corr_type]

        cmp = self._setup_plot_format_for_paper()
        cmp.set_size_inches(2.5, 1.25)
        _labels = {"not_correlated": "Not correlated",
                   "moderately_correlated": "Moderately correlated",
                   "highly_correlated": "Highly correlated"}
        for corr_type in VIF_on_all_corpora_results.keys():
            a, b = np.unique(VIF_on_all_corpora_results[corr_type], return_counts=True)
            plt.plot(a, b)
        plt.savefig(os.path.join(save_path, "small_version.png"), dpi=500)
        cmp.set_size_inches(5, 5)
        plt.xlabel("VIF-type of dimensions used in attribution (in %)")
        plt.ylabel("No. of authors")
        plt.legend([_labels[_x] for _x in VIF_on_all_corpora_results.keys()])
        plt.savefig(os.path.join(save_path, "big_version.png"), dpi=500)
        plt.show()

    def plot_tree_overfit_test_for_multiple_corpus(self, keys, train_test_ratio, target_fold, save_path):
        """
        Combine and plot tree overfit test results for multiple corpus
        :param keys: tuple, (dataset-path, working-dir-path) datasets to plot results for
        :param train_test_ratio: float, supports 0.25, 0.5 and 0.75
        :param target_fold: int, targeted fold in cross-validation
        :param save_path: string, path to save plot
        :return: None
        """
        data_to_plot = []
        for _data_path, _result_path, _corpus_name in keys:
            print("Currently processing: ", _corpus_name)
            _test_output_data, _ = TraSEExperiments(dataset_path=_data_path,
                                                    working_directory_path=_result_path).tree_overfit_test(
                train_test_split_ratio=train_test_ratio, fold=target_fold, show_plots=False)
            data_to_plot += _test_output_data

        cmp = self._setup_plot_format_for_paper()
        cmp.set_size_inches(2.5, 1.25)
        for _train_capture_ratio, _test_capture_ratio in data_to_plot:
            plt.scatter(_train_capture_ratio, _test_capture_ratio, c='C0', s=1)

        plt.savefig(os.path.join(save_path, "small_version.png"), dpi=500)
        cmp.set_size_inches(5, 5)
        plt.xlabel("Ratio captured by prominent decision path in training")
        plt.ylabel("Ratio captured by prominent decision path in testing")
        plt.savefig(os.path.join(save_path, "big_version.png"), dpi=500)
        plt.show()

    @staticmethod
    def run_evaluation_on_multiple_classifiers(dataset_path, working_directory_path, train_test_split_ratio,
                                               fold_select):
        """
        Save evaluation results using a battery of classifiers
        :param dataset_path: string, path to dataset
        :param working_directory_path: string, working directory path
        :param train_test_split_ratio: float, supports 0.25, 0.5 and 0.75
        :param fold_select: int, targeted fold in cross-validation
        :return: None
        """
        _trase_eval_obj = TraSEEval(dataset_path=dataset_path, working_directory_path=working_directory_path)

        cv_config_path = os.path.join(working_directory_path, "crossValFolds.json")
        with open(cv_config_path, "r") as fp:
            cv_folds = json.load(fp)

        train_files, test_files, train_file_names, test_file_names = _trase_eval_obj.get_data(cv_folds,
                                                                                              train_test_split_ratio,
                                                                                              fold_select)
        train_data, train_labels = zip(*train_files)
        test_data, test_labels = zip(*test_files)
        print("Data splits....")
        print("Train files: ", len(train_labels), " Test files: ", len(test_labels))

        classifier_list = [LogisticRegression(random_state=0), LinearSVC(random_state=0), MultinomialNB(), GaussianNB(),
                           ExtraTreesClassifier(random_state=0), RandomForestClassifier(random_state=0),
                           AdaBoostClassifier(random_state=0), GradientBoostingClassifier(random_state=0),
                           HistGradientBoostingClassifier(random_state=0), SGDClassifier(random_state=0)]

        all_classifier_log = ""
        for classifier in classifier_list:
            clf_name = str(classifier).split("(")[0]
            print("Training model: ", clf_name)
            all_classifier_log += "Classifier: " + clf_name + "\n"
            model = classifier.fit(list(train_data), list(train_labels))
            print("Making predictions")
            predictions = model.predict(list(test_data))
            acc, f1, precision, recall, log_text = _trase_eval_obj.generate_classification_report(test_labels,
                                                                                                  predictions)
            all_classifier_log += log_text
            print("Evaluation on " + clf_name + " complete")

        with open(os.path.join(working_directory_path, 'AllClassifierLogs.txt'), 'w') as f_log_dump:
            f_log_dump.write(all_classifier_log)

    @staticmethod
    def display_corpus_level_performance_scores_for_all_inclusive_evaluation(working_directory_path):
        """
        Displays corpus-level performance scores for all inclusive evaluation
        :param working_directory_path: string, path to saved data from all-inclusive analysis
        :return: None
        """
        print("Displaying corpus-level performance measures for all-inclusive criteria.......")
        split_ratios = [0.25, 0.5, 0.75]
        for _split_ratio in split_ratios:
            for _fold in range(0, n_CROSS_VALIDATE):
                print("Split-ratio: ", str(_split_ratio), "Fold: ", str(_fold))
                log_file_name = "_".join(["all-inclusive", str(_split_ratio).split(".")[-1], str(_fold), "testLog.txt"])
                with open(os.path.join(working_directory_path, log_file_name)) as f_log_read:
                    corpus_gt_labels, corpus_predictions = dict(), dict()
                    for _entry in f_log_read.readlines():
                        _corpus_name, _, _, _gt_label, _, _predicted_label = _entry.split(SEPARATOR)
                        try:
                            corpus_gt_labels[_corpus_name].append(int(_gt_label))
                            corpus_predictions[_corpus_name].append(int(_predicted_label))
                        except KeyError:
                            corpus_gt_labels[_corpus_name] = [int(_gt_label)]
                            corpus_predictions[_corpus_name] = [int(_predicted_label)]
                for _corpus_name in corpus_gt_labels.keys():
                    all_corpus_gt_labels = {_x: None for _x in np.unique(corpus_gt_labels[_corpus_name])}
                    corpus_predicted_labels = []
                    for label in corpus_predictions[_corpus_name]:
                        try:
                            _ = all_corpus_gt_labels[label]
                            corpus_predicted_labels.append(label)
                        except KeyError:
                            corpus_predicted_labels.append(-1)  # out of corpus
                    acc = round(accuracy_score(corpus_gt_labels[_corpus_name], corpus_predictions[_corpus_name]), 2)
                    f1 = round(f1_score(corpus_gt_labels[_corpus_name], corpus_predicted_labels, average="macro"), 2)
                    print("Corpus: ", _corpus_name, "Accuracy: ", acc, " macro-F1: ", f1)


if __name__ == "__main__":
    # # EXPT: Standalone protocol
    # dataset_name = "Guardian10"
    # data_path = "C:/work_stuff/datasets/" + dataset_name
    # result_path = "C:/work_stuff/TraSE_" + dataset_name
    # trase = TraSEEval(dataset_path=data_path, working_directory_path=result_path, use_multiprocessing=10)
    # # trase.extract_features(overwrite_existing_data=False)
    # trase.train_and_evaluate_on_dataset(train_test_split_ratio=0.75)

    # # EXPT: All-inclusive protocol
    # trase_exp = TraSEExperiments(dataset_path="", working_directory_path="")
    # trase_exp.evaluate_on_all_inclusive_protocol(feature_directory_path=extract_to_path, output_path=extract_to_path,
    #                                              corpus_list=corpus_list, split=0.75, fold_select=0)

    # # EXPT: Cross-topic protocol
    # trase_exp = TraSEExperiments(dataset_path="", working_directory_path="")
    # dataset_name = "Guardian10"
    # data_path = "C:/work_stuff/datasets/" + dataset_name
    # result_path = "C:/work_stuff/TraSE_" + dataset_name
    # trase_exp.evaluate_on_cross_topic_protocol(dataset_path=data_path, feature_directory_path=result_path,
    #                                            output_path=result_path, split=0.75)

    # # EXPT: Cross-genre protocol
    # dataset_name = "Guardian10"
    # data_path = "C:/work_stuff/datasets/" + dataset_name
    # result_path = "C:/work_stuff/TraSE_" + dataset_name
    # trase_exp = TraSEExperiments(dataset_path="", working_directory_path="")
    # trase_exp.evaluate_on_cross_genre_protocol(dataset_path=data_path, feature_directory_path=result_path,
    #                                            corpus_name=dataset_name)

    # # EXPT: PAN protocol (cross-domain)
    # data_path = "C:/work_stuff/datasets/PAN/AA/PAN19/problem00005"
    # result_path = "C:/work_stuff/TraSE_" + "PAN19_problem00005"
    # trase_PAN = StandardizePANdataset(original_PAN_dataset_path=data_path, working_directory_path=result_path, options="ground_truth", use_multiprocessing=7)
    # # trase_PAN.extract_features()
    # trase_PAN.train_and_evaluate_on_dataset()

    # # EXPT: Classifier analysis
    # dataset_name = "Guardian10"
    # data_path = "C:/work_stuff/datasets/" + dataset_name
    # result_path = "C:/work_stuff/TraSE_" + dataset_name
    # train_test_ratio, target_fold = 0.75, 0
    # trase_exp = TraSEExperiments(dataset_path=data_path, working_directory_path=result_path)
    # # Q: Why decision trees only?
    # trase_exp.multi_collinearity_analysis(train_test_split_ratio=train_test_ratio, fold=target_fold, show_plots=True)
    # # Q: Are decision trees truly capturing style? Are they overfitting?
    # trase_exp.tree_overfit_test(train_test_split_ratio=train_test_ratio, fold=target_fold, show_plots=True)
    # # Q: Is style generalizable across folds?
    # trase_exp.cross_fold_decision_tree_similarity(train_test_split_ratio=train_test_ratio, show_plots=True)

    # # EXPT: Age-Style relation
    # trase_exp = TraSEExperiments(dataset_path="", working_directory_path="")
    # dataset_keys = []
    # dataset_keys.append(("C:/work_stuff/datasets/BAC", "BAC"))
    # dataset_keys.append(("C:/work_stuff/datasets/PAN/AP/PAN13/train/en", "PAN13AP"))
    # dataset_keys.append(("C:/work_stuff/datasets/PAN/AP/PAN14/train/socialmedia", "PAN14AP-socialmedia"))
    # dataset_keys.append(("C:/work_stuff/datasets/PAN/AP/PAN14/train/reviews", "PAN14AP-reviews"))
    # dataset_keys.append(("C:/work_stuff/datasets/PAN/AP/PAN15/train", "PAN15AP"))
    # working_dir_path = "C:/work_stuff"
    # trase_exp.age_vs_style_experiment(dataset_path_keys=dataset_keys, working_directory_path=working_dir_path,
    #                                   core_count=10, data_integrity_check=False)
    # -----------------------------------------------------------------------------------------------------------------
    pass

