# HYBRID DATA PREPARATION, SHUFFLE!
# [np.array([ordinal], [PI], [label])], [same]

import pandas as pd
import numpy as np
from tqdm import tqdm


class DataPreparation():
    desired_length = 1024 # 32*32. DNA sequence length. Input length must not exceed 1024.
    
    TRAIN_POS = "D:/Projects/ORI/Trainig data/PREPARED/1.4/TRAIN_POS_1.4.fa"
    TRAIN_NEG = "D:/Projects/ORI/Trainig data/PREPARED/1.4/TRAIN_NEG_1.5.fa" 
    TEST_POS = "D:/Projects/ORI/Trainig data/PREPARED//1.4/TEST_POS_1.4.fa"
    TEST_NEG_1 = "D:/Projects/ORI/Trainig data/PREPARED/1.4/TEST_NEG_1_1.4.fa"
    TEST_NEG_2 = "D:/Projects/ORI/Trainig data/PREPARED/1.4/TEST_NEG_2_1.5.fa"
   
    
    LABELS_TRAIN = {TRAIN_POS: 1, TRAIN_NEG: 0}
    LABELS_TEST_1 = {TEST_POS: 1, TEST_NEG_1: 0}
    LABELS_TEST_2 = {TEST_POS: 1, TEST_NEG_2: 0}
    
    training_data = []
    testing_data_1 = []
    testing_data_2 = []


    def ordinal_encoding(self, sequence):
        """Takes RAW DNA sequence as input, returns np.array with real numbers for each nucleotide
        """
        code = {"A": 0.25, "C": 0.50, "G": 0.75, "T": 1.00, "N": 0.00}
        return np.array([code[x] for x in sequence])

    def make_training_data(self):
        for label in self.LABELS_TRAIN:
            with open(label, "r") as file:
                for line in tqdm(file):
                    if not line.startswith(">"):
                        line = line[:-1].upper()
                        if len(line) < self.desired_length:
                            line_ext = line.ljust(self.desired_length, "N")
                            self.training_data.append([self.ordinal_encoding(line_ext), self.pi_value2(line), np.eye(2)[self.LABELS_TRAIN[label]]])
                        elif len(line) == self.desired_length:
                            self.training_data.append([self.ordinal_encoding(line), self.pi_value2(line), np.eye(2)[self.LABELS_TRAIN[label]]])

        np.random.shuffle(self.training_data)
        np.save("D:/Projects/ORI/hybrid_train_1.5.npy", self.training_data)
        print("Number of training samples: ", len(self.training_data))

    def make_testing_data_1(self):
        for label in self.LABELS_TEST_1:
            with open(label, "r") as file:
                for line in tqdm(file):
                    if not line.startswith(">"):
                        line = line[:-1].upper()
                        if len(line) < self.desired_length:
                            line_ext = line.ljust(self.desired_length, "N")
                            self.testing_data_1.append([self.ordinal_encoding(line_ext), self.pi_value2(line), np.eye(2)[self.LABELS_TEST_1[label]]])
                        elif len(line) == self.desired_length:
                            self.testing_data_1.append([self.ordinal_encoding(line), self.pi_value2(line), np.eye(2)[self.LABELS_TEST_1[label]]])

        np.random.shuffle(self.testing_data_1)
        np.save("D:/Projects/ORI/hybrid_test_1_1.5.npy", self.testing_data_1)
        print("Number of testing_1 samples: ", len(self.testing_data_1))


    def make_testing_data_2(self):
        for label in self.LABELS_TEST_2:
            with open(label, "r") as file:
                for line in tqdm(file):
                    if not line.startswith(">"):
                        line = line[:-1].upper()
                        if len(line) < self.desired_length:
                            line_ext = line.ljust(self.desired_length, "N")
                            self.testing_data_2.append([self.ordinal_encoding(line_ext), self.pi_value2(line), np.eye(2)[self.LABELS_TEST_2[label]]])
                        elif len(line) == self.desired_length:
                            self.testing_data_2.append([self.ordinal_encoding(line), self.pi_value2(line), np.eye(2)[self.LABELS_TEST_2[label]]])

        np.random.shuffle(self.testing_data_2)
        np.save("D:/Projects/ORI/hybrid_test_2_1.5.npy", self.testing_data_2)
        print("Number of testing_2 samples: ", len(self.testing_data_2))
    

data_train = DataPreparation().make_training_data()
data_test_1 = DataPreparation().make_testing_data_1()
data_test_2 = DataPreparation().make_testing_data_2()
print("DONE")