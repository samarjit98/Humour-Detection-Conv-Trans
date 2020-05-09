from torch.utils.data import Dataset
import numpy as np 
import torch
import os
import pickle
import re
from scipy import stats
from sklearn.model_selection import train_test_split

MAX_LEN = 200

def token(sentence):
    tokens = []
    for t in re.findall("[a-zA-Z0-9]+",sentence.lower()):
        tokens.append(t)
    return tokens

char2num = {}
num2char = {}

def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

class HumorDataset(Dataset):
    def __init__(self, 
                    input_file='./data/data_humor.txt', 
                    load=False, 
                    test=False):
        super(HumorDataset, self).__init__()

        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        self.test = test
        
        if(load):
            file_content = open(input_file, 'r+').read().split('\n')

            dataset_stats = {
                    'sentences': 0, 
                    '0': 0, 
                    '1': 0, 
                }

            char_sequences = []
            labels = []
            for sentence in file_content:
                sentence = sentence.split('\t')

                if(len(sentence) < 2):
                    continue
                '''
                tokens = token(sentence[0])
                label = label2num[sentence[1]]
                '''
                tokens = sentence[0].lower().split(" ")
                print(tokens)
                label = int(sentence[1])

                char_sequence = []
                for tok in tokens:
                    # char_sequence.append("#")
                    for char in tok:
                        char_sequence.append(char)
                    # char_sequence.append("#")
                    char_sequence.append(' ')

                if(len(char_sequence) > MAX_LEN):
                    continue

                dataset_stats['sentences'] += 1
                dataset_stats[sentence[1]] += 1
                
                char_sequences.append(char_sequence)
                labels.append(label)

            print(dataset_stats)

            characters = set()
            for char_sequence in char_sequences:
                for char in char_sequence:
                    characters.add(char)
            characters = sorted(list(characters))

            print("Characters: {}".format(len(characters)))
            print(characters)

            char_num = 0
            for char in characters:
                char2num[char] = char_num
                num2char[char_num] = char
                char_num += 1

            for char_sequence in char_sequences:
                num_sequence = []
                for char in char_sequence:
                    num_sequence.append(char2num[char])
                self.train_data.append(num_sequence)

            self.train_data = np.array(self.train_data)
            self.train_label = np.array(labels)

            self.train_data, self.test_data, self.train_label, self.test_label \
                                                        = train_test_split(self.train_data, 
                                                                            self.train_label, 
                                                                            test_size=0.2, 
                                                                            random_state=42)

            file_handler = open('./pickle/train_set.pkl', 'wb+')
            pickle.dump((self.train_data, self.train_label), file_handler)
            file_handler.close()

            file_handler = open('./pickle/test_set.pkl', 'wb+')
            pickle.dump((self.test_data, self.test_label), file_handler)
            file_handler.close()

            file_handler = open('./pickle/mapping_char2num.pkl', 'wb+')
            pickle.dump(char2num, file_handler)
            file_handler.close()

            file_handler = open('./pickle/mapping_num2char.pkl', 'wb+')
            pickle.dump(num2char, file_handler)
            file_handler.close()

        else:
            if(not self.test):
                file_handler = open('./pickle/train_set.pkl', 'rb+')
                self.train_data, self.train_label = pickle.load(file_handler)
                file_handler.close()
            else:
                file_handler = open('./pickle/test_set.pkl', 'rb+')
                self.test_data, self.test_label = pickle.load(file_handler)
                file_handler.close()

    def __len__(self):
        if(not self.test):
            return len(self.train_data)
        else:
            return len(self.test_data) 
    
    def __getitem__(self, idx):
        if(not self.test):
            inputs, targets = torch.tensor(self.train_data[idx]).type(torch.FloatTensor), \
                                torch.tensor(self.train_label[idx]).type(torch.LongTensor)
            inputs = pad_tensor(inputs, MAX_LEN, 0).type(torch.LongTensor)
            return inputs, targets
        else:
            inputs, targets = torch.tensor(self.test_data[idx]).type(torch.FloatTensor), \
                                torch.tensor(self.test_label[idx]).type(torch.LongTensor)
            inputs = pad_tensor(inputs, MAX_LEN, 0).type(torch.LongTensor)
            return inputs, targets
        

if __name__ == '__main__':
    HumorDataset(load=True)
        
'''
{'sentences': 3418, '0': 1681, '1': 1737}
Characters: 67
[' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '>', '?', '@', '[', '\\', ']', '^', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

'''
