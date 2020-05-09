import os
import pickle

files = os.listdir('.')

for f in files:
	if '.pkl' in f:
		print(f)
		file_handler = open(f, 'rb+')
		train_acc, test_acc = pickle.load(file_handler)
		file_handler.close()
		print("MAX ACC: ", max(test_acc))
		print("MAX AT: ", test_acc.index(max(test_acc)))
		print("AVG ACC: ", sum(test_acc)/len(test_acc))
		print()

'''
➜  terminal_logs python3 accuracies.py   
cnn_filtered.pkl
MAX ACC:  72.913
MAX AT:  53
AVG ACC:  67.55296999999995

cnn_raw.pkl
MAX ACC:  72.076
MAX AT:  36
AVG ACC:  67.57264

'''
