import os
import pickle

files = os.listdir('.')

for f in files:
	if '.pkl' in f and 'loss' not in f:
		print(f)
		file_handler = open(f, 'rb+')
		train_acc, test_acc = pickle.load(file_handler)
		file_handler.close()
		print("MAX ACC: ", max(test_acc))
		print("MAX AT: ", test_acc.index(max(test_acc)))
		print("AVG ACC: ", sum(test_acc)/len(test_acc))
		print()
