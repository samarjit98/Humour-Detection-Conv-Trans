import os
import pickle
import matplotlib.pyplot as plt

files = os.listdir('.')

for f in files:
	if '.pkl' in f:
		file_handler = open(f, 'rb+')
		train_acc, test_acc = pickle.load(file_handler)
		train_acc = train_acc[:65]
		test_acc = test_acc[:65]
		file_handler.close()
		train_x = list(range(1, len(train_acc) + 1))
		test_x = list(range(1, len(test_acc) + 1))
		plt.figure()
		#plt.ylim(20, 110)
		plt.ylim(0, 100)
		plt.plot(train_x, train_acc, label = "Train Accuracy")
		plt.plot(test_x, test_acc, label = "Test Accuracy") 
		plt.xlabel('Epoch')
		plt.xlabel('Accuracy')
		plt.legend()
		plt.savefig(os.path.join("./", f[0:-4] + ".png"))
