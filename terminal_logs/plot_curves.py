import os
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

files = os.listdir('.')

for f in files:
	if '.pkl' in f and 'loss' not in f:
		file_handler = open(f, 'rb+')
		train_acc, test_acc = pickle.load(file_handler)
		train_acc = gaussian_filter1d(train_acc[:185], sigma=1)
		test_acc = gaussian_filter1d(test_acc[:185], sigma=1)
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

	if '.pkl' in f and 'loss' in f:
		file_handler = open(f, 'rb+')
		train_loss, test_loss = pickle.load(file_handler)
		file_handler.close()

		train_loss = gaussian_filter1d(train_loss[:180], sigma=1)
		test_loss = gaussian_filter1d(test_loss[:180], sigma=1)
		train_x = list(range(1, len(train_loss) + 1))
		test_x = list(range(1, len(test_loss) + 1))
		plt.figure()
		#plt.ylim(20, 110)
		plt.ylim(0.4, 1.0)
		plt.plot(train_x, train_loss, label = "Train Loss")
		plt.plot(test_x, test_loss, label = "Test Loss") 
		plt.xlabel('Epoch')
		plt.xlabel('Loss')
		plt.legend()
		plt.savefig(os.path.join("./", f[0:-4] + ".png"))
