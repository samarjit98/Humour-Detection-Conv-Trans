import os
import pickle

files = os.listdir(".")
for f in files:
	if ".txt" in f: 
		print(f)
		content = open(f, 'r').read()
		content = [ x.split()[0][0:-1] for x in  content.split("Acc: ")][1:]
		content = [float(x) for x in content]

		train_acc = content[::2]
		test_acc = content[1::2]
		print(content[0:2])
		print(train_acc[0])
		print(test_acc[0])

		file_handler = open(f[0:-4] + ".pkl", 'wb+')
		pickle.dump((train_acc, test_acc), file_handler)
		file_handler.close()

