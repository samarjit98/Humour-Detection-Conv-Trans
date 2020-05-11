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

for f in files:
	if ".txt" in f: 
		print(f)
		content = open(f, 'r').read()
		content = [ x.split()[0][0:-1] for x in  content.split("Train Loss: ")][1:]
		content = [float(x) for x in content]

		train_loss = content
		print(train_loss)
		print(len(train_loss))

		content1 = open(f, 'r').read()
		content1 = [ x.split()[0][0:-1] for x in  content1.split("Test Loss: ")][1:]
		content1 = [float(x) for x in content1]

		test_loss = content1
		print(test_loss)
		print(len(test_loss))

		file_handler = open(f[0:-4] + "_loss.pkl", 'wb+')
		pickle.dump((train_loss, test_loss), file_handler)
		file_handler.close()

