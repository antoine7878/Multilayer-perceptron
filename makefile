split:
	python ./mlp.py split data.csv

train:
	python ./mlp.py train modelfile

eval:
	python ./mlp.py eval modelfile
