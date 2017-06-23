import csv

INPUT_FILE = '../data/training.1600000.processed.noemoticon.csv'
OUTPUT_POS_FILE = '../data/train/train_pos_full.txt'
OUTPUT_NEG_FILE = '../data/train/train_neg_full.txt'
OUTPUT_NEU_FILE = '../data/train/train_neu_full.txt'
OUTPUT_TEST_FILE = '../data/test/test_data.txt'

with open(INPUT_FILE, 'r', encoding='utf8', errors='ignore') as input, open(OUTPUT_POS_FILE, 'w') as pos, open(OUTPUT_NEG_FILE, 'w') as neg, open(OUTPUT_NEU_FILE, 'w') as neu, open(OUTPUT_TEST_FILE, 'w') as test:
	reader = csv.reader(input)
	for line in reader:
		test.write(line[5] + '\n')
		if line[0] == '0': # negative
			neg.write(line[5] + '\n')
		elif line[0] == '2': # neutral
			neu.write(line[5] + '\n')
		elif line[0] == '4': # positive
			pos.write(line[5] + '\n')
		else:
			continue
	


