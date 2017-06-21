import sys
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import csv
import numpy as np
import os
from collections import defaultdict
import constants as CONSTS
import argparse

print('Loading data...')
feature_embeded = np.load("tfidf_embeded_risky_topic.npy")

LOG_DIR = 'logs_risky_topic_v2'
if not os.path.exists(LOG_DIR):
	os.makedirs(LOG_DIR)

# metafile = os.path.join(LOG_DIR, 'metadata.tsv')
filename = 'viz_risky_v2.csv'


def write_meta(LOG_DIR):
	print('Writing lablels...')
	for company in CONSTS.TAG_INDUSTRY_MAP.keys():
		metafile = os.path.join(LOG_DIR, 'metadata_'+company+'.tsv')
		with open(metafile, 'w', encoding='utf8') as s:
			s.write('\t'.join(['Tweet', 'Topic']) + '\n')
			with open(filename, 'r') as f:
				reader = csv.reader(f)	
				for i, line in enumerate(reader):
					if (line[1] == company):
						tweet = line[3].splitlines()
						tweet = ''.join(t for t in tweet if t)
						s.write('\t'.join([tweet, line[1] + '-' + line[4]]) + '\n')



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser(description='make embedding for tensorboard')
	parser.add_argument('-meta', '--make_meta', action='store_true', help='make meta files')

	args = parser.parse_args()
	if args.make_meta:
		write_meta(LOG_DIR)

	print('Creating a TensorFlow session...')
	tf.reset_default_graph()
	# X = tf.Variable([0.0], name='feature_embeded')
	place = tf.placeholder(tf.float32, shape=[None, feature_embeded.shape[1]])
	# set_x = tf.assign(X, place, validate_shape=False)

	model = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(model)

	inds = defaultdict(list)
	with open(filename, 'r') as f:
		reader = csv.reader(f)
		for i, line in enumerate(reader):
			inds[line[1]].append(i)	

	summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
	print('Creating a TensorFlow projector...')
	config = projector.ProjectorConfig()

	for company in inds.keys():
		temp = feature_embeded[inds[company],:]
		X = tf.Variable([0.0], name=company)
		set_x = tf.assign(X, place, validate_shape=False)
		sess.run(set_x, feed_dict={place: temp})
		embedding = config.embeddings.add()
		embedding.tensor_name = X.name
		embedding.metadata_path = os.path.join(LOG_DIR, 'metadata_' + company + '.tsv')

	projector.visualize_embeddings(summary_writer, config)

	print('Saving model...')
	saver = tf.train.Saver()
	saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
