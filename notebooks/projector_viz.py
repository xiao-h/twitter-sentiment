from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf
import numpy as np
import os

print('Loading data...')
feature_embeded = np.load("tfidf_embeded.npy")

LOG_DIR = 'logs'

metafile = os.path.join(LOG_DIR, 'metadata.tsv')

if not os.path.isfile(metafile):
	print('Writing lablels...')
	TRAIN_POS = 'train_pos_full_orig.txt'
	TRAIN_NEG = 'train_neg_full_orig.txt'
	with open(TRAIN_POS, 'r', encoding='utf8', errors='ignore') as f, open(TRAIN_NEG, 'r', encoding='utf8', errors='ignore') as g, open(metafile, 'w', encoding='utf8') as s:
		s.write('\t'.join(['Tweet', 'Risk']) + '\n')
		for line in f:
			s.write('\t'.join([line.rstrip(), '1']) + '\n')
		for line in g:
			s.write('\t'.join([line.rstrip(), '0']) + '\n')

print('Creating a TensorFlow session...')
tf.reset_default_graph()
sess = tf.InteractiveSession()
X = tf.Variable([0.0], name='feature_embeded')
# embedding_var = tf.Variable(feature_embeded, name='feature_embeded')
place = tf.placeholder(tf.float32, shape=feature_embeded.shape)
set_x = tf.assign(X, place, validate_shape=False)
sess.run(tf.global_variables_initializer())
sess.run(set_x, feed_dict={place: feature_embeded})


print('Creating a TensorFlow projector...')
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = 'feature_embeded:0'
embedding.metadata_path = metafile
projector.visualize_embeddings(summary_writer, config)

print('Saving model...')
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, 'model.ckpt'))
