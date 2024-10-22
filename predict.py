import tensorflow as tf
import numpy as np

import pickle
# Used for reliably getting the current hostname.
import socket
import time
import sys
import matplotlib.pyplot as plt


tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 1)")

# Example: './data/runs/euler/local-w2v-275d-1466050948/checkpoints/model-96690'
# tf.flags.DEFINE_string("checkpoint_file",
#     './data/runs/cnn-w2v-50d-1497986970-GloVe/checkpoints/model-500'
#     , "Checkpoint file from the training run.")
tf.flags.DEFINE_string("checkpoint_file",
    './data/runs/cnn-w2v-50d-1498201372-GloVe/checkpoints/model-7040'
    , "Checkpoint file from the training run.")
# tf.flags.DEFINE_string("checkpoint_file",
#     './data/runs/cnn-w2v-50d-1497990785-GloVe/checkpoints/model-220'
#     , "Checkpoint file from the training run.")
# tf.flags.DEFINE_string("checkpoint_file",
#     './data/runs/cnn-w2v-50d-1498057976-GloVe/checkpoints/model-220'
#     , "Checkpoint file from the training run.")

tf.flags.DEFINE_string(
    "validation_data_fname",
    "./data/preprocessing/full-trainX.npy",
    "The numpy dump of the validation data for Kaggle. Should ideally be"
    " preprocessed the same way as the training data.")
tf.flags.DEFINE_string(
    "y_data_fname",
    "./data/preprocessing/full-trainY.npy",
    "The numpy dump of the validation data for Kaggle. Should ideally be"
    " preprocessed the same way as the training data.")
tf.flags.DEFINE_string(
    "input_x_name",
    "input_x",
    "The graph node name of the input data. Hint: if you forget to name it,"
    " it's probably called 'Placeholder'.")
tf.flags.DEFINE_string(
    "predictions_name",
    "output/predictions",
    "The graph node name of the prediction computation. Hint: if you forget to"
    " name it, it's probably called 'Softmax' or 'output/Softmax'.")
tf.flags.DEFINE_float("dev_ratio", 0.01, "Percentage of data used for validation. Between 0 and 1. (default: 0.1)")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

if FLAGS.checkpoint_file is None:
    raise ValueError("Please specify a TensorFlow checkpoint file to use for"
                     " making the predictions (--checkpoint_file <file>).")

validation_data_fname = FLAGS.validation_data_fname
print("Validation data file: {0}".format(validation_data_fname))
validation_data_full = np.load(validation_data_fname)
y_data_fname = FLAGS.y_data_fname
y_data_full = np.load(y_data_fname)

# Randomly shuffle data
import datetime
random_seed = datetime.datetime.now().microsecond
print("Using random seed: {0}".format(random_seed))
np.random.seed(random_seed)
n_data = len(y_data_full)
shuffle_indices = np.random.permutation(np.arange(n_data))
# shuffle_indices = list(range(0, 8000))
# shuffle_indices.extend(list(range(800001,808001)))
# shuffle_indices = np.asarray(shuffle_indices)

x_shuffled = validation_data_full[shuffle_indices]
y_shuffled = y_data_full[shuffle_indices]

# validation_data = x_shuffled
# y_data = y_shuffled

# Train/dev split
n_dev = int(FLAGS.dev_ratio * n_data)
n_train = n_data - n_dev
x_train, x_dev = x_shuffled[:n_train], x_shuffled[n_train:]
y_train, y_dev = y_shuffled[:n_train], y_shuffled[n_train:]

validation_data = x_dev
y_data = y_dev

assert len(x_train) + len(x_dev) == n_data
assert len(y_train) + len(y_dev) == n_data

checkpoint_file = FLAGS.checkpoint_file
timestamp = int(time.time())
filename = "./data/output/prediction_lstm_{0}.csv".format(timestamp)
meta_filename = "{0}.meta".format(filename)
print("Predicting using checkpoint file [{0}].".format(checkpoint_file))
# print("Will write predictions to file [{0}].".format(filename))

print("Validation data shape: {0}".format(validation_data.shape))

graph = tf.Graph()
with graph.as_default():
    
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        print("Loading saved meta graph...")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        print("Restoring variables...")
        saver.restore(sess, checkpoint_file)
        print("Finished TF graph load.")

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name(FLAGS.input_x_name).outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name(FLAGS.predictions_name).outputs[0]

        # Collect the predictions here
        all_predictions = []
        print("Computing predictions...")
        for (id, row) in enumerate(validation_data):
            if (id + 1) % 1000 == 0:
                print("Done tweets: {0}/{1}".format(id + 1, len(validation_data)))

            prediction = sess.run(predictions, {
                input_x: [row],
                dropout_keep_prob: 1.0
            })[0]
            all_predictions.append((id + 1, prediction))

        print("Prediction done.")
        # print("Writing predictions to file...")
        # submission = open(filename, 'w+')
        # print('Id,Prediction', file=submission)

        scores = np.asarray([pred[0] for id, pred in all_predictions])
        n = len(all_predictions)

        isort = np.argsort(scores) # from the smallest
        #np.save('isort', isort)
        y_test = []
        for i in range(0, y_data.shape[0]):
            y_test.append(1) if np.array_equal(y_data[i], np.array([0 ,1])) else y_test.append(0)
        #np.save('y_test', y_test)
        y_test = np.array(y_test)[isort]

#        y_test = np.load('y_test.npy')
#        y_test = y_test[isort]

        tpr = np.empty((n,), dtype='float32')
        npos = sum(y_test)
        for i in range(1, n+1):
            tpr[i-1] = sum(y_test[:i])/npos
        rpp = np.arange(1/n, 1+1/n, 1/n)

        plt.figure()
        plt.plot(rpp, tpr)
        # plt.plot(fpr, tpr)
        # plt.plot(pred_tfidf[:,0], pred_tfidf[:,1])
        plt.grid(True)
        plt.show()

        sys.exit(0)


        # Ensure that IDs are from 1 to 10000, NOT from 0. Otherwise Kaggle
        # rejects the submission.
        for id, pred in all_predictions:
            print("%d,-1" % id)
            if pred[0] >= 0.5:
                print("%d,-1" % id)
            else:
                print("%d,1" % id)



        with open(meta_filename, 'w') as mf:
            print("Generated from checkpoint: {0}".format(checkpoint_file), file=mf)
            print("Hostname: {0}".format(socket.gethostname()), file=mf)

        print("...done.")
        print("Wrote predictions to: {0}".format(filename))
        print("Wrote some simple metadata about how the predictions were"
              " generated to: {0}".format(meta_filename))
