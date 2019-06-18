# -*- coding: utf-8 -*-
""" Main file
    Author : Elvys LINHARES PONTES
    Version: 0.5"""
    
import siamese_neural_network_local_context
import argparse, data_util
import sys, os, pickle
import tensorflow as tf
import scipy.stats as meas
import numpy as np
from tqdm import tqdm

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser()
parser.add_argument("--bool_load_model", type=str2bool, default=False,
                    help="Bool to load a pretrained model.")
parser.add_argument("--bool_pretrain", type=str2bool, default=True,
                    help="Bool to pretrain the system.")
parser.add_argument("--bool_train", type=str2bool, default=True,
                    help="Bool to train the system.")
parser.add_argument("--bool_test", type=str2bool, default=True,
                    help="Bool to test the system.")
parser.add_argument("--batch_size", type=int, default=32,
                    help="Batch size")
parser.add_argument("--nb_epochs_pretrain", type=int, default=66,
                    help="Number of epochs of pretrain")
parser.add_argument("--nb_epochs", type=int, default=301,
                    help="Number of epochs")
parser.add_argument("--max_length", type=int, default=50,
                    help="Max sentence length")
parser.add_argument("--hidden_size", type=int, default=50,
                    help="Hidden layer size.")
parser.add_argument("--forget_bias", type=str, default=2.5,
                    help="Forget bias.")
parser.add_argument("--learning_rate", type=float, default=0.1,
                    help="Learning rate.")
parser.add_argument("--number_layers", type=int, default=1,
                    help="Number of layers.")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout.")
parser.add_argument("--word_emb_size", type=int, default=300,
                    help="word embedding size.")
parser.add_argument("--local_context_size", type=int, default=5,
                    help="Local context size.")
args = parser.parse_args()

def load_dataset(syn_aug= True):
    """
        Load dataset.
    """
    pretrain  = pickle.load(open("data/stsallrmf.p","rb"), encoding='latin1')#[:-8]
    train     = pickle.load(open("data/semtrain.p",'rb'))
    test      = pickle.load(open("data/semtest.p",'rb'))
    val       = train[:int( len(train)*0.15 )]
    train     = train[int( len(train)*0.15 ):]

    # Expand dataset
    if syn_aug:
        train = data_util.expand(train)
        print ("Dataset Expanded")

    return np.asarray(pretrain), np.asarray(train), np.asarray(val), np.asarray(test)

def calculate_correlation(prediction, reference):
    """
        Calculate the error and correlations of predictions.
    """
    predictions, references = [], []
    prediction = list((np.asarray(prediction) * 4.0) + 1.0)
    predictions.extend(prediction)
    references.extend(reference)

    predictions = np.array(predictions)
    references  = np.array(references)
    print("Error\tCorrelation_Pearson\t\tCorrelation_Spearman:")
    print(str( np.mean(np.square(predictions-references)) ) + "\t" + str( meas.pearsonr(predictions,references)[0] ) + "\t\t" + str( meas.spearmanr(references,predictions)[0] ) )

    return meas.pearsonr(predictions,references)[0]

def test_network(sess, network, test):
    # Initialize iterator with test data
    feed_dict={ network.x1:          test[0],
                network.len1:        test[1],
                network.x2:          test[2],
                network.len2:        test[3],
                network.y:           test[4],
                network.batch_size:  np.array(test[0]).shape[0]}
    sess.run(network.iter.initializer, feed_dict=feed_dict)
    for _ in tqdm(range(1)):
        loss, prediction, reference = sess.run([network.loss_test, network.prediction_test, network.reference])

    return loss, prediction, reference

def training_network(sess, network, train, val, path_save, nb_epochs):
    old_correlation, count = 0.0, 0
    n_batches       = np.array(train[0]).shape[0] // args.batch_size

    # Initialise iterator with train data
    feed_dict={ network.x1:         train[0],
                network.len1:       train[1],
                network.x2:         train[2],
                network.len2:       train[3],
                network.y:          train[4],
                network.batch_size: args.batch_size}
    sess.run(network.iter.initializer, feed_dict=feed_dict)

    for eidx in range(nb_epochs):
        tot_loss = 0
        for _ in tqdm(range(n_batches)):
            _, loss_value = sess.run([network.train_op, network.loss])
            tot_loss += loss_value
        print("Iter: {}, Loss: {:.4f}".format(eidx, tot_loss / n_batches))

        # Early stopping
        if not eidx % 10:
            print("Early stopping")
            loss, prediction, reference = test_network(sess, network, val)
            correlation = calculate_correlation(prediction, reference)

            if correlation > old_correlation:
                old_correlation = correlation
                count = 0
                print("Saving model...")
                network.saver.save(sess, path_save)
                print("Model saved in file: %s" % path_save)
            else:
                count += 1
                if count >= 10:
                    break

            # Reload training dataset
            feed_dict={ network.x1:         train[0],
                        network.len1:       train[1],
                        network.x2:         train[2],
                        network.len2:       train[3],
                        network.y:          train[4],
                        network.batch_size: args.batch_size}
            sess.run(network.iter.initializer, feed_dict=feed_dict)

def main():
    # Path to model
    path_save    = 'models/se_' + str(args.hidden_size) + 'fb_' + str(args.forget_bias) + 'nl_' + str(args.number_layers)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    path_save    += '/model'
    # Load datasets
    pretrain, train, val, test = load_dataset()
    # Get [ emb1, lengths1, emb2, lengths2, y ]
    pretrain = data_util.prepare_data(pretrain, maxlen=args.max_length, training=True)
    train    = data_util.prepare_data(train, maxlen=args.max_length, training=True)
    val      = data_util.prepare_data(val, maxlen=args.max_length, training=False)
    test     = data_util.prepare_data(test, maxlen=args.max_length, training=False)
    # Create network
    network = siamese_neural_network_local_context.SiameseLSTMCNN(
            sequence_embedding  = args.hidden_size,
            forget_bias         = args.forget_bias,
            learning_rate       = args.learning_rate,
            number_layers       = args.number_layers,
            max_length          = args.max_length,
            word_emb_size       = args.word_emb_size,
            local_context_size  = args.local_context_size,
            dropout             = args.dropout)
    # Initialize tensorflow
    with tf.Session() as sess:
        # Initialize variables
        sess.run(network.initialize_variables)
        if args.bool_load_model:
            network.saver.restore(sess, path_save)
            print("Model restored!")
        if args.bool_pretrain:
            # Pretraining network
            print("Pretraining network ...")
            training_network(sess, network, pretrain, val, path_save, args.nb_epochs_pretrain)
        if args.bool_train:
            # Training network
            print("Training network ...")
            training_network(sess, network, train, val, path_save, args.nb_epochs)
        if args.bool_test:
            # Testing network
            print("Test network ...")
            loss, prediction, reference = test_network(sess, network, test)
            calculate_correlation(prediction, reference)

if __name__ == '__main__':
    main()
