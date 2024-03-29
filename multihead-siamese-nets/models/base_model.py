import tensorflow as tf

from layers.basics import optimize
from layers import losses

class BaseSiameseNet:
    
    def __init__(
            self,
            max_sequence_len,
            vocabulary_size,
            main_cfg,
            model_cfg,
    ):
        self.x1 = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_len])
        self.x2 = tf.placeholder(dtype=tf.int32, shape=[None, max_sequence_len])
        self.is_training = tf.placeholder(dtype=tf.bool)
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
        self.sentences_lengths = tf.placeholder(dtype=tf.int32, shape=[None])
        
        self.debug = None
        self.debug_vars = dict()
        
        self.loss_function = losses.get_loss_function(main_cfg['PARAMS'].get('loss_function'))
        self.embedding_size = main_cfg['PARAMS'].getint('embedding_size')
        self.learning_rate = main_cfg['TRAINING'].getfloat('learning_rate')
        
        with tf.variable_scope('embeddings'):
            word_embeddings = tf.get_variable('word_embeddings',
                                              [vocabulary_size, self.embedding_size])
            self.embedded_x1 = tf.gather(word_embeddings, self.x1)
            self.embedded_x2 = tf.gather(word_embeddings, self.x2)
        
        with tf.variable_scope('siamese'):
            self.predictions = self.siamese_layer(max_sequence_len, model_cfg)
        
        with tf.variable_scope('loss'):
            self.loss = self.loss_function(self.labels, self.predictions)
            self.opt = optimize(self.loss, self.learning_rate)
        
        with tf.variable_scope('metrics'):
            self.temp_sim = tf.rint(self.predictions)
            self.correct_predictions = tf.equal(self.temp_sim, tf.to_float(self.labels))
            self.accuracy = tf.reduce_mean(tf.to_float(self.correct_predictions))

            self.f1_score = tf.contrib.metrics.f1_score(tf.to_float(self.labels), tf.to_float( self.temp_sim))
            
            #print value inside in labels and temp_sim
            
        
        with tf.variable_scope('summary'):
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("f1_score", self.f1_score)
            self.summary_op = tf.summary.merge_all()
    
    def siamese_layer(self, sequence_len, model_cfg):
        """Implementation of specific siamese layer"""
        raise NotImplementedError()
