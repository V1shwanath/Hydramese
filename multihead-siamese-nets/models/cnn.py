import tensorflow as tf

from layers.convolution import cnn_layers
from layers.recurrent import rnn_layer
from layers.attention import stacked_multihead_attention

from layers.similarity import manhattan_similarity
from models.base_model import BaseSiameseNet
from utils.config_helpers import parse_list


class CnnSiameseNet(BaseSiameseNet):
    
    def __init__(
            self,
            max_sequence_len,
            vocabulary_size,
            main_cfg,
            model_cfg,
    ):
        BaseSiameseNet.__init__(
            self,
            max_sequence_len,
            vocabulary_size,
            main_cfg,
            model_cfg,
        )
    
    def siamese_layer(self, sequence_len, model_cfg):
        # CNN Specific variables
        num_filters = parse_list(model_cfg['PARAMS']['num_filters'])
        filter_sizes = parse_list(model_cfg['PARAMS']['filter_sizes'])
        dropout_rate = float(model_cfg['PARAMS']['dropout_rate'])
        
        # RNN Specific variables
        hidden_size = 128
        cell_type = "GRU"
        
        
        # Multi-head Attention Specific variables
        num_blocks = model_cfg['PARAMS'].getint('num_blocks')
        num_heads = model_cfg['PARAMS'].getint('num_heads')
        use_residual = model_cfg['PARAMS'].getboolean('use_residual')
        att_dropout_rate = 0.1
        
        # CNN Layers
        print("CNN layers are starting now-------------------------------------------------------------------")
        out1 = cnn_layers(
            self.embedded_x1,
            sequence_len,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
        )
        
        out2 = cnn_layers(
            self.embedded_x2,
            sequence_len,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            is_training=self.is_training,
            dropout_rate=dropout_rate,
            reuse=True,
        )
        
        print("RNN layers are starting now-------------------------------------------------------------------")
        # RNN Layers
        outputs_sen1 = rnn_layer(
            embedded_x=self.embedded_x1,
            hidden_size=hidden_size,
            bidirectional=False,
            cell_type=cell_type,
        )
        outputs_sen2 = rnn_layer(
            embedded_x=self.embedded_x2,
            hidden_size=hidden_size,
            bidirectional=False,
            cell_type=cell_type,
            reuse=True,
        )   
        out3 = tf.reduce_mean(outputs_sen1, axis=1)
        out4 = tf.reduce_mean(outputs_sen2, axis=1)
        
        # Multi-head Attention Layers
        print("MultiHead layers are starting now-------------------------------------------------------------------")
        attention_out1, self.debug_vars['attentions_x1'] = stacked_multihead_attention(
            self.embedded_x1,
            num_blocks=2,
            num_heads=8,
            use_residual=False,
            is_training=self.is_training,
            dropout_rate=att_dropout_rate,
        )
        
        attention_out2, self.debug_vars['attentions_x2'] = stacked_multihead_attention(
            self.embedded_x2,
            num_blocks=2,
            num_heads=8,
            use_residual=False,
            is_training=self.is_training,
            dropout_rate=att_dropout_rate,
            reuse=True,
        )
        
        out5 = tf.reduce_sum(attention_out1, axis=1)
        out6 = tf.reduce_sum(attention_out2, axis=1)
        
        cnn_mansim = manhattan_similarity(out1, out2)
        rnn_mansim = manhattan_similarity(out3, out4)
        multihead_mansim = manhattan_similarity(out5, out6)
        
        
        # average_similarity = tf.reduce_mean(tf.stack([cnn_mansim, rnn_mansim,multihead_mansim]), axis=0)
        print("these are the final ouputs for-----------------------------------------------------------------------",cnn_mansim, rnn_mansim, multihead_mansim)
        
        # Combine outputs
        combined_outputs = tf.concat([out1, out2, out3, out4], axis=1)

        # Fully-connected layer 1 with ReLU activation
        fc1 = tf.layers.dense(combined_outputs, units=6, activation=tf.nn.relu)
        
        #add a dropout layer
        fc2 = tf.layers.dropout(fc1, rate=0.3, training=self.is_training)

        # Fully-connected layer 2 with sigmoid activation for similarity score
        similarity_score = tf.layers.dense(fc2, units=1, activation=tf.nn.sigmoid)

        return similarity_score