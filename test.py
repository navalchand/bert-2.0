import tensorflow as tf
import modeling as modeling

# allow gpu growth
tf.config.gpu.set_per_process_memory_growth(True)

# bert config
init_checkpoint = "/Users/naval/Desktop/Naval/CBM/BERT-NER-master/bert/multi_cased_L-12_H-768_A-12/bert_model.ckpt"
bert_config_file = "/Users/naval/Desktop/Naval/CBM/BERT-NER-master/bert/multi_cased_L-12_H-768_A-12/bert_config.json"
bert_config = modeling.BertConfig.from_json_file(bert_config_file)

# fake input for initializing weights
input_ids = tf.Variable(tf.ones(shape=(1, 512), dtype=tf.int32))

bert = modeling.BertModel(bert_config, is_training=True)
bert(inputs = input_ids)

print("Loading weights")
bert.load_weights(init_checkpoint)
print("Weights loadad!!!")