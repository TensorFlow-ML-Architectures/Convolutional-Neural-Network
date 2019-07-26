'''
This program exists solely to provide an example of how to load a generic tensorflow graph into your program. This accounts for the bare-minimum variables stored within the graph:
+ data placeholder
+ class placeholder
+ final graph layer placeholder
+ [Optional] dropout placeholder
'''
import tensorflow as tf

# Create a loader for the graph
def graph_loader(logdir=None, path_to_checkpoint=None, x_placeholder=None, y_placeholder=None, final_graph_layer_placeholder=None, keep_prob_placeholder=None):
  with tf.Session() as sess:
    #load the graph
    restore_saver = tf.train.import_meta_graph(path_to_checkpoint)
    #reload all the params to the graph
    restore_saver.restore(sess, tf.train.latest_checkpoint(logdir))
    global model
    model = tf.get_default_graph()
    
    #store the variables
    global x
    x = model.get_tensor_by_name(x_placeholder)
    global y_
    y_ = model.get_tensor_by_name(y_placeholder)
    global final_graph_layer
    final_graph_layer = model.get_tensor_by_name(final_graph_layer_placeholder)

    global keep_prob
    if keep_prob_placeholder != None:
      keep_prob = model.get_tensor_by_name(keep_prob_placeholder)


#EXAMPLE ON HOW TO LOAD THIS GRAPH w/ (minimal) user changes necessary
graph_loader(logdir='<MAIN_LOGGING_DIR>', path_to_checkpoint='<YOUR_PATH_TO_MODEL>.ckpt.meta', x_placeholder='x:0', y_placeholder='y_:0', final_graph_layer_placeholder='y_conv:0', keep_prob_placeholder='keep_prob:0')

