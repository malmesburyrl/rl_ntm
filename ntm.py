import numpy as np
import tensorflow as tf


class NTM():
    pass

def input(input_content, memory_content):

    return None


def output():


    input_control = 0 #[-1,0,1]
    memory_control = 0 #[-1,0,1]
    memory_content = np.zeros(10)  #vector len(M) where M is the size of memory vectors
    output_control = 0 #[0,1]
    output_content = np.zeros(10) #vector len(V) where V is num of characters in vocabulary

    return input_control, memory_control, memory_content, output_control, output_content
