from tensorflow.python import pywrap_tensorflow
import numpy as np

# reader = pywrap_tensorflow.NewCheckpointReader('/home/lucy/PycharmProjects/MTCNN-Tensorflow-master/data/MTCNN_model/PNet_landmark/PNet-30')
# reader = pywrap_tensorflow.NewCheckpointReader('/home/lucy/PycharmProjects/MTCNN-Tensorflow-master/data/MTCNN_model/RNet_No_Landmark/RNet-22')
reader = pywrap_tensorflow.NewCheckpointReader('/home/lucy/PycharmProjects/MTCNN-Tensorflow-master/data/MTCNN_model1/ONet_No_Landmark/ONet-20')

var_to_shape_map = reader.get_variable_to_shape_map()
param = []

for key in var_to_shape_map:
    print("tensor_name ", key)
    param.append(reader.get_tensor(key))

np.save('ONet1.npy', param)