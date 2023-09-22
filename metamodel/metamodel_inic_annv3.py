import onnxruntime, random
import numpy as np

with open('const/limits_annv3.csv', 'r') as f:
        data = f.readlines()[1:]
        data = [i.strip().split(',') for i in data]

limits = {'min': {i[0]: float(i[2]) for i in data}}
limits.update({'max': {i[0]: float(i[3]) for i in data}})

col_order_original = [i[0] for i in data]
col_order = {name: i for i, name in enumerate(col_order_original)}

inputs_original = np.genfromtxt('inputs_annv3.csv', delimiter=',', names=True)
col_order_changed = [i.replace('app_ori_360', 'app_ori_-360') for i in inputs_original.dtype.names]
col_order_changed = [col_order[i] for i in col_order_changed]

inputs = np.array(inputs_original.tolist(), dtype='float32')[:, col_order_changed]

inputs = inputs.T

for i, colin in enumerate(inputs):

        minn = limits['min'][col_order_original[i]]
        maxx = limits['max'][col_order_original[i]]

        X_std = (colin - minn) / (maxx - minn)
        X_scaled = X_std * (1 - (-1)) + (-1)
        inputs[i] = X_scaled

inputs = inputs.T

temp_model_file = 'const/inic_annv3_cool_0.99693_25.34_5.58.onnx'
sess = onnxruntime.InferenceSession(temp_model_file)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

res = sess.run([label_name], {input_name: inputs})[0]
np.savetxt('results_annv3.csv', res, delimiter=',', fmt='%.2f')
