# qmin
Package for computation of mutual information of individual neurons in pytorch NNs.

## Usage
```
qmins = qmin.compute_neighbours_qmin(my_pytorch_module, my_dataset, quantization_degree=4)
df = qmin.create_qmin_weights_dataframe(qmins, my_pytorch_module)

df.plot.scatter(x="QMIN", y="AbsWgs")
print(df.corr())
```
