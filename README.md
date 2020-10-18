# SY-GNN

 PyTorch implementation of [Symmetrical Graph Neural Network for Quantum Chemistry with Dual Real and Momenta Space](https://pubs.acs.org/doi/abs/10.1021/acs.jpca.0c03201)

## Requirements:
ase>=3.10.1

numpy

python>=3.6

tensorboard>=1.15.0

tensorboardx>=2.0

torch>=1.5.0

## Demo

### Database generation

We provides `QM_debug.db`, which is a subset of QM-sym dataset in this repo. If you do not want to use the `QM_debug.db` to run the model. You need to build your database from XYZ files. 

Modify `dbpath` and `xyzpath` in the last line in the file `netpack/datasets/qm_sym.py`. The `dbpath` is the save path of the generated database file, and `xyzpath` is the folder storage XYZ files.

### Training

You can use following code to run with demo small database from QM-sym: (on Linux system)

```shell script
CUDA_VISIBLE_DEVICES=0 python -W ignore main.py train sygnn --cuda --data_size 16 ./database/QM_debug.db ./logs/QM/debug
```

It should finish in few minutes if you have an CUDA device, the program should be expected to terminated without any error. The output log file in the `log` folder.

If you want to run with your own dataset, following the README of QM-sym dataset to build your XYZ files, and specify the folder to your `.xyz` file in the `qm_sym.py` to generate new database file.

## Reference
Some codes are from [SchNetPack](https://github.com/atomistic-machine-learning/schnetpack).

The data is from [QM-sym database](https://github.com/XI-Lab/QM-sym-database).
