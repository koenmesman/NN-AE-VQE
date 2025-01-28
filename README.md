# NN-AE-VQE
In this work we aim to improve a well-established quantum computing method for calculating the inter-atomic potential, the variational quantum eigensolver (VQE), by presenting a quantum auto-encoded (QAE) VQE with neural-network predictions of the quantum ansatz parameters: NN-AE-VQE. To reduce the number of VQE circuit parameters, we apply a quantum auto-encoder to compress a quantum state representation of the atomic system, to which a naive circuit ansatz is applied. To avoid computationally expensive parameter optimization, we train a classical neural network to predict the circuit parameters.

This project is still in development, find the details in the current version of our paper: https://arxiv.org/abs/2411.15667
