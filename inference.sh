#! /bin/bash
# inference for GLUE tasks, including RTE, MRPC, MNLI, QNLI, QQP, STS-B, SST-2

# RTE
echo ---infering RTE---
python ./inferences/inference_rte.py
echo ------------------

# MRPC
echo ---infering MRPC---
python ./inferences/inference_mrpc.py
echo ------------------

# MNLI
echo ---infering MNLI---
python ./inferences/inference_mnli.py
echo ------------------

# QNLI
echo ---infering QNLI---
python ./inferences/inference_qnli.py
echo ------------------

# QQP
echo ---infering QQP---
python ./inferences/inference_qqp.py
echo -----------------

# STS-B
echo ---infering STS-B---
python ./inferences/inference_stsb.py
echo ------------------

# SST-2
echo ---infering SST-2---
python ./inferences/inference_sst2.py
echo ------------------