# Explainable GNN-based Approach to Fault Forecasting in Cloud Service Debugging

This repository contains supplementary materials for the paper:

<b>
  R. Lovas, E. Rigó, D. Unyi, B. Gyires-Tóth
  
  Explainable GNN-based Approach to Fault Forecasting in Cloud Service Debugging (under review)
</b>

If you need more data, code or information about our work, please contact the corresponding author: Róbert Lovas (robert.lovas (at) sztaki.hu)

![alt text](https://github.com/BME-SmartLab/GNN-Fault-Forecasting/blob/main/dataflow.png)

## How to Run the Experiments

First clone our repository:
```
git clone https://github.com/BME-SmartLab/GNN-Fault-Forecasting && cd GNN-Fault-Forecasting
```
Then build the docker image using our Dockerfile:
```
docker build -t gnn-fault-forecasting .
```
Next run the docker image interactively, mounting our repository into the container:
```
docker run -it -e USERNAME=myusername --gpus all --mount type=bind,source=$(pwd),target=/home/myusername/GNN-Fault-Forecasting gnn-fault-forecasting
```
And voilà, you can train the fault forecasting model or the fault explainability model:
```
python train_fault_forecasting_model.py
python train_fault_explainer_model.py
```

## Citing

If you use our code or results, please cite our paper: 

```
@article{
  to-be-published
}

```

## Acknowledgement

<p align="justify">
  The work reported in this paper, carried out as a collaboration between HUN-REN SZTAKI and BME, has been partly supported by the the European Union project RRF-2.3.1-21-2022-00004 within the framework of the Artificial Intelligence National Laboratory.
  This work was partially funded by the National Research, Development and Innovation Office (NKFIH) under OTKA Grant Agreement No. K 132838.
  The presented work of R. Lovas was also supported by the János Bolyai Research Scholarship of the Hungarian Academy of Sciences.
  On behalf of the ‘MILAB - SmartLab' cloud project, we are grateful for the possibility to use HUN-REN Cloud which helped us achieve the results published in this paper.
</p>
