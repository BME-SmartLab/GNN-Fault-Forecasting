# Revolutionizing Cloud Service Debugging: A GNN-Driven Approach to SLA Forecasting

This repository contains supplementary materials for the paper:

<b>
  R. Lovas, E. Rigó, D. Unyi, B. Gyires-Tóth
  
  Revolutionizing Cloud Service Debugging: A GNN-Driven Approach to SLA Forecasting (under review)
</b>

If you need more data, code or information about our work, please contact the corresponding author: Róbert Lovas (robert.lovas (at) sztaki.hu)

![alt text](https://github.com/BME-SmartLab/GNN-SLA-Forecast/blob/main/dataflow.png)

## How to Run the Experiments

First clone our repository:
```
git clone https://github.com/BME-SmartLab/GNN-SLA-Forecast && cd GNN-SLA-Forecast
```
Then build the docker image using our Dockerfile:
```
docker build -t sla-forecast .
```
Next run the docker image interactively, mounting our repository into the container:
```
docker run -it -e USERNAME=myusername --mount type=bind,source=$(pwd),target=/home/myusername/GNN-SLA-Forecast sla-forecast
```
And voilà, you can execute the experiment you are interested in:
```
python exp1.py
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
  The presented work of D. Unyi was also supported by the ÚNKP-23-3-II-BME-399 New National Excellence Program of the Ministry for Culture and Innovation from the source of the National Research, Development and Innovation Fund.
  On behalf of the 'MILAB - SmartLab' cloud project, we are grateful for the possibility to use ELKH Cloud \cite{mihalyhederPresentFutureELKH2022} which helped us achieve the results published in this paper.
</p>
