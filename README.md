### Learning Invariant Rules from Data for Interpretable Anomaly Detection



## Getting Started

#### Install dependencies (with python 3.7) 

```shell
pip install -r requirements.txt
```

#### Reproduce results for invariant rule-based anomaly detection

```shell
cd experiments
python main_ir.py --dataset <dataset> --mode <mode> --reproduce
```

dataset can be one in [swat,batadal,kddcup99,gaspipeline,annthyroid,cardio].
mode can be one in [DTImpurity,UniformBins,KmeansBins]

#### Run the PUMP experiment

```shell
cd experiments
python PUMP_experiement.py
```

#### Run the WADI experiment

```shell
cd experiments
python WADI_experiement.py
```

#### Run the SWAT experiment

```shell
cd experiments
python SWAT_experiement.py
```
