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

Specify dataset to one of the following: swat, batadal, kddcup99, gaspipeline, annthyroid or cardio.

Specify mode to one of the following: DTImpurity, UniformBins or KmeansBins.

#### Run new experiments for invariant rule-based anomaly detection

```shell
cd experiments
python main_ir.py --dataset <dataset> --mode <mode> --theta <theta> --gamma <gamma>
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
