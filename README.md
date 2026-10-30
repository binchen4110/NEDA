# NEDA

This is the code of NEDA.

### Environment

```bash
conda create -n neda python=3.9
conda activate neda
pip install -r requirements.txt
```

### Preprocess data

Run the following commands from the project root:

```bash
python preprocess.py --data-dir data/ICEWS14s --seed 2024
```

### Train models

After training, the best validation checkpoint is automatically evaluated on the test set. Results are saved to `results.json` in the output directory.

#### ICEWS14s

```bash
python train.py --dataset ICEWS14s --seed 2024 \
    --dimension 200 --layers 2 --hops 2 --lr 0.001 --dropout 0.2 \
    --epochs 30 --patience 3 --device cuda:0 --output-dir runs/ICEWS14s
```

#### ICEWS18

```bash
python train.py --dataset ICEWS18 --seed 2024 \
    --dimension 200 --layers 2 --hops 2 --lr 0.001 --dropout 0.2 \
    --epochs 30 --patience 3 --device cuda:0 --output-dir runs/ICEWS18
```

#### ICEWS22

```bash
python train.py --dataset ICEWS22 --seed 2024 \
    --dimension 200 --layers 2 --hops 2 --lr 0.001 --dropout 0.2 \
    --epochs 30 --patience 3 --device cuda:0 --output-dir runs/ICEWS22
```

#### ICEWS23

```bash
python train.py --dataset ICEWS23 --seed 2024 \
    --dimension 200 --layers 2 --hops 2 --lr 0.001 --dropout 0.2 \
    --epochs 30 --patience 3 --device cuda:0 --output-dir runs/ICEWS23
```

#### ICEWS05-15

```bash
python train.py --dataset ICEWS05-15 --seed 2024 \
    --dimension 200 --layers 2 --hops 2 --lr 0.001 --dropout 0.2 \
    --epochs 30 --patience 3 --device cuda:0 --output-dir runs/ICEWS05-15
```

#### GDELT

```bash
python train.py --dataset GDELT --seed 2024 \
    --dimension 200 --layers 2 --hops 2 --lr 0.001 --dropout 0.2 \
    --epochs 30 --patience 3 --device cuda:0 --output-dir runs/GDELT
```
