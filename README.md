# DBRHD handwritten digit recognition

## Dataset

DBRHD dataset

## Method

- CNN
- KNN
- DNN(using numpy)
- DNN(using sklearn)

## Usage

### Run algorithm

For example: run knn algorithm

```shell
python main.py run --method knn --params 1 3 5 7
```

### Run time benchmark

Test running time and save as csv

```shell
python main.py benchmark
```

### Plot the result

```shell
python main.py plot {filepath}
```

**> Make sure you have result savd in {filepath}**
