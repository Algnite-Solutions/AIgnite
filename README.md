# AIgnite

AIgnite provides Agent capabilities to multiple applications such as PaperIgnition.

---

## Dev Installation

```
pip install -r requirements.txt
```
There are two ways of install this lib in your local env
1. pip install -e .
2. export PYTHOTNPATH=$AIGNITE_PATH/src/

## Docker Installation
We use docker to quicky deploy the package on cloud instances.
```
docker build -t aignite:latest .
docker run -it --rm aignite
```