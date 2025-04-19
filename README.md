# AIgnite

AIgnite provides Agent capabilities to multiple applications such as PaperIgnition.

---

## Dev Installation

```
pip install -r requirements.txt
python install -e
```


## Docker Installation
We use docker to quicky deploy the package on cloud instances.
```
docker build -t aignite:latest .
docker run -it --rm aignite
```