# Reproducible environment with JupyterLab. Python and everything else live in the image.
#
#   docker build -t fmflow .
#   docker run --rm -p 127.0.0.1:8888:8888 fmflow                         # JupyterLab on http://localhost:8888 (local only)
#   docker run --rm fmflow python tasks.py test                           # fast tests
#   docker run --rm -v "$PWD/data/private:/work/data/private" fmflow python tasks.py development   # with your data
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends git make && rm -rf /var/lib/apt/lists/*
WORKDIR /work
# dependencies first (cached layer), then the code
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[plot,notebooks,dev,gen,io]"
COPY . .
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/work/notebooks", "--ServerApp.token=", "--ServerApp.password="]
