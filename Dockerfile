FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    ipykernel \
    setuptools \
    wheel \
    jupyterlab \
    importlib_resources \
    tensorflow-datasets \
    ipynbname

RUN pip install --no-cache-dir "protobuf>=3.20,<5"


WORKDIR /workspace
EXPOSE 8888

CMD ["jupyter", "notebook", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''"]
