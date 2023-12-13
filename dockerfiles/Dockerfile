FROM tensorflow/tensorflow:2.14.0rc1-gpu-jupyter
#RUN pip install gensim tensorflow-text==2.14.0-rc0 pydot tensorflow-datasets nltk scipy scikit-learn pandas sentencepiece transformers graphviz seaborn torch torchvision omegaconf
COPY requirements_lavis.txt /tmp/
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements_lavis.txt
RUN mkdir -p /export/home/.cache/lavis/
RUN python3 -m pip install --no-cache-dir opencv-python-headless albumentations
RUN python3 -m pip install -U flash-attn --no-build-isolation
#COPY ../lavis/datasets/coco/images /export/home/.cache/lavis/coco/images/
CMD ["bash" "-c" "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
