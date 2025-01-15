FROM mambaorg/micromamba:1.4.7-bionic-cuda-11.3.1

USER root

RUN apt update && apt -y install wget git
ENV SYNTHESEUS_CACHE_DIR="/app/models/"

SHELL ["bash", "-c"]


COPY environment_full.yml /tmp/environment_full.yml


RUN micromamba env create -f /tmp/environment_full.yml -y && \
    micromamba clean --all --yes


RUN micromamba run -n syntheseus-full pip install 'syntheseus[root-aligned]'

EXPOSE 9502


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh


ENTRYPOINT ["/bin/bash", "-c", "/entrypoint.sh"]

