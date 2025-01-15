#!/bin/bash

eval "$(micromamba shell hook --shell=bash)"
micromamba activate syntheseus-full

#python /app/server.py