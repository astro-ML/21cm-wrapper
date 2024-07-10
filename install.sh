#!/bin/bash

echo "Start installation of the 21cm-wrapper."
echo "Building new environment 21cm-wrapper-env..."
# Create an new virtual environment
python -m venv ./21cm-wrapper-env
source ./21cm-wrapper-penv/bin/activate
pip install -r requirements.txt
pip install ./21cmFAST -e
