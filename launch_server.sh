#!/bin/bash
pip install -r requirements.txt 
uvicorn api:app --host=0.0.0.0 --port=80