#!/bin/bash

mkdir -p ../Holter_ECG/

wget -r -N -c -np -A "*" --no-parent --cut-dirs=5 -nH -P ../Holter_ECG/ https://physionet.org/files/music-sudden-cardiac-death/1.0.1/Holter_ECG/
