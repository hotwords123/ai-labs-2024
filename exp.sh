#!/bin/bash

seeds=(0 1 2)

mkdir -p results

xargs -P 8 -I {} python policy_gradient.py --seed {} --plot results/{}.svg <<< $(printf "%s\n" "${seeds[@]}")
