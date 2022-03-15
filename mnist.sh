#!/bin/sh

for i in {1..10}
do
	echo "${i}th trials"
	python3 manifold_learning.py mnist
done
