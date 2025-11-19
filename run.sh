#!/bin/bash

for i in {1..25}; do
    echo "Run $i/25 starting..."
    uv run experiment.py
    echo "Run $i/25 finished."
    echo "----------------------"
done

echo "All 25 runs completed!"
