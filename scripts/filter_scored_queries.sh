#!/bin/bash

# Basic variables
SCORED_FILE="data/scored_queries.jsonl"
FILTERED_FILE="data/filtered_queries.jsonl"
FILTER_TYPE="top"
PERCENTAGE=30
N=100

# Script paths
FILTER_SCRIPT="experiments/filter.py"

python "$FILTER_SCRIPT" \
    --scored_file "$SCORED_FILE" \
    --filtered_file "$FILTERED_FILE" \
    --filter_type "$FILTER_TYPE" \
    --percentage "$PERCENTAGE" \
    --N "$N"