#!/bin/bash

# 1. Sort and group imports via isort
echo "Running isort..."
isort .

echo "Formatting and linting complete."