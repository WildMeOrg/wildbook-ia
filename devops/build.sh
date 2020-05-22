#!/usr/bin/env bash

cd base && docker build -t wildme/ibeis-base:latest .
cd ..
cd dependencies && docker build -t wildme/ibeis-dependencies:latest .
cd ..
cd provision && docker build -t wildme/ibeis-provision:latest .
cd ..
docker build -t wildme/ibeis:latest .
