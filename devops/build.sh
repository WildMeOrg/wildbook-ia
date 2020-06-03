#!/usr/bin/env bash

cd base && docker build -t wildme/wbia-base:latest .
cd ..
cd dependencies && docker build -t wildme/wbia-dependencies:latest .
cd ..
cd provision && docker build -t wildme/wbia-provision:latest .
cd ..
docker build -t wildme/wbia:latest .
