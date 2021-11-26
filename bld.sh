#!/usr/bin/env bash

ls dist/*
for f in dist/*.whl; do
  docker run -i --rm -v $PWD:/v -w /v --net=host quay.io/pypa/manylinux2010_x86_64 bash -x -e /v/tools/build/auditwheel repair --plat manylinux2010_x86_64 $f
done
sudo chown -R $(id -nu):$(id -ng) .
ls wheelhouse/*

