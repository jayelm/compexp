#!/usr/bin/env bash
set -e

# Start from parent directory of script
cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "Download densenet161 trained on Places365"
echo "Downloading $MODEL"
mkdir -p zoo
pushd zoo
wget --progress=bar \
   http://places2.csail.mit.edu/models_places365/whole_densenet161_places365_python36.pth.tar
popd

echo "done"
