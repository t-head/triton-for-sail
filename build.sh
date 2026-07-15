#!/bin/bash

set +e
DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
#build options
BUILD_DEVICE_PPU=1
BUILD_PLATFORM="ubuntu"

while [[ -n $1 ]];
do
    case "$1" in
        ppu)    BUILD_DEVICE_PPU=1;;
        alios) BUILD_PLATFORM="alios";;
        ubuntu) BUILD_PLATFORM="ubuntu";;
    esac
    shift
done

export BUILD_PLATFORM=${BUILD_PLATFORM}

cd $DIR
BUILD_CMD="python setup.py bdist_wheel"

echo $BUILD_CMD
eval $BUILD_CMD || exit 1

cd dist
pip uninstall -y triton
pip install `ls | grep -E "triton.*whl"`
