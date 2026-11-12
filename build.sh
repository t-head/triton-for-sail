#!/bin/bash

set +e
export TRITON_DEPENDENCY_BASE_URL="${TRITON_DEPENDENCY_BASE_URL-https://download.t-head.cn/artifactory/generic-tools/triton/deps}"
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

cd "$DIR"

python -m pip install -r python/requirements.txt || exit 1

if [[ -f /etc/os-release ]] && (. /etc/os-release; [[ "$ID" == alios ]]) \
    && [[ -z "${LLVM_SYSPATH-}" && -z "${TRITON_LLVM_SYSTEM_SUFFIX-}" ]]; then
    TRITON_CACHE="${TRITON_HOME:-$HOME}/.triton"
    LLVM_DOWNLOAD_ARGS=()
    case "${TRITON_OFFLINE_BUILD^^}" in
        ON|1|YES|TRUE|Y) LLVM_DOWNLOAD_ARGS+=(--triton-offline-build);;
    esac
    python python/build_helpers.py write_thirdparty_cmake_vars \
        --triton-cache-path "$TRITON_CACHE" "${LLVM_DOWNLOAD_ARGS[@]}" \
        --packages llvm --output /dev/null || exit 1
    LLVM_SDK="$TRITON_CACHE/llvm/llvm-alios7u2-x64"
    export PATH="$LLVM_SDK/bin:$PATH"
    export TRITON_BUILD_WITH_CLANG_LLD=1
    export CFLAGS="${CFLAGS-} --gcc-toolchain=\"$LLVM_SDK/toolchain\""
    export CXXFLAGS="${CXXFLAGS-} --gcc-toolchain=\"$LLVM_SDK/toolchain\""
    export TRITON_APPEND_CMAKE_ARGS="\
-DCMAKE_SHARED_LINKER_FLAGS='-fuse-ld=lld -static-libstdc++' \
-DCMAKE_MODULE_LINKER_FLAGS='-fuse-ld=lld -static-libstdc++' \
-DCMAKE_EXE_LINKER_FLAGS='-fuse-ld=lld -static-libstdc++' ${TRITON_APPEND_CMAKE_ARGS-}"
fi

BUILD_CMD="python setup.py bdist_wheel"

echo $BUILD_CMD
eval $BUILD_CMD || exit 1

cd dist
pip uninstall -y triton
pip install `ls | grep -E "triton.*whl"`
