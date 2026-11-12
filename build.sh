#!/bin/bash
set -e

export GIT_CONFIG_GLOBAL=/dev/null
export TRITON_DEPENDENCY_BASE_URL="${TRITON_DEPENDENCY_BASE_URL-https://download.t-head.cn/artifactory/generic-tools/triton/deps}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

python -m pip install -r python/requirements.txt

if [[ -f /etc/os-release ]] && (. /etc/os-release; [[ "$ID" == alios ]]); then
    TRITON_CACHE="${TRITON_HOME:-$HOME}/.triton"
    python python/build_helpers.py write_thirdparty_cmake_vars \
        --triton-cache-path "$TRITON_CACHE" --packages llvm --output /dev/null
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

python -m pip uninstall -y triton
python -m pip install -e . --no-build-isolation -v
