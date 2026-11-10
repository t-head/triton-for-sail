#!/bin/bash
set -e

export GIT_CONFIG_GLOBAL=/dev/null
export TRITON_DEPENDENCY_BASE_URL="${TRITON_DEPENDENCY_BASE_URL-https://download.t-head.cn/artifactory/generic-tools/triton/deps}"

if [[ "$(python -c 'import nanobind; print(nanobind.__version__)' 2>/dev/null)" != "2.10.2" ]]; then
    python -m pip install nanobind==2.10.2
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

python -m pip uninstall -y triton
python -m pip install -e . --no-build-isolation -v
