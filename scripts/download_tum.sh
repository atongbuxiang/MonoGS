#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets/tum
cd datasets/tum

download_and_extract() {
    local split="$1"
    local name="$2"
    local archive="${name}.tgz"
    local url="https://cvg.cit.tum.de/rgbd/dataset/${split}/${archive}"

    if [[ ! -d "${name}" ]]; then
        if [[ ! -f "${archive}" ]]; then
            wget -c "${url}" -O "${archive}"
        fi
        tar -xzf "${archive}"
    else
        echo "${name} already exists, skipping."
    fi
}

download_and_extract "freiburg1" "rgbd_dataset_freiburg1_desk"
download_and_extract "freiburg2" "rgbd_dataset_freiburg2_xyz"
download_and_extract "freiburg3" "rgbd_dataset_freiburg3_long_office_household"
