#!/usr/bin/env bash
export env_name="rl"
export job_name="QWENB_FALCON"

export exp_name="qwenb_falcon"
export student="config/model/local/unsloth/qwen3-8b.yaml"
: "${save_dir:?Must export save_dir=<output directory>}"

bash scripts/bash/new/run_student.sh
