#!/usr/bin/env bash
export env_name="rl"
export job_name="GPT-5"

export exp_name="gpt5"
export model="config/model/remote/gpt-5-minimal-mini.yaml"
: "${save_dir:?Must export save_dir=<output directory>}"

bash scripts/bash/new/run_llm.sh

