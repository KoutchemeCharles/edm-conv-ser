# Teaching Language Models How to Code Like Learners

**Preference Optimization for Student Simulation**

[Charles Koutcheme](https://github.com/koutch) · [Juho Leinonen](https://github.com/juholeinonen) · [Arto Hellas](https://github.com/artohellas)  
*Educational Data Mining (EDM) 2026*

---

We train language models to simulate how students iteratively solve programming problems — producing realistic, imperfect submission trajectories that mirror authentic learning behavior. The pipeline covers three training paradigms: **SFT**, **DPO**, and **GRPO** with execution-based rewards.

**Dataset:** [`koutch/falcon_code`](https://huggingface.co/datasets/koutch/falcon_code) on Hugging Face Hub.

---

## Installation

```bash
conda env create -f environment.yaml
conda activate rl
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Set API keys before running:
```bash
export HF_ACCESS_TOKEN="<your_token>"   # HuggingFace Hub access
export WANDB_API_KEY="<your_key>"       # experiment tracking
export OPENAI_API_KEY="<your_key>"      # only for GPT-based experiments
```

---

## Pipeline

All stages share a single entry point driven by a JSON config file:

```bash
python scripts/run.py --config config/experiments/<name>/<id>.json [--test_run]
```

| Stage | `task.name` in config | Description |
|-------|----------------------|-------------|
| Preprocess | `preprocess` | Conversationalize student trajectories |
| SFT | `sft` | Supervised fine-tuning on next-submission prediction |
| DPO | `dpo` | Preference optimization from temporal submission order |
| GRPO | `grpo` | RL with execution-based grade reward |
| Evaluate | `eval` | Multi-step rollout evaluation |
| One-step eval | `one_step` | Single-step CodeBLEU evaluation |

See [`scripts/README.md`](scripts/README.md) for detailed run instructions and [`config/README.md`](config/README.md) for the configuration system.

---

## Repository Layout

```
├── src/          # Core library — models, training, data, utils
├── scripts/      # Entry points and bash launchers
├── config/       # Experiment configs (model × data × task)
├── outputs/      # Generated results and tables
└── logs/         # Training and evaluation logs
```

See [`src/README.md`](src/README.md) for the paper-to-code mapping and module descriptions.

---

## Citation

```bibtex
@inproceedings{koutcheme2026teaching,
  title     = {Teaching Language Models How to Code Like Learners: Preference Optimization for Student Simulation},
  author    = {Koutcheme, Charles and Leinonen, Juho and Hellas, Arto},
  booktitle = {Proceedings of the 19th International Conference on Educational Data Mining (EDM)},
  year      = {2026}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
