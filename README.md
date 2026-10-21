# LLMs From Scratch to Advanced — Notebook Guide

A visual, hands-on Jupyter course that connects the mathematical foundations of language modeling to a small GPT-style model and a selection of modern LLM engineering patterns. It is intended for learning and experimentation, not as a production LLM framework or a benchmark suite.

**Project files:** [Open the notebook](LLM Engineering Guide.ipynb) · [Read the validation report](<Validation Report.md>)

> **Filename note:** The supplied project brief refers to `LLM Engineering Guide.ipynb`, but that exact file was not present in the inspected workspace. The notebook actually available alongside the dependency files and checkpoint is `LLM Engineering Guide.ipynb`. This README references the available file; no notebook was renamed.

## What is covered

- **Foundations:** AI/ML/DL/NLP/LLM concepts; vectors, matrices, probability, entropy, cross-entropy, gradients and optimization.
- **Text and Transformer mechanics:** character and regex tokenization, toy byte-pair encoding, embeddings, positional information, RoPE illustration, Q/K/V attention, causal masking, multi-head tensors and decoder blocks.
- **A GPT-style model from scratch:** character-level data, next-token targets, a small PyTorch decoder, training/validation-loss inspection, checkpoint saving, attention visualization and autoregressive generation.
- **Practical LLM methods:** pretrained-model concepts, prompting, structured-output validation, instruction tuning, response-only loss masking, a toy LoRA layer, QLoRA concepts and preference optimization/alignment.
- **Retrieval and systems:** local TF–IDF retrieval, retrieval metrics on synthetic labels, a restricted calculator/document-search tool demo, inference concepts, evaluation, reliability and safety.
- **Architecture alternatives and multimodality:** conceptual Transformer/Mamba comparisons and a synthetic image-patch embedding demonstration.
- **Further study:** capstone project workflows, a glossary, formula reference, troubleshooting notes and reading links.

## Notebook roadmap

The notebook has seven parts and 26 numbered chapters:

| Part | Chapters | Focus |
|---|---:|---|
| I — Foundations of AI and Language Modeling | 1–4 | Core concepts, mathematics, neural networks and text representation |
| II — From Language Modeling to Transformers | 5–9 | Language modeling, positions, attention and Transformer architecture |
| III — Build a Small GPT-Style Model From Scratch | 10–13 | Data, model implementation, training and generation |
| IV — Modern LLMs and Practical Usage | 14–18 | Pretrained models, prompting, fine-tuning, LoRA/QLoRA and alignment |
| V — Retrieval, Agents and Production Systems | 19–22 | RAG, tools, serving, evaluation and safety |
| VI — Architecture Alternatives and Multimodal Models | 23–24 | Transformer/Mamba trade-offs and multimodal representations |
| VII — Capstone Projects and Further Study | 25–26 | Project workflows and reference material |

## Implementations versus demonstrations

It is important to distinguish runnable code from conceptual material:

- **Runnable core implementation:** the notebook builds and trains a small character-level GPT-like decoder directly in PyTorch. It has 104,640 parameters in the audited configuration and trains for 180 steps in the default example. The training corpus is only 660 characters, so generated text is intentionally low quality and should not be interpreted as a useful assistant or a quality benchmark.
- **Runnable local demonstrations:** numerical and attention visualizations, a toy BPE example, a small LoRA wrapper, response-only loss masking, TF–IDF retrieval, synthetic-label retrieval metrics, a restricted local calculator/document-search demo, structured-output checks and synthetic image patch projection.
- **Conceptual or simplified material:** QLoRA, RLHF/DPO alignment, deployment optimization, and Transformer-versus-Mamba trade-offs are primarily explanations and workflow diagrams. The Mamba comparison is not a full implementation of the published Mamba architecture.
- **Optional external integrations:** pretrained text generation and real image captioning require additional setup. The pretrained-model code is disabled by default, and the image-captioning snippet is commented out. No full LLM fine-tuning, PEFT adapter training, QLoRA training, RLHF/DPO optimization, distributed serving or broad benchmark is run by the core notebook.
- **Synthetic vision example:** a generated geometric image is split into patches and passed through a randomly initialized linear layer. This demonstrates tensor processing, not pretrained vision understanding.

## Requirements

The supplied `requirements.txt` contains the core packages used for the local notebook path, including NumPy, Matplotlib, scikit-learn, Pillow and PyTorch. It also includes JupyterLab, IPython-kernel support and notebook-format utilities. A GPU is not required for the small core example; runtime depends on the machine and package builds.

### Windows PowerShell

From the directory containing the project files:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency file does not pin every package to an exact version. Choose a Python/PyTorch build supported by your operating system and hardware. For platform-specific PyTorch installation options, use the official selector at <https://pytorch.org/get-started/locally/>; the appropriate command depends on your CPU/GPU and driver environment.

## Launch the notebook

```bash
jupyter lab "LLM Engineering Guide.ipynb"
```

Run the core cells in order. The notebook selects an available device and ran on CPU in the validation environment. Running all cells is not a guarantee that optional model downloads or integrations will work on every machine.

## Optional pretrained-model and adapter packages

Install the optional requirements only when you intend to experiment with those integrations:

```bash
python -m pip install -r requirements-optional.txt
```

The optional file lists `transformers`, `accelerate` and `peft`. The pretrained GPT-2 demonstration is guarded by `RUN_PRETRAINED_DEMO = False`; enable it manually only after installing the relevant packages and confirming network access, disk/cache space and the model's terms of use. The real image-captioning example likewise requires Transformers, Pillow, a model download and a permitted local image.

`bitsandbytes` is intentionally commented out in `requirements-optional.txt` because 4-bit quantization support varies by platform, CUDA environment and package version. Installing the optional requirements alone does **not** turn the notebook into a complete PEFT/QLoRA training pipeline.

## Saved GPT checkpoint

The supplied `tiny_gpt_checkpoint.pt` was inspected using restricted PyTorch loading (`weights_only=True`) and successfully loaded with `strict=True` into the notebook's `TinyGPT` class when reconstructed with the matching configuration. Its top-level contents are `model_state_dict`, `vocab` and `block_size`; the vocabulary has 29 characters and the context block size is 48. The checkpoint is a small educational model artifact, **not a pretrained general-purpose LLM**. See the [Validation Report](<Validation Report.md>) for the test details.

**Important:** the notebook's training cell saves to `tiny_gpt_checkpoint.pt` in the kernel's current working directory and overwrites a same-named file. Back up the existing checkpoint before running that training cell if you want to preserve the supplied artifact.

## Repository layout

The following is the intended public-repository layout for the supplied project assets. The machine-specific `.venv/` directory is deliberately excluded.

```text
project-root/
├── LLM Engineering Guide.ipynb
├── README.md
├── requirements.txt
├── requirements-optional.txt
├── tiny_gpt_checkpoint.pt
└── Validation Report.md
```

No GitHub remote URL or repository name was supplied, so none is assumed here. No `LICENSE` file was present in the inspected project assets; the project's license has therefore not been established by these files.

## Limitations

- The small GPT trains on a tiny, repetitive teaching corpus; it is not a capable chatbot or meaningful benchmark model.
- In the fresh CPU run, training loss decreased while sampled validation loss increased, consistent with overfitting in this tiny setup.
- The RAG demo is local sparse TF–IDF retrieval with a rule-based prompt preview, not a dense-embedding/vector-database stack or a live RAG answer-generation service.
- PEFT/QLoRA, alignment training, real pretrained vision-language inference and distributed production serving are not freshly executed by the core notebook.
- The validation report documents what was executed and what remains untested.

## Validation

See [Validation Report.md](<Validation Report.md>) for the audit scope, environment, cell-level execution results, checkpoint assessment, observed warning and limitations.
