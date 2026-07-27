# IdentityStory: Taming Your Identity-Preserving Generator for Human-Centric Story Generation (AAAI 2026)

[![Page](https://img.shields.io/badge/Project-Page-green?logo=github&logoColor=white)](https://correr-zhou.github.io/IdentityStory/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2512.23519)

<br>

**Abstract**

Recent visual generative models enable story generation with consistent characters from text, but human-centric story generation faces additional challenges, such as maintaining detailed and diverse human face consistency and coordinating multiple characters across different images. This paper presents IdentityStory, a framework for human-centric story generation that ensures consistent character identity across multiple sequential images. By taming identity-preserving generators, the framework features two key components: Iterative Identity Discovery, which extracts cohesive character identities, and Re-denoising Identity Injection, which re-denoises images to inject identities while preserving desired context. Experiments on the ConsiStory-Human benchmark demonstrate that IdentityStory outperforms existing methods, particularly in face consistency, and supports multi-character combinations. The framework also shows strong potential for applications such as infinite-length story generation and dynamic character composition.

## 🛠️ Environment Setup

We recommend using a clean Conda environment with Python 3.10:

```bash
git clone https://github.com/Correr-Zhou/IdentityStory.git
cd IdentityStory

conda create -n identitystory python=3.10 -y
conda activate identitystory

pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/IDEA-Research/GroundingDINO.git
```

If the default PyTorch installation does not match your CUDA version, reinstall PyTorch manually. For example, for CUDA 12.4:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1 torchvision==0.19.1
```

GroundingDINO is installed after the base requirements because its build step imports PyTorch.
If the direct GitHub install is interrupted by network issues, clone the official GroundingDINO repository manually and run `pip install --no-build-isolation -e .` inside that checkout.
The full ID Injection workflow requires a CUDA GPU because GroundingDINO and SAM mask extraction run on GPU.

## 📦 Data and Model Preparation

IdentityStory downloads the PhotoMaker adapter and the default SDXL base model, [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), from Hugging Face when they are first used. The ID Injection stage also needs GroundingDINO and SAM checkpoints:

```text
IdentityStory/
├── models/
│   ├── GroundingDINO/
│   │   ├── GroundingDINO_SwinB.cfg.py
│   │   └── groundingdino_swinb_cogcoor.pth
│   └── sam/
│       └── sam_vit_h_4b8939.pth
└── bench/
    ├── ConsiStory-Human-single/
    ├── ConsiStory-Human-multi-2/
    └── ConsiStory-Human-multi-3/
```

Download the external checkpoints from their official releases:

```bash
mkdir -p models/GroundingDINO models/sam

wget -O models/GroundingDINO/GroundingDINO_SwinB.cfg.py \
  https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinB_cfg.py

wget -O models/GroundingDINO/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth

wget -O models/sam/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

The benchmark JSON files are included under `bench/`. Each story JSON contains one or more characters and a list of target prompts:

| Field | Description |
| :--- | :--- |
| `character` | Character descriptions used for identity discovery. |
| `token` | Identity tokens that appear in the target prompts. |
| `prompts` | Story prompts used by the ID Injection stage. |
| `style` | Optional prefix used for region prompts. |
| `settings` | Optional per-prompt scene/action phrases. Multi-character stories use `style`, shortened character descriptions, and `settings` to build the injection prompt. |
| `settings_types` | Optional metadata describing each setting. It is included in the benchmark JSON files but is not required by the runtime. |

## ⚡ Quick Start

Run the single-character example:

```bash
bash scripts/run_identitystory_single.sh
```

Run the two-character example:

```bash
bash scripts/run_identitystory_multi.sh
```

The scripts assume your Python environment is already active. If you want the script to activate a Conda environment, set `CONDA_ENV_NAME`:

```bash
CONDA_ENV_NAME=identitystory bash scripts/run_identitystory_single.sh
```

The main entrypoint also supports direct command-line usage:

```bash
python main.py \
  --json_dir bench/ConsiStory-Human-single \
  --output_dir results/demo_single \
  --max_stories 1 \
  --prompt_indices 0 \
  --device cuda \
  --suffix demo_single
```

For multi-character generation, switch the JSON directory:

```bash
python main.py \
  --json_dir bench/ConsiStory-Human-multi-2 \
  --output_dir results/demo_multi \
  --max_stories 1 \
  --prompt_indices 0 \
  --device cuda \
  --suffix demo_multi
```

By default, outputs are organized as:

```text
results/demo_single/
├── id_search/
│   └── story_000/
│       └── id_000_<token>/
│           ├── final_id_embeds.pt
│           ├── photomaker_final_gen.png
│           └── prompt.txt
└── id_injection/
    └── story_000_prompt_000_demo_single/
        ├── stage-1.png
        ├── stage-2.png
        └── config.txt
```

## 🧭 Advanced Usage

The full workflow has two stages:

| Stage | Description | Main outputs |
| :--- | :--- | :--- |
| ID Search (Iterative Identity Discovery) | Generates candidate identity images and filters their embeddings with iterative SVD. | `final_id_embeds.pt`, `photomaker_final_gen.png` |
| ID Injection (Re-denoising Identity Injection) | Uses segmentation masks and re-denoising to inject discovered identities into story prompts. | `stage-1.png`, `stage-2.png` |

Commonly edited arguments:

```bash
--search_num_images 64
--search_batch_size 16
--search_num_inference_steps 50
--svd_components 10
--svd_keep_ratio 0.6
--svd_iter_num 3
--cfg_scale 7
--start_timestep 10
--max_dilation_kernel 50
```

To run only ID Search:

```bash
python main.py \
  --json_dir bench/ConsiStory-Human-single \
  --output_dir results/id_search_only \
  --skip_injection
```

To reuse existing ID Search outputs and run only ID Injection:

```bash
python main.py \
  --json_dir bench/ConsiStory-Human-single \
  --output_dir results/id_search_only \
  --skip_id_search
```

## 🧾 Preparing Your Own Stories

A single-character story JSON can be written as:

```json
{
  "character": "young woman with long black hair and fair skin",
  "token": "woman",
  "style": "A close-up photo of",
  "prompts": [
    "A woman walking through a quiet street.",
    "A woman reading a book near a window."
  ]
}
```

A multi-character story uses lists for `character` and `token`:

```json
{
  "character": [
    "young boy with curly black hair",
    "young girl with long brown hair"
  ],
  "token": ["boy", "girl"],
  "style": "A close-up photo of",
  "prompts": [
    "A boy and a girl sitting together in a classroom."
  ]
}
```

The identity tokens must appear in the story prompts so the injection stage can locate the corresponding regions.

## 🗂️ File Structure

```text
IdentityStory/
├── main.py                         # Unified ID Search and ID Injection entrypoint.
├── id_search/                      # PhotoMaker-based identity discovery.
│   ├── photomaker_pipeline.py
│   └── photomaker/
├── id_injection/                   # Re-denoising identity injection pipeline.
│   └── pipeline.py
├── utils/
│   ├── layout_extraction.py        # GroundingDINO + SAM mask extraction.
│   └── svd_filter.py               # Iterative SVD identity filtering.
├── bench/                          # ConsiStory-Human JSON benchmark samples.
├── scripts/                        # Example workflow launchers.
├── requirements.txt
└── README.md
```

## 🤝 Acknowledgements

This codebase builds on excellent open-source projects including [PhotoMaker](https://github.com/TencentARC/PhotoMaker), [Diffusers](https://github.com/huggingface/diffusers), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [Segment Anything](https://github.com/facebookresearch/segment-anything), [InsightFace](https://github.com/deepinsight/insightface), and [OMG](https://github.com/kongzhecn/OMG).

## 🔗 Citation

If IdentityStory is helpful for your research or projects, please consider citing our work:

```bibtex
@inproceedings{zhou2026identitystory,
  title={Identitystory: Taming your identity-preserving generator for human-centric story generation},
  author={Zhou, Donghao and Lin, Jingyu and Shen, Guibao and Liu, Quande and Gao, Jialin and Liu, Lihao and Du, Lan and Chen, Cunjian and Fu, Chi-Wing and Hu, Xiaowei and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={16},
  pages={13593--13601},
  year={2026}
}
```

## 📬 Contact

For questions about IdentityStory, please contact Donghao Zhou at [dhzhou@link.cuhk.edu.hk](mailto:dhzhou@link.cuhk.edu.hk).
