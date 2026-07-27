import argparse
import glob
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.svd_filter import iterative_svd_filter


DEFAULT_NEGATIVE_PROMPT = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch)"
DEFAULT_INJECTION_NEGATIVE_PROMPT = "noisy, blurry, soft, deformed, ugly"


@dataclass
class IdentitySpec:
    token: str
    character: str
    search_prompt: str
    output_dir: str


@dataclass
class StorySpec:
    name: str
    path: str
    story: Dict[str, Any]
    identities: List[IdentitySpec]


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("_") or "identity"


def add_trigger_to_character(character: str, token: str, trigger_word: str = "img") -> str:
    """Insert the PhotoMaker trigger word immediately after the first matching token."""
    character = character.strip()
    token = token.strip()
    if not character or not token:
        return character

    pattern = re.compile(rf"(?<!\w)({re.escape(token)})(?!\w)", flags=re.IGNORECASE)
    replaced, count = pattern.subn(rf"\1 {trigger_word}", character, count=1)
    if count:
        return replaced
    return f"{character} {token} {trigger_word}".strip()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_story_specs(json_dir: str, output_dir: str, trigger_word: str = "img") -> List[StorySpec]:
    specs: List[StorySpec] = []
    json_files = sorted(glob.glob(os.path.join(json_dir, "story_*.json")))

    for json_file in json_files:
        try:
            story = load_json(json_file)
        except Exception as exc:
            print(f"Error loading {json_file}: {exc}")
            continue

        tokens = [str(token).strip() for token in _as_list(story.get("token")) if str(token).strip()]
        characters = [str(character).strip() for character in _as_list(story.get("character")) if str(character).strip()]
        if not tokens or not characters:
            print(f"Skipping {json_file}: missing token or character")
            continue
        if len(tokens) != len(characters):
            print(f"Skipping {json_file}: token count {len(tokens)} != character count {len(characters)}")
            continue

        story_name = os.path.splitext(os.path.basename(json_file))[0]
        story_search_dir = os.path.join(output_dir, "id_search", story_name)
        identities = []
        for identity_idx, (token, character) in enumerate(zip(tokens, characters)):
            identities.append(
                IdentitySpec(
                    token=token,
                    character=character,
                    search_prompt=add_trigger_to_character(character, token, trigger_word),
                    output_dir=os.path.join(story_search_dir, f"id_{identity_idx:03}_{_safe_name(token)}"),
                )
            )

        specs.append(StorySpec(name=story_name, path=json_file, story=story, identities=identities))

    return specs


def image_grid(imgs, rows: int, cols: int, size_after_resize: int):
    from PIL import Image

    assert len(imgs) == rows * cols, "Number of images does not match grid dimensions."
    w, h = size_after_resize, size_after_resize
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        img = img.resize((w, h))
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def save_final_id_embeds(final_id_embeds, file_path: str) -> None:
    import torch

    torch.save(final_id_embeds, file_path)


def load_final_id_embeds(file_path: str):
    import torch

    try:
        return torch.load(file_path, weights_only=True)
    except TypeError:
        return torch.load(file_path)


def load_prompt_and_embeds(subdir: str):
    prompt_path = os.path.join(subdir, "prompt.txt")
    embeds_path = os.path.join(subdir, "final_id_embeds.pt")

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
    if not os.path.exists(embeds_path):
        raise FileNotFoundError(f"Final ID embeds not found at {embeds_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    return prompt, load_final_id_embeds(embeds_path)


def select_device() -> str:
    import torch

    try:
        if torch.cuda.is_available():
            return "cuda"
        if sys.platform == "darwin" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def build_photomaker_pipeline(base_model_path: str, trigger_word: str, device: Optional[str] = None):
    import torch
    from diffusers import EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from id_search.photomaker_pipeline import PhotoMakerIdentitySearchPipeline

    device = device or select_device()
    torch_dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    photomaker_ckpt = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
    pipe = PhotoMakerIdentitySearchPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        variant="fp16" if torch_dtype == torch.float16 else None,
    ).to(device)
    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_ckpt),
        subfolder="",
        weight_name=os.path.basename(photomaker_ckpt),
        trigger_word=trigger_word,
    )
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if device == "cuda":
        pipe.enable_model_cpu_offload()
    return pipe


def sample_image(
    pipe,
    input_prompt,
    input_neg_prompt=None,
    generator=None,
    concept_models=None,
    num_inference_steps=50,
    guidance_scale=3,
    controller=None,
    face_app=None,
    image=None,
    stage=None,
    region_masks=None,
    controlnet_conditioning_scale=None,
    args=None,
    **extra_kwargs,
):
    image_condition = [image] if image is not None else None
    return pipe(
        prompt=input_prompt,
        concept_models=concept_models,
        negative_prompt=input_neg_prompt,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        cross_attention_kwargs={"scale": 0.8},
        image=image_condition,
        face_app=face_app,
        stage=stage,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        region_masks=region_masks,
        args=args,
        **extra_kwargs,
    ).images


def generate_images(
    pipe,
    prompt: str,
    negative_prompt: str,
    final_id_embeds,
    num: int,
    num_batch: int,
    start_merge_step: int,
    num_inference_steps: int,
    guidance_scale: float,
):
    images = []
    pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    while len(images) < num:
        cur_num_batch = min(num_batch, num - len(images))
        cur_images = pipe.infer_united(
            prompt,
            negative_prompt=negative_prompt,
            final_id_embeds=final_id_embeds,
            num_images_per_prompt=cur_num_batch,
            start_merge_step=start_merge_step,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images
        images.extend(cur_images)
    return images[:num]


def process_identity_prompt(
    pipe,
    prompt: str,
    negative_prompt: str,
    output_subdir: str,
    *,
    num_generated_img: int,
    batch_size: int,
    total_loop: int,
    svd_components: int,
    svd_keep_ratio: float,
    svd_iter_num: int,
    final_grid_images: int,
    image_size: int,
    start_merge_step: int,
    num_inference_steps: int,
    guidance_scale: float,
):
    os.makedirs(output_subdir, exist_ok=True)
    final_gen_path = os.path.join(output_subdir, "photomaker_final_gen.png")
    final_embeds_path = os.path.join(output_subdir, "final_id_embeds.pt")
    prompt_txt_path = os.path.join(output_subdir, "prompt.txt")

    if os.path.exists(final_gen_path) and os.path.exists(final_embeds_path):
        print(f"Files already exist in {output_subdir}, skipping ID search.")
        return load_final_id_embeds(final_embeds_path)

    with open(prompt_txt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    final_id_embeds = None
    for loop in range(total_loop):
        print(f"ID search loop {loop + 1}/{total_loop}: {prompt}")
        images = generate_images(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            final_id_embeds=final_id_embeds,
            num=num_generated_img,
            num_batch=batch_size,
            start_merge_step=start_merge_step,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        final_id_embeds = pipe.get_final_id_embeds(images)
        embeddings = final_id_embeds.to(dtype=__import__("torch").float32).reshape(-1, 2048)
        filtered_embeddings = iterative_svd_filter(
            data_mat=embeddings,
            r=svd_components,
            ratio=svd_keep_ratio,
            iter_num=svd_iter_num,
        )
        final_id_embeds = filtered_embeddings.mean(dim=0, keepdim=True).reshape(1, 1, 1, -1)

    final_gen = pipe.infer_united(
        prompt,
        negative_prompt=negative_prompt,
        final_id_embeds=final_id_embeds,
        num_images_per_prompt=final_grid_images,
        start_merge_step=start_merge_step,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images
    grid = image_grid(final_gen, 1, final_grid_images, size_after_resize=image_size)
    grid.save(final_gen_path)
    save_final_id_embeds(final_id_embeds, final_embeds_path)
    return final_id_embeds


def build_injection_prompt(story: Dict[str, Any], prompt_idx: int) -> str:
    prompts = story.get("prompts") or []
    if prompt_idx < 0 or prompt_idx >= len(prompts):
        raise ValueError(f"prompt_idx {prompt_idx} out of range for {len(prompts)} prompts")

    tokens = [str(token).strip() for token in _as_list(story.get("token"))]
    characters = [str(character).strip() for character in _as_list(story.get("character"))]
    if len(tokens) > 1 and story.get("settings") and story.get("style"):
        settings = story["settings"]
        if prompt_idx >= len(settings):
            raise ValueError(f"prompt_idx {prompt_idx} out of range for {len(settings)} settings")
        descs = []
        for character, token in zip(characters, tokens):
            character = character.strip()
            if not character.startswith("a "):
                character = "a " + character
            match = re.search(rf"(?<!\w){re.escape(token)}(?!\w)", character, flags=re.IGNORECASE)
            descs.append(character[: match.end()].strip() if match else f"a {token}")
        return f"{story['style'].strip()} {' and '.join(descs)} {settings[prompt_idx]}".strip()

    return str(prompts[prompt_idx]).strip()


def _story_prompt_indices(story: Dict[str, Any], requested: Optional[Sequence[int]]) -> List[int]:
    prompts = story.get("prompts") or []
    if requested:
        indices = list(requested)
    else:
        indices = list(range(len(prompts)))
    for prompt_idx in indices:
        if prompt_idx < 0 or prompt_idx >= len(prompts):
            raise ValueError(f"prompt_idx {prompt_idx} out of range for {len(prompts)} prompts")
    return indices


def build_injection_models(args, device: str, width: int, height: int):
    import torch
    from diffusers import ControlNetModel
    from groundingdino.models import build_model
    from groundingdino.util.slconfig import SLConfig
    from groundingdino.util.utils import clean_state_dict
    from huggingface_hub import hf_hub_download
    from insightface.app import FaceAnalysis
    from id_search.photomaker_pipeline import PhotoMakerIdentitySearchPipeline
    from segment_anything import SamPredictor, build_sam
    from id_injection.pipeline import IdentityInjectionPipeline

    if args.segment_type != "GroundingDINO":
        raise ValueError(f"Unsupported segment_type: {args.segment_type}. Use GroundingDINO.")
    if device != "cuda":
        raise ValueError("ID Injection currently requires a CUDA device because GroundingDINO/SAM masks run on GPU.")

    def build_dino_segment_model(ckpt_repo_id, sam_checkpoint):
        ckpt_filename = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = os.path.join(ckpt_repo_id, "GroundingDINO_SwinB.cfg.py")
        dino_args = SLConfig.fromfile(ckpt_config_filename)
        model = build_model(dino_args)
        dino_args.device = "cpu"
        checkpoint = torch.load(os.path.join(ckpt_repo_id, ckpt_filename), map_location="cpu", weights_only=False)
        log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print("Model loaded from {} \n => {}".format(ckpt_filename, log))
        model.eval()

        sam_model = build_sam(checkpoint=sam_checkpoint)
        sam_model.cuda()
        return model, SamPredictor(sam_model)

    def build_model_sd(
        pretrained_model,
        controlnet_path,
        face_adapter,
        torch_device,
        prompts,
        antelopev2_path,
        latent_width,
        latent_height,
        style_lora,
        condition_checkpoint,
        adapter_ratio,
    ):
        controlnet = None
        if controlnet_path is not None and os.path.exists(controlnet_path):
            controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
        pipe = IdentityInjectionPipeline.from_pretrained(
            pretrained_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(torch_device)
        controller = None

        pipe_concept = PhotoMakerIdentitySearchPipeline.from_pretrained(
            pretrained_model,
            torch_dtype=torch.float16,
            variant="fp16",
        ).to(torch_device)
        photomaker_ckpt = hf_hub_download(
            repo_id="TencentARC/PhotoMaker",
            filename="photomaker-v1.bin",
            repo_type="model",
        )
        pipe_concept.load_photomaker_adapter(
            os.path.dirname(photomaker_ckpt),
            subfolder="",
            weight_name=os.path.basename(photomaker_ckpt),
            trigger_word="img",
        )
        pipe_concept.fuse_lora()

        if condition_checkpoint is not None and os.path.exists(condition_checkpoint):
            t2i_controlnet = ControlNetModel.from_pretrained(condition_checkpoint, torch_dtype=torch.float16).to(
                torch_device
            )
            pipe.controlnet2 = t2i_controlnet

        if style_lora is not None and os.path.exists(style_lora):
            pipe.load_lora_weights(style_lora, weight_name="pytorch_lora_weights.safetensors", adapter_name="style")
            pipe_concept.load_lora_weights(
                style_lora,
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="style",
            )

        app = None
        if antelopev2_path is not None and os.path.exists(antelopev2_path):
            app = FaceAnalysis(
                name="antelopev2",
                root=antelopev2_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))

        return pipe, controller, pipe_concept, app

    pipe, controller, pipe_concepts, face_app = build_model_sd(
        args.pretrained_model,
        args.controlnet_path,
        args.face_adapter_path,
        __import__("torch").device(device),
        None,
        args.antelopev2_path,
        width // 32,
        height // 32,
        args.style_lora,
        args.t2i_controlnet_path,
        args.adapter_ratio,
    )
    detect_model, sam = build_dino_segment_model(args.dino_checkpoint, args.sam_checkpoint)
    return pipe, controller, pipe_concepts, face_app, detect_model, sam


def run_injection_for_prompt(
    args,
    story_spec: StorySpec,
    prompt_idx: int,
    id_dirs: List[str],
    models,
    device: str,
):
    import torch
    from PIL import Image
    from utils.layout_extraction import predict_mask, process_masks, save_mask_tensor_as_image

    pipe, controller, pipe_concepts, face_app, detect_model, sam = models
    width, height = args.width, args.height
    if args.spatial_condition and os.path.exists(args.spatial_condition):
        spatial_condition = Image.open(args.spatial_condition).convert("RGB").resize((width, height))
        print("use pose condition")
    else:
        spatial_condition = None

    segment_words = [identity.token for identity in story_spec.identities]
    prompt = build_injection_prompt(story_spec.story, prompt_idx)
    input_prompt = [[prompt] * 2, []]
    for identity, id_dir in zip(story_spec.identities, id_dirs):
        region_prompt, _ = load_prompt_and_embeds(id_dir)
        if len(story_spec.identities) > 1 and story_spec.story.get("style"):
            region_prompt = f"{story_spec.story['style'].strip()} {region_prompt}"
        input_prompt[1].append([region_prompt, args.injection_negative_prompt, id_dir])

    for i, word in enumerate(segment_words):
        word_input_ids = pipe.tokenizer(word)["input_ids"][1]
        if word_input_ids not in pipe.tokenizer(prompt)["input_ids"]:
            raise ValueError(f'"{word}" does not exist in global prompt: "{prompt}"')
        if word_input_ids not in pipe.tokenizer(input_prompt[1][i][0])["input_ids"]:
            raise ValueError(f'"{word}" does not exist in region prompt: "{input_prompt[1][i][0]}"')

    save_dir = os.path.join(
        args.output_dir,
        "id_injection",
        f"{story_spec.name}_prompt_{prompt_idx:03}_{args.suffix}",
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"save images to: {save_dir}")
    if args.save_info:
        print(f"Injection prompt: {prompt}")

    common_kwargs = {
        "height": height,
        "width": width,
        "t2i_image": spatial_condition,
        "t2i_controlnet_conditioning_scale": args.controlNet_ratio,
    }

    seed = args.seed
    while True:
        image = sample_image(
            pipe,
            input_prompt=input_prompt,
            concept_models=pipe_concepts,
            input_neg_prompt=[args.injection_negative_prompt] * len(input_prompt),
            generator=torch.Generator(device).manual_seed(seed),
            controller=controller,
            face_app=face_app,
            controlnet_conditioning_scale=args.IdentityNet_rate,
            stage=1,
            guidance_scale=args.cfg_scale,
            args=args,
            **common_kwargs,
        )
        try:
            word_counts = dict(Counter(segment_words))
            word_masks = {}
            for word, cnt in word_counts.items():
                word_masks[word] = predict_mask(
                    detect_model,
                    sam,
                    image[0],
                    word,
                    cnt,
                    confidence=0.1,
                    threshold=0.5,
                )
        except Exception as exc:
            seed += 3533
            print(f"{story_spec.name}-{prompt_idx} segment failed ({exc}); regenerating with seed {seed}...")
            continue

        mask_list = []
        mask_indices = {word: 0 for word in word_masks}
        for word in segment_words:
            mask_list.append(word_masks[word][mask_indices[word]])
            mask_indices[word] += 1
        if len(mask_list) == len(segment_words):
            break
        seed += 1
        print(f"{story_spec.name}-{prompt_idx} mask count mismatch; regenerating with seed {seed}...")

    pcs_mask_list = process_masks(mask_list)
    mask_dir = os.path.join(save_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    for idx, (mask, pcs_mask) in enumerate(zip(mask_list, pcs_mask_list)):
        save_mask_tensor_as_image(mask, os.path.join(mask_dir, f"mask_{idx}_{segment_words[idx]}.png"))
        save_mask_tensor_as_image(pcs_mask, os.path.join(mask_dir, f"pcs_mask_{idx}_{segment_words[idx]}.png"))

    image = sample_image(
        pipe,
        input_prompt=input_prompt,
        concept_models=pipe_concepts,
        input_neg_prompt=[args.injection_negative_prompt] * len(input_prompt),
        generator=torch.Generator(device).manual_seed(seed),
        controller=controller,
        face_app=face_app,
        stage=2,
        controlnet_conditioning_scale=args.IdentityNet_rate,
        region_masks=pcs_mask_list,
        guidance_scale=args.cfg_scale,
        args=args,
        **common_kwargs,
    )
    image[0].save(os.path.join(save_dir, "stage-1.png"))
    image[1].save(os.path.join(save_dir, "stage-2.png"))

    with open(os.path.join(save_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write(f"story_json: {story_spec.path}\n")
        f.write(f"prompt_idx: {prompt_idx}\n")
        f.write(f"prompt: {prompt}\n")
        f.write(f"id_dirs: {id_dirs}\n")
        f.write(f"segment_words: {segment_words}\n")
        f.write(f"seed: {seed}\n")


def run_workflow(args) -> None:
    story_specs = load_story_specs(args.json_dir, args.output_dir, args.trigger_word)
    if args.max_stories is not None:
        story_specs = story_specs[: args.max_stories]
    print(f"Loaded {len(story_specs)} story specs from {args.json_dir}.")

    for story_spec in story_specs:
        print(f"\nStory: {story_spec.name}")
        for identity in story_spec.identities:
            print(f"  ID search [{identity.token}]: {identity.search_prompt} -> {identity.output_dir}")
        for prompt_idx in _story_prompt_indices(story_spec.story, args.prompt_indices):
            print(f"  Injection prompt {prompt_idx}: {build_injection_prompt(story_spec.story, prompt_idx)}")

    if args.dry_run:
        print("Dry run complete. No models were loaded and no images were generated.")
        return

    id_pipe = None
    if not args.skip_id_search:
        id_pipe = build_photomaker_pipeline(args.pretrained_model, args.trigger_word, args.device)
        for story_spec in story_specs:
            for identity in story_spec.identities:
                process_identity_prompt(
                    pipe=id_pipe,
                    prompt=identity.search_prompt,
                    negative_prompt=args.search_negative_prompt,
                    output_subdir=identity.output_dir,
                    num_generated_img=args.search_num_images,
                    batch_size=args.search_batch_size,
                    total_loop=args.search_total_loop,
                    svd_components=args.svd_components,
                    svd_keep_ratio=args.svd_keep_ratio,
                    svd_iter_num=args.svd_iter_num,
                    final_grid_images=args.search_final_grid_images,
                    image_size=args.search_grid_image_size,
                    start_merge_step=args.start_merge_step,
                    num_inference_steps=args.search_num_inference_steps,
                    guidance_scale=args.search_guidance_scale,
                )

    if args.skip_injection:
        print("Skipping injection stage.")
        return

    device = args.device or select_device()
    injection_models = build_injection_models(args, device, args.width, args.height)
    for story_spec in story_specs:
        id_dirs = [identity.output_dir for identity in story_spec.identities]
        for id_dir in id_dirs:
            load_prompt_and_embeds(id_dir)
        for prompt_idx in _story_prompt_indices(story_spec.story, args.prompt_indices):
            run_injection_for_prompt(args, story_spec, prompt_idx, id_dirs, injection_models, device)

    print("Unified ID search + injection workflow completed.")


def parse_prompt_indices(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Unified PhotoMaker SVD ID search and ID injection from story JSON.")

    parser.add_argument("--json_dir", default="bench/ConsiStory-Human-single", type=str)
    parser.add_argument("--output_dir", default="results/identitystory", type=str)
    parser.add_argument("--pretrained_model", default="stabilityai/stable-diffusion-xl-base-1.0", type=str)
    parser.add_argument("--trigger_word", default="img", type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_id_search", action="store_true")
    parser.add_argument("--skip_injection", action="store_true")
    parser.add_argument("--max_stories", default=None, type=int)
    parser.add_argument("--prompt_indices", type=parse_prompt_indices, default=None, help="Comma-separated prompt indices.")

    parser.add_argument("--search_num_images", default=64, type=int)
    parser.add_argument("--search_batch_size", default=16, type=int)
    parser.add_argument("--search_total_loop", default=1, type=int)
    parser.add_argument("--svd_components", default=10, type=int)
    parser.add_argument("--svd_keep_ratio", default=0.6, type=float)
    parser.add_argument("--svd_iter_num", default=3, type=int)
    parser.add_argument("--search_final_grid_images", default=4, type=int)
    parser.add_argument("--search_grid_image_size", default=1024, type=int)
    parser.add_argument("--search_num_inference_steps", default=50, type=int)
    parser.add_argument("--search_guidance_scale", default=5.0, type=float)
    parser.add_argument("--search_negative_prompt", default=DEFAULT_NEGATIVE_PROMPT, type=str)

    parser.add_argument("--controlnet_path", default="", type=str)
    parser.add_argument("--spatial_condition", default="", type=str)
    parser.add_argument("--t2i_controlnet_path", default="", type=str)
    parser.add_argument("--face_adapter_path", default="", type=str)
    parser.add_argument("--antelopev2_path", default="", type=str)
    parser.add_argument("--style_lora", default="", type=str)
    parser.add_argument("--IdentityNet_rate", default=0.8, type=float)
    parser.add_argument("--adapter_ratio", default=0.8, type=float)
    parser.add_argument("--controlNet_ratio", default=0.8, type=float)
    parser.add_argument("--dino_checkpoint", default="models/GroundingDINO", type=str)
    parser.add_argument("--sam_checkpoint", default="models/sam/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--segment_type", default="GroundingDINO", choices=["GroundingDINO"], type=str)
    parser.add_argument("--injection_negative_prompt", default=DEFAULT_INJECTION_NEGATIVE_PROMPT, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--suffix", default="unified", type=str)
    parser.add_argument("--cfg_scale", default=7.0, type=float)
    parser.add_argument("--start_merge_step", default=10, type=int)
    parser.add_argument("--start_timestep", default=10, type=int)
    parser.add_argument("--max_dilation_kernel", default=50, type=float)
    parser.add_argument("--width", default=1024, type=int)
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--save_info", action="store_true")
    return parser.parse_args()


def main(args=None) -> None:
    run_workflow(args or parse_args())


if __name__ == "__main__":
    main()
