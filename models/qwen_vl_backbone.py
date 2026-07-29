import torch
import logging
import torch.nn as nn

from transformers import AutoModelForImageTextToText
from peft import LoraConfig, TaskType, get_peft_model

from data_provider.file_loader import file_loader
from .layers import MLP
import os


class QwenVLBackbone(nn.Module):

    def __init__(
        self,
        device,
        model_path="/root/work/Qwen3.5-4B",
        torch_dtype=torch.bfloat16,
        use_lora=True,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        gradient_checkpointing=False,  
    ):
        logging.info("Start initializing QwenVLBackbone")
        super(QwenVLBackbone, self).__init__()

        self.device = device
        self.model_path = model_path
        self.model_dtype = torch_dtype

        self.road_cnt = file_loader.get_intersection_cnt()

        self.qwen = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch_dtype,  
            trust_remote_code=True,  
            low_cpu_mem_usage=True,  
        )


        self.qwen.config.use_cache = False
        if hasattr(self.qwen.config, "text_config"):
            self.qwen.config.text_config.use_cache = False

        self.hidden_size = self.qwen.config.text_config.hidden_size

        if gradient_checkpointing:
            if hasattr(self.qwen, "gradient_checkpointing_enable"):
                self.qwen.gradient_checkpointing_enable()
            if hasattr(self.qwen, "enable_input_require_grads"):
                self.qwen.enable_input_require_grads()

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,   
                lora_alpha=lora_alpha,   
                lora_dropout=lora_dropout,   
                bias="none",  
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )

            self.qwen = get_peft_model(self.qwen, lora_config)
            self.qwen.print_trainable_parameters()

        N_NODE = self.road_cnt
        N_OUT = N_NODE + 1  
        self.tasks_mlp = nn.ModuleDict({
            "road_clas": MLP(self.hidden_size, self.hidden_size, N_OUT),
        })

        logging.info(
            f"gradient_checkpointing arg = {gradient_checkpointing}, "
            f"qwen.is_gradient_checkpointing = "
            f"{getattr(self.qwen, 'is_gradient_checkpointing', 'N/A')}"
        )

        logging.info(
            f"Finish initializing QwenVLBackbone, hidden_size={self.hidden_size}"
        )
        # Visual processor 
        self._processor = None  
        
        self.visual_feature_dim = None  

        #self.image_resize_size = (224, 224)
        self.image_feature_cache_dir = "./cache/qwen35_vl_4b_imgfeat_224_pooler"
        os.makedirs(self.image_feature_cache_dir, exist_ok=True)

        self.image_feature_bank = None
        self.use_image_feature_bank = False
        self.image_feature_bank_strict = True

    
    def forward(self, x, activate_heads, attention_mask=None):

        x = x.to(dtype=self.model_dtype)

        qwen_outputs = self.qwen(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        hidden_states = getattr(qwen_outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                "Qwen output does not contain hidden_states. "
                "Please check output_hidden_states=True."
            )

        hidden = hidden_states[-1].float()

        outputs = {
            name: self.tasks_mlp[name](hidden)
            for name in activate_heads
            if name in self.tasks_mlp
        }

        outputs["hidden"] = hidden

        return outputs

    def _image_cache_path(self, image_path):
        import hashlib

        abs_path = os.path.abspath(image_path)
        key = hashlib.md5(abs_path.encode("utf-8")).hexdigest()
        return os.path.join(self.image_feature_cache_dir, key + ".pt")

    def preload_image_feature_bank(self, image_paths, strict=True):
        if image_paths is None:
            image_paths = []

        unique_paths = []
        seen = set()

        for p in image_paths:
            if p is None or not str(p).strip():
                continue

            abs_p = os.path.abspath(str(p))
            if abs_p not in seen:
                seen.add(abs_p)
                unique_paths.append(abs_p)

        logging.info(
            f"Start preloading image feature bank: {len(unique_paths)} unique images"
        )

        bank = {}
        missing = []
        failed = []

        for p in unique_paths:
            cache_path = self._image_cache_path(p)

            if not os.path.exists(cache_path):
                missing.append((p, cache_path))
                continue

            try:
                feat = torch.load(cache_path, map_location="cpu")
                feat = feat.float().view(-1).contiguous()

                bank[p] = feat

                if self.visual_feature_dim is None:
                    self.visual_feature_dim = feat.numel()
                    logging.info(
                        f"Visual feature dimension loaded from feature bank: "
                        f"{self.visual_feature_dim}"
                    )

            except Exception as e:
                failed.append((p, cache_path, str(e)))

        if self.visual_feature_dim is None and len(bank) > 0:
            first_feat = next(iter(bank.values()))
            self.visual_feature_dim = first_feat.numel()

        self.image_feature_bank = bank
        self.use_image_feature_bank = True
        self.image_feature_bank_strict = strict

        logging.info(
            f"Finish preloading image feature bank: "
            f"loaded={len(bank)}, missing={len(missing)}, failed={len(failed)}, "
            f"strict={strict}"
        )

        if missing or failed:
            msg = (
                f"Image feature bank preload incomplete. "
                f"loaded={len(bank)}, missing={len(missing)}, failed={len(failed)}. "
                f"Example missing={missing[:3]}, example failed={failed[:3]}. "
                f"Fallback to on-the-fly image feature extraction."
            )

            if strict:
                logging.warning(msg)
            else:
                logging.info(msg)

            self.image_feature_bank = None
            self.use_image_feature_bank = False
            self.image_feature_bank_strict = False

            return {
                "loaded": len(bank),
                "missing": len(missing),
                "failed": len(failed),
                "fallback_to_extract": True,
            }

        return {
            "loaded": len(bank),
            "missing": len(missing),
            "failed": len(failed),
            "fallback_to_extract": False,
        }


    def extract_image_features_from_bank(self, image_paths, strict=None):

        if strict is None:
            strict = self.image_feature_bank_strict

        if self.image_feature_bank is None:
            raise RuntimeError(
                "image_feature_bank is None. "
                "Please call preload_image_feature_bank(...) before training."
            )

        if not image_paths:
            return None

        if self.visual_feature_dim is None:
            if len(self.image_feature_bank) == 0:
                return None
            self.visual_feature_dim = next(iter(self.image_feature_bank.values())).numel()

        zero_feat = torch.zeros(
            self.visual_feature_dim,
            dtype=torch.float32,
        )

        empty_count = 0
        bank_miss_count = 0
        bank_miss_examples = []

        output_features = []

        for p in image_paths:
            if p is None or not str(p).strip():
                output_features.append(zero_feat)
                empty_count += 1
                continue

            abs_p = os.path.abspath(str(p))
            feat = self.image_feature_bank.get(abs_p, None)

            if feat is None:
                bank_miss_count += 1
                if len(bank_miss_examples) < 5:
                    bank_miss_examples.append(abs_p)

                if strict:
                    raise RuntimeError(
                        f"Image feature not found in CPU feature bank: {abs_p}"
                    )

                output_features.append(zero_feat)
            else:
                output_features.append(feat.float().view(-1))

        if not output_features:
            return None

        if bank_miss_count > 0:
            logging.warning(
                f"{bank_miss_count}/{len(image_paths)} non-empty image features "
                f"are missing in CPU feature bank. Zero features are used for them. "
                f"Examples: {bank_miss_examples}"
            )
        features = torch.stack(output_features, dim=0)
        features = features.to(self.device, dtype=torch.float32)

        return features.detach()



    def extract_image_features(self, image_paths):
    
        if getattr(self, "use_image_feature_bank", False):
            try:
                return self.extract_image_features_from_bank(
                    image_paths,
                    strict=True,
                )
            except RuntimeError as e:
                logging.warning(
                    f"Image feature bank lookup failed: {e}. "
                    f"Fallback to on-the-fly image feature extraction."
                )

                self.image_feature_bank = None
                self.use_image_feature_bank = False
                self.image_feature_bank_strict = False

        if not image_paths:
            return None

        from PIL import Image

        output_features = [None for _ in image_paths]

        images_to_extract = []
        indices_to_extract = []
        paths_to_extract = []

        for idx, p in enumerate(image_paths):
            if p is None or not str(p).strip():
                logging.debug(f"Image path is empty at index {idx}; zero feature will be used.")
                continue

            if not os.path.exists(p):
                logging.warning(f"Image path does not exist at index {idx}: {p}")
                continue

            cache_path = self._image_cache_path(p)

            if os.path.exists(cache_path):
                try:
                    feat = torch.load(cache_path, map_location="cpu")
                    feat = feat.float().view(-1)  # [D]
                    output_features[idx] = feat

                    if self.visual_feature_dim is None:
                        self.visual_feature_dim = feat.numel()
                        logging.info(
                            f"Visual feature dimension loaded from cache: "
                            f"{self.visual_feature_dim}"
                        )

                    continue
                except Exception as e:
                    logging.warning(
                        f"Failed to load cached image feature: "
                        f"path={p}, cache={cache_path}, error={e}. "
                        f"Will recompute it."
                    )

            try:
                img = Image.open(p).convert("RGB")
                #img = img.resize(self.image_resize_size)
                images_to_extract.append(img)
                indices_to_extract.append(idx)
                paths_to_extract.append(p)
            except Exception as e:
                logging.warning(f"Cannot load image at index {idx}, path={p}: {e}")

        if images_to_extract:
            if self._processor is None:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                )

            with torch.no_grad():
                inputs = self._processor(
                    images=images_to_extract,
                    return_tensors="pt",
                    padding=True,
                )

                pixel_values = inputs["pixel_values"].to(
                    self.device,
                    dtype=self.model_dtype,
                )

                grid_thw = inputs.get("image_grid_thw", None)
                if grid_thw is None:
                    raise RuntimeError(
                        "AutoProcessor did not return image_grid_thw. "
                        "Qwen3.5 vision encoder requires image_grid_thw/grid_thw."
                    )

                grid_thw = grid_thw.to(self.device, dtype=torch.long)

                visual_encoder = self.qwen.base_model.model.model.visual
                visual_encoder.eval()

                logging.info(
                    f"Visual encoder found: "
                    f"{type(visual_encoder).__module__}."
                    f"{type(visual_encoder).__qualname__}"
                )
                logging.info(
                    f"cache miss images={len(images_to_extract)}, "
                    f"total_paths={len(image_paths)}, "
                    f"pixel_values shape={tuple(pixel_values.shape)}, "
                    f"grid_thw shape={tuple(grid_thw.shape)}"
                )
            
                visual_outputs = visual_encoder(
                    hidden_states=pixel_values,
                    grid_thw=grid_thw,
                )

                use_merged_tokens = False

                if hasattr(visual_outputs, "pooler_output") and visual_outputs.pooler_output is not None:
                    visual_tokens = visual_outputs.pooler_output
                    use_merged_tokens = True
                elif hasattr(visual_outputs, "last_hidden_state") and visual_outputs.last_hidden_state is not None:
                    visual_tokens = visual_outputs.last_hidden_state
                    use_merged_tokens = False
                elif isinstance(visual_outputs, (list, tuple)):
                    if len(visual_outputs) >= 2 and visual_outputs[1] is not None:
                        visual_tokens = visual_outputs[1]
                        use_merged_tokens = True
                    else:
                        visual_tokens = visual_outputs[0]
                        use_merged_tokens = False
                else:
                    visual_tokens = visual_outputs
                    use_merged_tokens = False

                
                if visual_tokens.dim() == 2:
                    raw_token_counts = (
                        grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
                    )

                    spatial_merge_size = getattr(
                        visual_encoder,
                        "spatial_merge_size",
                        getattr(
                            getattr(visual_encoder, "config", None),
                            "spatial_merge_size",
                            2,
                        ),
                    )
                    spatial_merge_unit = spatial_merge_size * spatial_merge_size

                    if use_merged_tokens:
                        token_counts = raw_token_counts // spatial_merge_unit
                    else:
                        token_counts = raw_token_counts

                    token_counts = token_counts.detach().cpu().tolist()

                    if sum(token_counts) != visual_tokens.size(0):
                        alt_raw_counts = raw_token_counts.detach().cpu().tolist()
                        alt_merged_counts = (
                            raw_token_counts // spatial_merge_unit
                        ).detach().cpu().tolist()

                        if sum(alt_raw_counts) == visual_tokens.size(0):
                            token_counts = alt_raw_counts
                        elif sum(alt_merged_counts) == visual_tokens.size(0):
                            token_counts = alt_merged_counts
                        else:
                            raise RuntimeError(
                                f"Visual token count mismatch: "
                                f"sum(token_counts)={sum(token_counts)}, "
                                f"visual_tokens.size(0)={visual_tokens.size(0)}, "
                                f"grid_thw={grid_thw.detach().cpu().tolist()}, "
                                f"use_merged_tokens={use_merged_tokens}, "
                                f"spatial_merge_size={spatial_merge_size}"
                            )

                    chunks = visual_tokens.split(token_counts, dim=0)
                    extracted_features = torch.stack(
                        [chunk.mean(dim=0) for chunk in chunks],
                        dim=0,
                    )

                elif visual_tokens.dim() == 3:
                    extracted_features = visual_tokens.mean(dim=1)
                elif visual_tokens.dim() == 4:
                    extracted_features = visual_tokens.mean(dim=(1, 2))
                else:
                    extracted_features = visual_tokens.view(
                        visual_tokens.size(0), -1
                    ).mean(dim=1, keepdim=True)

                extracted_features = extracted_features.detach().float().cpu()

            for local_idx, orig_idx in enumerate(indices_to_extract):
                feat = extracted_features[local_idx].view(-1)

                output_features[orig_idx] = feat

                cache_path = self._image_cache_path(paths_to_extract[local_idx])
                try:
                    torch.save(feat, cache_path)
                except Exception as e:
                    logging.warning(
                        f"Failed to save image feature cache: "
                        f"path={paths_to_extract[local_idx]}, "
                        f"cache={cache_path}, error={e}"
                    )

                if self.visual_feature_dim is None:
                    self.visual_feature_dim = feat.numel()
                    logging.info(
                        f"Visual feature dimension: {self.visual_feature_dim}"
                    )

        valid_features = [f for f in output_features if f is not None]
        if not valid_features:
            return None

        feature_dim = valid_features[0].numel()

        if self.visual_feature_dim is None:
            self.visual_feature_dim = feature_dim
            logging.info(
                f"Visual feature dimension: {self.visual_feature_dim}"
            )

        final_features = []
        missing_count = 0

        for feat in output_features:
            if feat is None:
                final_features.append(torch.zeros(feature_dim, dtype=torch.float32))
                missing_count += 1
            else:
                final_features.append(feat.float().view(-1))


        features = torch.stack(final_features, dim=0)
        features = features.to(self.device, dtype=torch.float32)

        return features.detach()



    