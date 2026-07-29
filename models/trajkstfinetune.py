import os
import math
import torch
import torch.nn as nn

# from transformers import GPT2Tokenizer

from config.args_config import args

import torch.nn.functional as F
import logging
import time

import pickle
import math



def masked_mean_pooling(hidden, mask):
    # hidden: [B, T, D]
    # mask:   [B, T]
    mask = mask.unsqueeze(-1).type_as(hidden)                 # [B, T, 1]
    summed = (hidden * mask).sum(dim=1)                       # [B, D]
    denom = mask.sum(dim=1).clamp(min=1e-6)                   # [B, 1]
    return summed / denom



class TrajKSTFineTune(nn.Module):
    def __init__(self, device, checkpoint=None, freeze_st_tokenizer=None, kg_data=None, **kwargs):
        super(TrajKSTFineTune, self).__init__()

        self.node_cnt = 1839
        self.device = device
        self.use_direction = kwargs.get("use_direction", False)
        if self.use_direction:
            from models.visual_direction import VisualDirectionModule
            
            self.captures_root = kwargs.get("captures_root", None)
            road_network_path = kwargs.get("road_network_path", None)

            if self.captures_root is None:
                from config.global_vars import captures_dir
                self.captures_root = captures_dir

            if road_network_path is None:
                from config.global_vars import road_network_file
                road_network_path = road_network_file

            # Load node xy coordinates from road network
            with open(road_network_path, "rb") as f:
                G = pickle.load(f)
            self.node_xy = torch.zeros(self.node_cnt, 2, device=self.device, dtype=torch.float)
            
            for nid, attrs in G.nodes(data=True):
                if "xy" in attrs:
                    self.node_xy[nid] = torch.tensor(attrs["xy"], dtype=torch.float)
            del G

            num_heading_bins = kwargs.get("num_heading_bins", getattr(args, "num_heading_bins", 16))
            self.direction_bins = num_heading_bins
            self._build_direction_bin_lookup()

            track_obs_positions = True

            logging.info("Direction module enabled, node_xy loaded")

        if args.llm_backbone == "qwen_vl":
            from models.qwen_vl_backbone import QwenVLBackbone
            self.backbone = QwenVLBackbone(
                device=device,
                model_path=args.qwen_model_path,
                torch_dtype=torch.bfloat16,
                use_lora=True,
            ).to(device)
            self.d_model = self.backbone.hidden_size
        else:
            # from models.backbone import Backbone
            # self.backbone = Backbone(device).to(device)
            # self.d_model = 768
            raise RuntimeError(
                "Models other than Qwen-VL are not supported"
            )

        if getattr(self, "use_direction", False):
            if hasattr(self.backbone, "visual_feature_dim") and self.backbone.visual_feature_dim is not None:
                d_img = self.backbone.visual_feature_dim
            else:
                d_img = self.backbone.hidden_size
            self.visual_direction = VisualDirectionModule(
                d_img=d_img,
                d_hidden=256,
                d_model=self.d_model,  # 2560/768
                num_heading_bins=self.direction_bins,
                num_nodes=self.node_cnt,
            ).to(device)
            logging.info(f"VisualDirectionModule initialized (d_img={d_img}, bins={self.direction_bins})")


        self.mse = nn.MSELoss()
        
        self.eos_id = self.node_cnt  # 1839
        self.ignore_index = -100

        # Gap empty classifier 
        self.gap_classifier = nn.Linear(
            self.d_model, 2,
        )  

        self.node_embedding = nn.Embedding(
            self.node_cnt,
            self.d_model,
        )   

        self.num_delta_time_buckets = 10

        self.delta_time_embedding = nn.Embedding(
            self.num_delta_time_buckets,
            self.d_model,
        )

        self.time_scale = 3600.0

        self.delta_time_encoder = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # 0: PAD
        # 1: TASK_TRAJ_RECOVERY
        # 2: OBS
        # 3: KG
        # 4: RECOVER
        # 5: BOS
        # 6: SEP
        self.special_token = nn.Embedding(
            11,
            self.d_model,
        ).to(device)   

        self.SPAD = 0
        self.STASK = 1  # [TASK]
        self.SOBS = 2   # [OBS]
        self.SKG = 3   # [KG]
        self.SRECOVER = 4
        self.SBOS = 5
        self.SSEP = 6
        self.SGAP = 7
        self.SGAP_START = 8
        self.SGAP_END = 9
        self.SDELTA_TIME = 10

        self.cross_entropy = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
        )   


        self.kg_dim = 128
        if kg_data is not None:
            from models.kg_compgcn import KGCompGCN
            self.kg_compgcn = KGCompGCN(
                num_entities=kg_data["num_entities"],
                num_rel_types=kg_data["num_rel_types"],  # 27
                kg_dim=self.kg_dim,
            ).to(device)
            self.kg_proj = nn.Linear(self.kg_dim, self.d_model)
            self.road2eidx = kg_data["road2eidx"]  # road_id -> entity_idx

            self.register_buffer("kg_edge_index", kg_data["edge_index"])  # 边列表
            self.register_buffer("kg_edge_type", kg_data["edge_type"])  # 边关系
            
            road_entity_indices = torch.zeros(self.node_cnt, dtype=torch.long, device=device)  
            for road_id_int, eidx in self.road2eidx.items():
                if 0 <= int(road_id_int) < self.node_cnt:
                    road_entity_indices[int(road_id_int)] = eidx  # road_id -> entity_idx
            self.register_buffer("road_entity_indices", road_entity_indices)  
            self.entity_id2idx = kg_data.get("entity_id2idx", {})
        else:
            self.kg_compgcn = None
            self.kg_proj = None
            self.road2eidx = {}
            self.entity_id2idx = {}
            self.kg_edge_index = None
            self.kg_edge_type = None

        self.loaded_pretrain = False

        # Direction module 
        if checkpoint is not None:
            self.load_pretrain_checkpoint(checkpoint)
            self.loaded_pretrain = True

        
    def _build_direction_bin_lookup(self):
       
        node_xy = self.node_xy

        if not torch.is_tensor(node_xy):
            node_xy = torch.tensor(node_xy, dtype=torch.float32)

        node_xy = node_xy.to(self.device, dtype=torch.float32)

        x = node_xy[:, 0]
        y = node_xy[:, 1]

        dx = x.view(1, -1) - x.view(-1, 1)  # [N, N]
        dy = y.view(1, -1) - y.view(-1, 1)  # [N, N]

        theta = torch.atan2(dy, dx)

        bin_size = 2.0 * math.pi / self.direction_bins

        bins = torch.remainder(
            torch.round(theta / bin_size).long(),
            self.direction_bins,
        )

        self.register_buffer(
            "dir_bin_lookup",
            bins,
            persistent=False,
        )

        logging.info(
            f"Built dir_bin_lookup: shape={tuple(bins.shape)}, "
            f"device={bins.device}"
        )

    def load_pretrain_checkpoint(self, checkpoint=None):
        ckpt_path = checkpoint or f"./checkpoints/{args.city}_pretrain_best.pth"
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        pretrained_state = checkpoint["model_state_dict"]
        current_state = self.state_dict()
        load_state = {}

        for k, v in pretrained_state.items():
            if k not in current_state:
                continue

            if current_state[k].shape == v.shape:
                load_state[k] = v
                continue

            if k == "tokenizer.space_road" \
                    and current_state[k].dim() == 2 \
                    and v.dim() == 2 \
                    and current_state[k].shape[1] == v.shape[1] \
                    and current_state[k].shape[0] == v.shape[0] + 1:
                tmp = current_state[k].clone()
                tmp[:v.shape[0]].copy_(v.to(tmp.device))

                
                load_state[k] = tmp
                print(f"Partially loaded `{k}`: {tuple(v.shape)} -> {tuple(current_state[k].shape)}")
                continue

            print(f"Skip `{k}`: ckpt {tuple(v.shape)} != model {tuple(current_state[k].shape)}")

        msg = self.load_state_dict(load_state, strict=False)
        print(f"Loaded checkpoint from: {ckpt_path}")
        print("missing_keys:", msg.missing_keys)
        print("unexpected_keys:", msg.unexpected_keys)

    def _to_int_list(self, seq):
        return [int(x) for x in seq]

    def _flatten_kg_path_nodes(self, kg_item):
        
        if kg_item is None:
            return []

        if len(kg_item) == 0:
            return []

        first = kg_item[0]

        if isinstance(first, (int, float)) or hasattr(first, "item"):
            return [int(x) for x in kg_item]

        flat = []
        for path in kg_item:
            if len(flat) > 0:  
                flat.append(("special", self.SSEP))
            for node in path:
                flat.append(("node", int(node)))

        return flat

    
    def _build_gap_context_embeddings(
            self,
            observed_nodes,  
            observed_times,  
            gap_start_nodes,  
            gap_end_nodes,
            gap_delta_time,
            device,
            entity_emb=None,
            dir_embed=None,
    ):
        batch_tokens = []

        for obs_seq, time_seq, s_node, e_node, dt in zip(
                observed_nodes,
                observed_times,
                gap_start_nodes,
                gap_end_nodes,
                gap_delta_time,
        ):
            item = []
            
            item.append(("special", self.STASK, None))
            item.append(("special", self.SGAP, None))

            item.append(("special", self.SOBS, None))

            obs_seq = self._to_int_list(obs_seq)
            time_seq = self._to_int_list(time_seq)

            assert len(obs_seq) == len(time_seq)

            prev_t = None

            for node, t in zip(obs_seq, time_seq):
                if prev_t is None:
                    local_dt = 0
                else:
                    local_dt = t - prev_t

                item.append(("node_time", node, (t, local_dt)))
                prev_t = t

            item.append(("special", self.SGAP_START, None))
            item.append(("node", int(s_node), None))

            item.append(("special", self.SGAP_END, None))
            item.append(("node", int(e_node), None))

            item.append(("special", self.SDELTA_TIME, None))
            item.append(("delta_time", None, float(dt)))


            item.append(("special", self.SRECOVER, None))

            """
            [('special', 1, None), ('special', 7, None), 
            ('special', 2, None), ('node_time', 698, (19, 0)), ('node_time', 701, (22, 3)), ('node_time', 684, (23, 1)), ('node_time', 1561, (28, 5)), 
            ('special', 8, None), ('node', 698, None), 
            ('special', 9, None), ('node', 701, None), 
            ('special', 10, None), ('delta_time', None, 3.0), 
            ('special', 4, None)]
            """

            batch_tokens.append(item)

        max_len = max(len(x) for x in batch_tokens)
        B = len(batch_tokens)

        context_embeds = torch.zeros(
            B,
            max_len,
            self.d_model,
            device=device,
        )

        context_mask = torch.zeros(
            B,
            max_len,
            dtype=torch.long,
            device=device,
        )

        for i, item in enumerate(batch_tokens):
            obs_idx = 0
            for j, (kind, value, extra) in enumerate(item):

                if kind == "special":
                    special_id = torch.tensor(
                        value,
                        dtype=torch.long,
                        device=device,
                    )
                    context_embeds[i, j] = self.special_token(special_id)

                elif kind == "node":
                    node_id = torch.tensor(
                        value,
                        dtype=torch.long,
                        device=device,
                    )
                    emb = self.node_embedding(node_id)
                    if entity_emb is not None:
                        road_id = int(value)
                        if road_id in self.road2eidx:
                            kg_idx = self.road2eidx[road_id] 
                            emb = emb + self.kg_proj(entity_emb[kg_idx])
                    if dir_embed is not None:
                        for obs_i, obs_nid in enumerate(observed_nodes[i]):
                            if int(obs_nid) == int(value):
                                if obs_i < dir_embed.size(1):
                                    emb = emb + dir_embed[i, obs_i]
                                break
                    context_embeds[i, j] = emb

                elif kind == "node_time":
                    node_id = torch.tensor(
                        value,
                        dtype=torch.long,
                        device=device,
                    )

                    emb = self.node_embedding(node_id)
                    if entity_emb is not None:
                        road_id = int(value)
                        if road_id in self.road2eidx:
                            kg_idx = self.road2eidx[road_id]
                            emb = emb + self.kg_proj(entity_emb[kg_idx])
                    if dir_embed is not None and obs_idx < dir_embed.size(1):
                        emb = emb + dir_embed[i, obs_idx]
                    obs_idx += 1
                    context_embeds[i, j] = emb

                elif kind == "delta_time":
                    delta_feat = torch.tensor(
                        [float(extra) / self.time_scale],
                        dtype=torch.float,
                        device=device,
                    )

                    context_embeds[i, j] = self.delta_time_encoder(delta_feat)

                context_mask[i, j] = 1

        return context_embeds, context_mask

    def _build_decoder_embeddings_and_labels(
            self,
            target_nodes,
            device,
    ):
        
        targets = [
            self._to_int_list(seq)
            for seq in target_nodes
        ]  

        max_target_len = max(len(x) for x in targets)

        decoder_len = max_target_len + 1
        B = len(targets)

        decoder_embeds = torch.zeros(
            B,
            decoder_len,
            self.d_model,
            device=device,
        )

        decoder_mask = torch.zeros(
            B,
            decoder_len,
            dtype=torch.long,
            device=device,
        )

        labels = torch.full(
            (B, decoder_len),
            fill_value=self.ignore_index,
            dtype=torch.long,
            device=device,
        )

        bos_id = torch.tensor(
            self.SBOS,
            dtype=torch.long,
            device=device,
        )

        for i, tgt in enumerate(targets):
            T = len(tgt)  

            
            decoder_embeds[i, 0] = self.special_token(bos_id)  # [BOS]
            decoder_mask[i, 0] = 1  

            if T > 0:
                tgt_tensor = torch.tensor(
                    tgt,
                    dtype=torch.long,
                    device=device,
                )

                # decoder input: BOS, y1, ..., yT
                decoder_embeds[i, 1:T + 1] = self.node_embedding(tgt_tensor)
                decoder_mask[i, 1:T + 1] = 1

                # labels: y1 ... yT EOS  -> -100, y2,y3 ... EOS
                labels[i, :T] = tgt_tensor
                labels[i, T] = self.eos_id  # 1839

        return decoder_embeds, decoder_mask, labels

    def _build_decoder_prefix_embeddings(self, prefix_nodes, device):
        B = len(prefix_nodes)
        max_prefix_len = max(len(x) for x in prefix_nodes)
        decoder_len = max_prefix_len + 1  # +1 for BOS

        decoder_embeds = torch.zeros(
            B,
            decoder_len,
            self.d_model,
            device=device,
        )

        decoder_mask = torch.zeros(
            B,
            decoder_len,
            dtype=torch.long,
            device=device,
        )

        bos_id = torch.tensor(
            self.SBOS,
            dtype=torch.long,
            device=device,
        )

        for i, prefix in enumerate(prefix_nodes):
            decoder_embeds[i, 0] = self.special_token(bos_id)
            decoder_mask[i, 0] = 1

            if len(prefix) > 0:
                prefix_tensor = torch.tensor(
                    prefix,
                    dtype=torch.long,
                    device=device,
                )
                decoder_embeds[i, 1:len(prefix) + 1] = self.node_embedding(prefix_tensor)
                decoder_mask[i, 1:len(prefix) + 1] = 1

        return decoder_embeds, decoder_mask

    def _get_start_nodes_from_observed(self, observed_nodes):
        start_nodes = []

        for seq in observed_nodes:
            seq = self._to_int_list(seq)
            if len(seq) == 0:
                raise ValueError("observed_nodes contains empty sequence, cannot force start.")
            start_nodes.append(seq[0])

        return start_nodes

    @torch.no_grad()
    def generate_trajectory(
            self,
            observed_nodes,
            timestamps,
            gap_start_nodes,
            gap_end_nodes,
            gap_delta_time,
            max_gen_len=64,
            min_gen_len=0,
            dir_prob=None,
            **gen_kwargs_eval,
    ):
        if self.kg_compgcn is not None:
            entity_emb = self.kg_compgcn(self.kg_edge_index, self.kg_edge_type)
        else:
            entity_emb = None

        dir_embed = None
        dir_prob_all = dir_prob
        batch_image_names = gen_kwargs_eval.get("batch_image_names", None)
        batch_direction_bins = gen_kwargs_eval.get("batch_direction_bins", None)


        if self.use_direction and batch_image_names is not None:
            dir_kwargs = {
                "batch_image_names": batch_image_names,
                "batch_direction_bins": batch_direction_bins,
            }

            dir_embed, dir_prob_new, dir_logits, dir_loss = self._extract_direction_features(
                dir_kwargs,
                observed_nodes,
                self.device,
            )

            if dir_prob_all is None:
                dir_prob_all = dir_prob_new


        context_embeds, context_mask = self._build_gap_context_embeddings(
            observed_nodes=observed_nodes,
            observed_times=timestamps,
            gap_start_nodes=gap_start_nodes,
            gap_end_nodes=gap_end_nodes,
            gap_delta_time=gap_delta_time,
            device=self.device,
            entity_emb=entity_emb,
            dir_embed=dir_embed,
        )

        B = len(observed_nodes)
        ctx_len = context_embeds.size(1)

        gap_outputs = self.backbone(
            context_embeds, ["road_clas"], attention_mask=context_mask,
        )
        gap_hidden = gap_outputs["hidden"]  # [B, ctx_len, D]
        gap_pooled = masked_mean_pooling(gap_hidden, context_mask)  # [B, D]
        gap_cls_logits = self.gap_classifier(gap_pooled)  # [B, 2]
        gap_cls_pred = torch.argmax(gap_cls_logits, dim=-1)  # 0=non-empty, 1=empty
        # print("gap cls pred empty:", (gap_cls_pred == 1).sum().item())
        # print("gap cls pred non-empty:", (gap_cls_pred == 0).sum().item())

       
        prefix_nodes = [[] for _ in range(B)]

        #  EOS
        predictions = [[] for _ in range(B)]

        finished = [bool(gap_cls_pred[i].item()) for i in range(B)]
       
        for step in range(max_gen_len):
            decoder_embeds, decoder_mask = self._build_decoder_prefix_embeddings(
                prefix_nodes=prefix_nodes,
                device=self.device,
            )

            gpt_input_embeds = torch.cat(
                [context_embeds, decoder_embeds],
                dim=1,
            )

            attention_mask = torch.cat(
                [context_mask, decoder_mask],
                dim=1,
            )

            outputs = self.backbone(
                gpt_input_embeds,
                ["road_clas"],
                attention_mask=attention_mask,
            )

            logits = outputs["road_clas"]  # [B, L_total, 1840]

            step_logits = []

            for i in range(B):
                last_decoder_pos = len(prefix_nodes[i])  # BOS 
                pos = ctx_len + last_decoder_pos
                step_logits.append(logits[i, pos])

            step_logits = torch.stack(step_logits, dim=0)  # [B, 1840]

            # KG bias
            if self.kg_compgcn is not None and entity_emb is not None:
                entity_emb_road = entity_emb[self.road_entity_indices]  # [1839, kg_dim]
                entity_emb_road_norm = F.normalize(entity_emb_road, p=2, dim=-1)

                prev_ids = torch.zeros(B, dtype=torch.long, device=self.device)
                for i_ in range(B):
                    if finished[i_]:
                        prev_ids[i_] = 0  
                    elif len(prefix_nodes[i_]) == 0:
                        prev_ids[i_] = int(gap_start_nodes[i_])
                    else:
                        prev_ids[i_] = prefix_nodes[i_][-1]

                prev_kg_indices = self.road_entity_indices[prev_ids]  # [B]
                prev_emb = entity_emb[prev_kg_indices]  # [B, kg_dim]
                prev_emb_norm = F.normalize(prev_emb, p=2, dim=-1)  # [B, kg_dim]

                kg_bias = torch.mm(prev_emb_norm, entity_emb_road_norm.t())

                kg_bias_padded = F.pad(kg_bias, (0, 1), value=0.0)  # [B, 1840]

                for i_ in range(B):
                    if finished[i_]:
                        kg_bias_padded[i_].zero_()

                step_logits = step_logits + args.kg_bias_weight * kg_bias_padded
            
            # Direction bias
            if step == 0 and self.use_direction and dir_prob_all is not None:
                try:
                    if dir_prob_all.dim() == 3:
                        heading_bias_step = dir_prob_all[:, 0, :]  # [B, K]
                    else:
                        heading_bias_step = dir_prob_all            # [B, K]

                    heading_bias_step = heading_bias_step.to(self.device)

                    B_gen = step_logits.size(0)

                    start_ids = torch.as_tensor(
                        gap_start_nodes[:B_gen],
                        dtype=torch.long,
                        device=self.device,
                    )

                    valid_start = (start_ids >= 0) & (start_ids < self.node_cnt)
                    safe_start_ids = start_ids.clamp(0, self.node_cnt - 1)

                    bins = self.dir_bin_lookup[safe_start_ids]

                    road_bias = torch.gather(
                        heading_bias_step,
                        dim=1,
                        index=bins,
                    )

                    road_bias = road_bias * valid_start.unsqueeze(1).to(road_bias.dtype)

                    active_mask = torch.tensor(
                        [not x for x in finished],
                        dtype=road_bias.dtype,
                        device=self.device,
                    )
                    road_bias = road_bias * active_mask.unsqueeze(1)

                    dir_bias_pad = F.pad(road_bias, (0, 1), value=0.0)

                    dir_w = args.dir_bias_weight

                    step_logits = step_logits + dir_w * dir_bias_pad.to(dtype=step_logits.dtype)

                except Exception as e:
                    logging.warning(f"Direction bias in generate error: {e}")
                    import traceback
                    traceback.print_exc()

            #  fusiion
            if step == 0:
                inter_w = args.fusion_bias_weight
                interaction_gen = kg_bias_padded * dir_bias_pad  # [B, 1840]
                step_logits += inter_w * interaction_gen
            

            for i in range(B):
                if finished[i]:
                    continue
                if not finished[i]:
                    if len(predictions[i]) < min_gen_len:
                        step_logits[i, self.eos_id] = torch.finfo(step_logits.dtype).min
                


            next_ids = torch.argmax(step_logits, dim=-1)  # [B]

            for i in range(B):
                if finished[i]:
                    continue

                next_id = int(next_ids[i].detach().cpu().item())

                if next_id == self.eos_id:
                    finished[i] = True
                    continue

                predictions[i].append(next_id)
                prefix_nodes[i].append(next_id)

            if all(finished):
                break

        return predictions

   
    def _extract_direction_features(self, kwargs, batch_observed_nodes, device):
        
        dir_loss = torch.tensor(0.0, device=device)
        dir_prob = None
        dir_logits = None
        dir_embed = None

        if not self.use_direction:
            return dir_embed, dir_prob, dir_logits, dir_loss

        batch_image_names = kwargs.get('batch_image_names', None)
        if batch_image_names is None:
            return dir_embed, dir_prob, dir_logits, dir_loss

        try:
            batch_direction_labels = kwargs.get('batch_direction_bins', None)

            all_paths = []
            obs_counts = []

            for sample_imgs in batch_image_names:
                
                obs_counts.append(len(sample_imgs))

                for fname in sample_imgs:
                    if fname and str(fname).strip():
                        all_paths.append(os.path.join(self.captures_root, fname))
                    else:
                        all_paths.append(None)


            if not all_paths:
                return dir_embed, dir_prob, dir_logits, dir_loss

            img_features = self.backbone.extract_image_features(all_paths)
            if img_features is None:
                return dir_embed, dir_prob, dir_logits, dir_loss

            B = len(batch_image_names)
            max_obs = max(obs_counts) if obs_counts else 0
            if max_obs == 0:
                return dir_embed, dir_prob, dir_logits, dir_loss

            feat_tensor = torch.zeros(B, max_obs, img_features.size(-1), device=device)
            nodeid_tensor = torch.zeros(B, max_obs, dtype=torch.long, device=device)
            xy_tensor = torch.zeros(B, max_obs, 2, device=device)

            obs_offsets = [0]
            for c in obs_counts:
                obs_offsets.append(obs_offsets[-1] + c)

            for i in range(B):
                n_obs = obs_counts[i]
                if n_obs == 0:
                    continue
                feat_tensor[i, :n_obs] = img_features[obs_offsets[i]:obs_offsets[i+1]]
                for j in range(n_obs):
                    nid = int(batch_observed_nodes[i][j]) if j < len(batch_observed_nodes[i]) else 0
                    nodeid_tensor[i, j] = nid
                    if nid < self.node_cnt:
                        xy_tensor[i, j] = self.node_xy[nid]

            dir_logits_out, dir_prob_out, dir_embed_out = self.visual_direction(
                feat_tensor, nodeid_tensor, xy_tensor,
            )
            dir_logits = dir_logits_out
            dir_prob = dir_prob_out
            dir_embed = dir_embed_out

            # Direction loss
            if batch_direction_labels is not None:
                dir_labels = torch.full((B, max_obs), -100, dtype=torch.long, device=device)
                for i in range(B):
                    labels_i = batch_direction_labels[i] if isinstance(batch_direction_labels[i], (list, tuple)) else []
                    for j in range(min(len(labels_i), max_obs)):
                        lbl = int(labels_i[j])
                        if lbl >= 0:
                            dir_labels[i, j] = lbl

                valid = dir_labels != -100
                if valid.any():
                    dir_loss = F.cross_entropy(
                        dir_logits[valid], dir_labels[valid],
                        ignore_index=-100,
                    )

        except Exception as e:
            logging.warning(f'Direction extraction error: {e}')
            import traceback
            traceback.print_exc()

        return dir_embed, dir_prob, dir_logits, dir_loss

    def forward(self, task_name, batch_sample_id, batch_observed_nodes, batch_target_nodes,
                batch_timestamps, batch_gap_start_nodes, batch_gap_end_nodes, batch_gap_delta_time,
                **kwargs):

        #  Run CompGCN to get KG entity embeddings 
        if self.kg_compgcn is not None:
            entity_emb = self.kg_compgcn(self.kg_edge_index, self.kg_edge_type)
        else:
            entity_emb = None


        # Extract direction features 
        dir_embed, dir_prob_all, dir_logits, dir_loss = self._extract_direction_features(
            kwargs, batch_observed_nodes, self.device,
        )

        # 构建context
        context_embeds, context_mask = self._build_gap_context_embeddings(
            observed_nodes=batch_observed_nodes,
            observed_times=batch_timestamps,
            gap_start_nodes=batch_gap_start_nodes,
            gap_end_nodes=batch_gap_end_nodes,
            gap_delta_time=batch_gap_delta_time,
            device=self.device,
            entity_emb=entity_emb,
            dir_embed=dir_embed,
        )

    
        # 构建decoder
        decoder_embeds, decoder_mask, labels = self._build_decoder_embeddings_and_labels(
            target_nodes=batch_target_nodes,
            device=self.device,
        )

        gpt_input_embeds = torch.cat(
            [context_embeds, decoder_embeds],
            dim=1,
        )  

        attention_mask = torch.cat(
            [context_mask, decoder_mask],
            dim=1,
        ) 


        outputs = self.backbone(
            gpt_input_embeds,
            ["road_clas"],
            attention_mask=attention_mask,
        )


        logits = outputs["road_clas"]  

        decoder_len = decoder_embeds.size(1) 

        decoder_logits = logits[:, -decoder_len:, :] 

        
        if self.use_direction and dir_prob_all is not None:
            try:
                heading_bias = dir_prob_all[:, 0, :]

                B_dec = decoder_logits.size(0)

                start_ids = torch.as_tensor(
                    batch_gap_start_nodes[:B_dec],
                    dtype=torch.long,
                    device=self.device,
                )

                valid_start = (start_ids >= 0) & (start_ids < self.node_cnt)
                safe_start_ids = start_ids.clamp(0, self.node_cnt - 1)

                bins = self.dir_bin_lookup[safe_start_ids]

                road_bin_bias = torch.gather(
                    heading_bias,
                    dim=1,
                    index=bins,
                )

                road_bin_bias = road_bin_bias * valid_start.unsqueeze(1).to(road_bin_bias.dtype)

                dir_bias_padded = F.pad(road_bin_bias, (0, 1), value=0.0)

                dir_weight = args.dir_bias_weight
                dir_bias_padded = dir_bias_padded.to(dtype=decoder_logits.dtype)

                decoder_logits[:, 0:1, :] = decoder_logits[:, 0:1, :] + dir_weight * dir_bias_padded.unsqueeze(1)

            except Exception as e:
                logging.warning(f"Direction bias error: {e}")
                import traceback
                traceback.print_exc()

        

        if self.kg_compgcn is not None and entity_emb is not None:
            dec_len = decoder_logits.size(1)  # T+1
            B_size = decoder_logits.size(0)

            prev_node_ids = torch.zeros(B_size, dec_len, dtype=torch.long, device=self.device)
            valid_mask = torch.zeros(B_size, dec_len, dtype=torch.bool, device=self.device)

            for i_idx in range(B_size):
                prev_node_ids[i_idx, 0] = int(batch_gap_start_nodes[i_idx])
            valid_mask[:, 0] = True

            if dec_len > 1:
                valid_mask[:, 1:] = (labels[:, :dec_len - 1] != self.ignore_index) & (labels[:, :dec_len - 1] < self.node_cnt)
                prev_node_ids[:, 1:] = labels[:, :dec_len - 1].clamp(min=0, max=self.node_cnt - 1)

            
            entity_emb_road = entity_emb[self.road_entity_indices]  
            entity_emb_road_norm = F.normalize(entity_emb_road, p=2, dim=-1)  

            prev_kg_indices = self.road_entity_indices[prev_node_ids] 
            prev_emb = entity_emb[prev_kg_indices]  
            prev_emb_norm = F.normalize(prev_emb, p=2, dim=-1)

            kg_bias = torch.matmul(prev_emb_norm, entity_emb_road_norm.t())

            kg_bias = kg_bias * valid_mask.unsqueeze(-1).float()

            kg_bias_padded = F.pad(kg_bias, (0, 1), value=0.0)  # [B, dec_len, 1840]

            decoder_logits = decoder_logits + args.kg_bias_weight * kg_bias_padded

        # fusion
        inter_weight = args.fusion_bias_weight
        kg_bias_at_pos0 = kg_bias_padded[:, 0, :]  
        dir_bias_at_pos0 = dir_bias_padded         
        interaction = kg_bias_at_pos0 * dir_bias_at_pos0
        decoder_logits[:, 0, :] += inter_weight * interaction

        loss = self.cross_entropy(
            decoder_logits.reshape(-1, decoder_logits.size(-1)),
            labels.reshape(-1),
        )

        # KG graph alignment loss 
        if self.kg_compgcn is not None and entity_emb is not None:
            num_edges = self.kg_edge_index.size(1)
            n = min(args.kg_loss_samples, num_edges)  # 随机采样
            perm = torch.randperm(num_edges, device=self.device)[:n]
            src, dst = self.kg_edge_index[:, perm]

            src_vec = entity_emb[src]   # [n, kg_dim]
            dst_vec = entity_emb[dst]   # [n, kg_dim]
            kg_loss = F.mse_loss(src_vec, dst_vec)

            loss = loss + args.kg_loss_weight * kg_loss

        # Gap empty classifier
        ctx_len = context_embeds.size(1)
        hidden = outputs["hidden"]  # [B, L_total, D]
        ctx_hidden = hidden[:, :ctx_len, :]  # [B, ctx_len, D]
        ctx_pooled = masked_mean_pooling(ctx_hidden, context_mask)  # [B, D]
        gap_cls_logits = self.gap_classifier(ctx_pooled)  # [B, 2]
 
        gap_is_empty = torch.tensor([
            1 if len(nodes) == 0 else 0 for nodes in batch_target_nodes
        ], device=self.device, dtype=torch.long)

        gap_cls_loss = F.cross_entropy(gap_cls_logits, gap_is_empty)

        # Add to total loss
        loss = loss + args.gap_cls_weight * gap_cls_loss

        # Direction loss
        dir_loss_weight = args.dir_loss_weight
        loss = loss + dir_loss_weight * dir_loss

        pred_ids = decoder_logits.argmax(dim=-1)  

        return {
            "loss": loss,
            "logits": decoder_logits,
            "labels": labels,
            "pred_ids": pred_ids,
            "sample_ids": batch_sample_id,
            "gap_cls_logits": gap_cls_logits,
            "gap_cls_labels": gap_is_empty,
            "gap_cls_loss": gap_cls_loss,
            "kg_loss": kg_loss,
            "dir_loss": dir_loss,
            "dir_prob": dir_prob_all,
        }

        
