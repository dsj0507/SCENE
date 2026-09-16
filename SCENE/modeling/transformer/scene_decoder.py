import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .scene_upsample import SCENEUpsample
from .remote_fc_branch import RemoteFineCoarseBranch


class SCENEDecoder(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_clip_guidance_dims=(256, 128),
        decoder_clip_guidance_proj_dims=(32, 16),
        use_clip_corr = True,
        use_remote_clip = True,
        use_rotate = True,
        use_efficient=True,
        nheads=4, 
        hidden_dim=128,
        feature_resolution=(24, 24),
        window_size=12,
        graph_guidance_dim=256,
        clip_dense_dim=768,
        raw_clip_logit_weight=1.0,
        logit_output_mode="fused",
        use_soft_region_context=True,
        use_graph_context_in_remote_fc=True,
        use_remote_fc_decoder=False,
        remote_fc_branch_dim=None,
        remote_fc_fine_kernels=(3, 5, 7),
        remote_fc_coarse_kernels=(7, 9, 11),
        remote_fc_fine_depth=1,
        remote_fc_coarse_depth=1,
    ) -> None:

        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_clip_corr = bool(use_clip_corr and text_guidance_dim > 0)
        self.use_remote_clip = use_remote_clip
        self.use_rotate = use_rotate
        self.raw_clip_logit_weight = float(raw_clip_logit_weight)
        self.logit_output_mode = str(logit_output_mode).lower()
        self.use_soft_region_context = bool(use_soft_region_context)
        self.use_graph_context_in_remote_fc = bool(use_graph_context_in_remote_fc)
        self.use_remote_fc_decoder = use_remote_fc_decoder
        remote_fc_branch_dim = (
            remote_fc_branch_dim if remote_fc_branch_dim is not None else decoder_dims[-1]
        )
        # The decoder now skips the old aggregation stack and feeds a fused
        # clip-dense / remote-graph feature directly into the upsample decoder.
        self.clip_dense_projection = nn.Sequential(
            nn.Conv2d(clip_dense_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        if self.use_clip_corr and clip_dense_dim != text_guidance_dim:
            raise ValueError(
                "USE_CLIP_CORR requires dense image and text embeddings in the "
                f"same space, got {clip_dense_dim} and {text_guidance_dim} channels."
            )
        self.clip_correlation_embedding = (
            nn.Sequential(
                nn.Conv2d(1, hidden_dim, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
            )
            if self.use_clip_corr
            else None
        )
        self.pre_decoder_fusion = nn.Sequential(
            nn.Conv2d(hidden_dim + remote_fc_branch_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.CLIP_decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
        ]) if decoder_clip_guidance_dims[0] > 0 else None
        
        if self.use_remote_clip:
            self.CLIP_decoder_guidance_projection_remote = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                ) for d, dp in zip(decoder_guidance_dims, decoder_clip_guidance_proj_dims)
            ]) if decoder_clip_guidance_dims[0] > 0 else None
        else:
            self.CLIP_decoder_guidance_projection_remote=None

        # RSKT_Upsample
        self.Fusiondecoder1=SCENEUpsample(hidden_dim, 
        decoder_dims[0],
        decoder_clip_guidance_proj_dims[0],
        use_remote_clip = use_remote_clip)
        self.Fusiondecoder2=SCENEUpsample(decoder_dims[0], 
        decoder_dims[1], 
        decoder_clip_guidance_proj_dims[1], 
        use_remote_clip = use_remote_clip)

        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)
        self.graph_pixel_query_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.graph_node_key_projection = nn.Sequential(
            nn.Linear(graph_guidance_dim, hidden_dim),
            nn.ReLU(),
        )
        self.graph_node_value_projection = nn.Sequential(
            nn.Linear(graph_guidance_dim, hidden_dim),
            nn.ReLU(),
        )
        self.graph_context_projection = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.graph_context_gate = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )
        if self.use_remote_fc_decoder and self.use_remote_clip:
            self.remote_fc_branch = RemoteFineCoarseBranch(
                remote_guidance_dims=decoder_guidance_dims,
                branch_dim=remote_fc_branch_dim,
                num_heads=nheads,
                use_efficient=use_efficient,
                feature_resolution=feature_resolution,
                decoder_dims=decoder_dims,
                window_size=window_size,
                graph_dense_dim=hidden_dim,
                fine_kernels=remote_fc_fine_kernels,
                coarse_kernels=remote_fc_coarse_kernels,
                fine_depth=remote_fc_fine_depth,
                coarse_depth=remote_fc_coarse_depth,
            )
        else:
            self.remote_fc_branch = None
        
    def Fusion_conv_decoer(self, x, clip_guidance, clip_guidance_remote):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.Fusiondecoder1(corr_embed, clip_guidance[0], clip_guidance_remote[0])
        corr_embed = self.Fusiondecoder2(corr_embed, clip_guidance[1], clip_guidance_remote[1])
        decoder_feature = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        decoder_logits = self.head(corr_embed) #head是一个二维卷积，作用是将特征图的通道数转换为1，得到每个像素的logit值
        decoder_logits = rearrange(decoder_logits, '(B T) () H W -> B T H W', B=B)
        return decoder_feature, decoder_logits

    def _prepare_clip_dense_features(self, img_feats):
        if isinstance(img_feats, list):
            aligned_features = [img_feats[0]]
            if len(img_feats) > 1:
                aligned_features.append(torch.rot90(img_feats[1], k=3, dims=(2, 3)))
            if len(img_feats) > 2:
                aligned_features.append(torch.rot90(img_feats[2], k=2, dims=(2, 3)))
            if len(img_feats) > 3:
                aligned_features.append(torch.rot90(img_feats[3], k=1, dims=(2, 3)))
            return torch.stack(aligned_features, dim=0).mean(dim=0)
        return img_feats

    def _prepare_clip_dense_encoding(self, img_feats):
        return self.clip_dense_projection(self._prepare_clip_dense_features(img_feats))

    def _build_raw_clip_logits(self, clip_dense_features, text_embeddings):
        if not self.use_clip_corr:
            return None
        if text_embeddings is None:
            raise ValueError("USE_CLIP_CORR requires class text embeddings.")
        if clip_dense_features.shape[1] != text_embeddings.shape[-1]:
            raise ValueError(
                "CLIP dense/text channel mismatch while building correlation: "
                f"{clip_dense_features.shape[1]} vs {text_embeddings.shape[-1]}."
            )

        dense_normalized = F.normalize(clip_dense_features.float(), dim=1)
        text_normalized = F.normalize(text_embeddings.float(), dim=-1)
        return torch.einsum("bchw,btc->bthw", dense_normalized, text_normalized)

    def _embed_clip_correlation(self, correlation):
        if correlation is None:
            return None
        batch_size, num_classes, height, width = correlation.shape
        correlation = rearrange(correlation, "B T H W -> (B T) () H W")
        correlation = correlation.to(dtype=self.clip_correlation_embedding[0].weight.dtype)
        correlation = self.clip_correlation_embedding(correlation)
        return rearrange(
            correlation,
            "(B T) C H W -> B C T H W",
            B=batch_size,
            T=num_classes,
            H=height,
            W=width,
        )

    def _build_clip_correlation_feature(self, clip_dense_features, text_embeddings):
        raw_clip_logits = self._build_raw_clip_logits(
            clip_dense_features,
            text_embeddings,
        )
        return self._embed_clip_correlation(raw_clip_logits)

    def _build_predecoder_feature(
        self,
        clip_dense_encoding,
        text_guidance,
        remote_spatial_feature,
        clip_correlation_feature=None,
    ):
        if clip_correlation_feature is not None:
            clip_class_feature = clip_dense_encoding.unsqueeze(2) + clip_correlation_feature
        elif text_guidance is None:
            return clip_dense_encoding.unsqueeze(2)
        else:
            num_classes = text_guidance.shape[1]
            clip_class_feature = clip_dense_encoding.unsqueeze(2).expand(-1, -1, num_classes, -1, -1)
            text_bias = text_guidance.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            clip_class_feature = clip_class_feature + text_bias

        if remote_spatial_feature is None:
            return clip_class_feature

        batch_size, _, num_classes, height, width = clip_class_feature.shape
        remote_shared_feature = remote_spatial_feature.unsqueeze(2).expand(
            -1, -1, num_classes, -1, -1
        )
        fused_feature = torch.cat([clip_class_feature, remote_shared_feature], dim=1)
        fused_feature = rearrange(fused_feature, "B C T H W -> (B T) C H W")
        fused_feature = self.pre_decoder_fusion(fused_feature)
        return rearrange(
            fused_feature,
            "(B T) C H W -> B C T H W",
            B=batch_size,
            T=num_classes,
            H=height,
            W=width,
        )

    def _build_soft_region_context(
        self,
        clip_dense_encoding,
        graph_state,
    ):
        if graph_state is None:
            return None

        node_features = graph_state.get("node_embeddings_refined")
        graph_outputs = graph_state.get("graph_outputs")
        proposal_items = graph_state.get("proposal_items")
        if (
            node_features is None
            or graph_outputs is None
            or proposal_items is None
            or node_features.numel() == 0
        ):
            return None

        graph_ptr = graph_outputs.get("graph_ptr")
        if graph_ptr is None or len(proposal_items) == 0:
            return None

        batch_size, hidden_dim, height, width = clip_dense_encoding.shape
        pixel_queries = self.graph_pixel_query_projection(clip_dense_encoding)
        context_maps = []
        coverage_maps = []
        scale = float(hidden_dim) ** -0.5

        for graph_idx, proposal_item in enumerate(proposal_items):
            start = int(graph_ptr[graph_idx].item())
            end = int(graph_ptr[graph_idx + 1].item())
            if end <= start:
                context_maps.append(torch.zeros((hidden_dim, height, width), device=clip_dense_encoding.device))
                coverage_maps.append(torch.zeros((1, height, width), device=clip_dense_encoding.device))
                continue

            query = pixel_queries[graph_idx].flatten(1).transpose(0, 1)
            node_slice = node_features[start:end].float()
            node_keys = self.graph_node_key_projection(node_slice)
            node_values = self.graph_node_value_projection(node_slice)
            affinity_logits = (query @ node_keys.transpose(0, 1)) * scale

            mask_source = proposal_item.get("mask_logits")
            if mask_source is not None and mask_source.numel() > 0:
                if mask_source.ndim == 4:
                    mask_source = mask_source.squeeze(1)
                spatial_prior = F.interpolate(
                    mask_source.float().unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).sigmoid()
            else:
                mask_source = proposal_item["masks"]
                if mask_source.ndim == 4:
                    mask_source = mask_source.squeeze(1)
                spatial_prior = F.interpolate(
                    mask_source.float().unsqueeze(1),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

            spatial_prior = F.avg_pool2d(
                spatial_prior.unsqueeze(1),
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze(1)

            node_scores = proposal_item["scores"].float()
            node_scores = node_scores / node_scores.max().clamp_min(1e-6)
            spatial_prior = spatial_prior * node_scores.view(-1, 1, 1)
            spatial_prior_flat = spatial_prior.flatten(1).transpose(0, 1)
            valid_pixels = spatial_prior_flat.sum(dim=-1, keepdim=True) > 1e-6

            attention_logits = affinity_logits + torch.log(spatial_prior_flat.clamp_min(1e-6))
            attention_weights = attention_logits.softmax(dim=-1)
            attention_weights = attention_weights * valid_pixels.to(dtype=attention_weights.dtype)

            context = attention_weights @ node_values
            context = context.transpose(0, 1).reshape(hidden_dim, height, width)
            context = self.graph_context_projection(context.unsqueeze(0)).squeeze(0)
            coverage = spatial_prior_flat.sum(dim=-1).reshape(1, height, width).clamp(max=1.0)
            context_maps.append(context * coverage)
            coverage_maps.append(coverage)

        graph_context = torch.stack(context_maps, dim=0)
        graph_coverage = torch.stack(coverage_maps, dim=0)
        fusion_gate = self.graph_context_gate(torch.cat([clip_dense_encoding, graph_context], dim=1))

        return {
            "graph_context_feature": graph_context,
            "graph_context_gate": fusion_gate,
            "graph_context_coverage": graph_coverage,
        }

    def _contextualize_clip_dense_with_graph(
        self,
        clip_dense_encoding,
        graph_context_outputs,
    ):
        if graph_context_outputs is None:
            return clip_dense_encoding

        graph_context = graph_context_outputs["graph_context_feature"].to(
            dtype=clip_dense_encoding.dtype
        )
        graph_gate = graph_context_outputs["graph_context_gate"].to(
            dtype=clip_dense_encoding.dtype
        )
        return clip_dense_encoding + graph_gate * graph_context

    def forward(
        self,
        img_feats,
        dino_feat,
        text_feats,
        appearance_guidance,
        appearance_guidance_remote,
        dino_guidance,
        graph_state=None,
    ):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C) T is class number
            apperance_guidance: tuple of (B, C, H, W)
        """
        text_embeddings = None
        projected_text_guidance = None
        CLIP_projected_decoder_guidance = [None, None]
        CLIP_projected_decoder_guidance_remote = [None, None]

        if self.CLIP_decoder_guidance_projection is not None:
            CLIP_projected_decoder_guidance = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection, appearance_guidance[1:])]

        if self.CLIP_decoder_guidance_projection_remote is not None:
            CLIP_projected_decoder_guidance_remote = [proj(g) for proj, g in zip(self.CLIP_decoder_guidance_projection_remote, appearance_guidance_remote)]

        if text_feats is not None:
            text_embeddings = F.normalize(text_feats.mean(dim=-2).float(), dim=-1)
        if self.text_guidance_projection is not None and text_embeddings is not None:
            projected_text_guidance = self.text_guidance_projection(text_embeddings)

        clip_dense_features = self._prepare_clip_dense_features(img_feats)
        clip_dense_encoding = self.clip_dense_projection(clip_dense_features)
        raw_clip_logits = self._build_raw_clip_logits(
            clip_dense_features,
            text_embeddings,
        )
        clip_correlation_feature = self._embed_clip_correlation(raw_clip_logits)
        graph_context_outputs = None
        if self.use_soft_region_context:
            graph_context_outputs = self._build_soft_region_context(clip_dense_encoding, graph_state)
        contextualized_clip_dense_encoding = self._contextualize_clip_dense_with_graph(
            clip_dense_encoding,
            graph_context_outputs,
        )
        remote_fc_feature = None
        remote_graph_enhanced = None
        remote_fine_routing_weights = None
        remote_coarse_routing_weights = None
        if self.remote_fc_branch is not None:
            remote_graph_context_outputs = (
                graph_context_outputs if self.use_graph_context_in_remote_fc else None
            )
            remote_outputs = self.remote_fc_branch(
                appearance_guidance_remote=appearance_guidance_remote,
                graph_dense_feature=None if remote_graph_context_outputs is None else remote_graph_context_outputs["graph_context_feature"],
                graph_context_coverage=None if remote_graph_context_outputs is None else remote_graph_context_outputs["graph_context_coverage"],
                target_hw=clip_dense_encoding.shape[-2:],
            )
            remote_fc_feature = remote_outputs["remote_fc_feature"]
            remote_graph_enhanced = remote_outputs["remote_graph_enhanced"]
            remote_fine_routing_weights = remote_outputs["fine_routing_weights"]
            remote_coarse_routing_weights = remote_outputs["coarse_routing_weights"]
        # Main CLIP owns the class axis; Remote/Graph guidance stays class-agnostic.
        predecoder_feature = self._build_predecoder_feature(
            contextualized_clip_dense_encoding,
            projected_text_guidance,
            remote_graph_enhanced,
            clip_correlation_feature,
        )
        decoder_feature, decoder_logits = self.Fusion_conv_decoer(
            predecoder_feature,
            CLIP_projected_decoder_guidance,
            CLIP_projected_decoder_guidance_remote,
        )

        fused_logits = decoder_logits
        if raw_clip_logits is not None and self.raw_clip_logit_weight != 0.0:
            raw_clip_logits_resized = F.interpolate(
                raw_clip_logits,
                size=decoder_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused_logits = (
                decoder_logits
                + self.raw_clip_logit_weight
                * raw_clip_logits_resized.to(dtype=decoder_logits.dtype)
            )

        if self.logit_output_mode == "fused":
            output_logits = fused_logits
        elif self.logit_output_mode == "decoder":
            output_logits = decoder_logits
        else:
            raise ValueError(f"Unsupported LOGIT_OUTPUT_MODE: {self.logit_output_mode}")

        return {
            "logits": output_logits,
            "decoder_logits": decoder_logits,
            "decoder_feature": decoder_feature,
            "raw_clip_logits": raw_clip_logits,
            "fused_logits": fused_logits,
            "graph_context_feature": None if graph_context_outputs is None else graph_context_outputs["graph_context_feature"],
            "graph_context_gate": None if graph_context_outputs is None else graph_context_outputs["graph_context_gate"],
            "graph_context_coverage": None if graph_context_outputs is None else graph_context_outputs["graph_context_coverage"],
            "remote_fc_feature": remote_fc_feature,
            "remote_graph_enhanced": remote_graph_enhanced,
            "remote_fine_routing_weights": remote_fine_routing_weights,
            "remote_coarse_routing_weights": remote_coarse_routing_weights,
        }
