"""Strategy B model: multi-label (per-category) token classification (spec section 4.2).

Instead of a single 83-way softmax head (Strategy A), this uses one linear head of width
``num_categories`` whose outputs are treated as INDEPENDENT per-category logits (sigmoid,
not softmax). A token may therefore belong to several categories at once, which preserves
the overlapping clause spans that Strategy A's single-label collapse discards.

Loss: masked BCEWithLogitsLoss. Label rows of -100 (special tokens, continuation
subwords) are excluded from the loss, matching the tokenize_align Strategy B encoding.

torch/transformers are imported lazily inside build_multilabel_model so this module can
be imported without the heavy stack (keeps the package import-safe for local tests).
"""

from __future__ import annotations

from . import config


def build_multilabel_model(model_name: str, categories: list[str]):
    """Construct a multi-label token-classification model over ``categories``.

    Returns an instance of a BERT-based model whose forward() returns an object with
    ``.logits`` of shape (batch, seq, num_categories) and, when ``labels`` are passed,
    a ``.loss`` computed with masked BCEWithLogitsLoss.
    """
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel
    from transformers.modeling_outputs import TokenClassifierOutput

    num_categories = len(categories)

    class MultiLabelTokenClassifier(nn.Module):
        """BERT encoder + a per-category sigmoid classification head."""

        def __init__(self):
            super().__init__()
            self.num_categories = num_categories
            self.categories = list(categories)
            self.config = AutoConfig.from_pretrained(
                model_name, num_labels=num_categories
            )
            # Carry id<->label maps for the per-category heads (index == category).
            self.config.id2label = {i: c for i, c in enumerate(categories)}
            self.config.label2id = {c: i for i, c in enumerate(categories)}
            self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
            dropout = getattr(self.config, "hidden_dropout_prob", 0.1)
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.config.hidden_size, num_categories)
            self.loss_fct = nn.BCEWithLogitsLoss(reduction="mean")

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    labels=None, **kwargs):
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                **{k: v for k, v in kwargs.items() if k in ("position_ids", "head_mask")},
            )
            sequence_output = self.dropout(outputs.last_hidden_state)
            logits = self.classifier(sequence_output)  # (B, T, C)

            loss = None
            if labels is not None:
                labels = labels.to(logits.dtype)
                # A position is valid if it is not the -100 ignore sentinel. The encoder
                # emits one row per token; ignored rows are all -100 (see tokenize_align).
                mask = labels[..., 0] != -100.0  # (B, T) True where token is supervised
                if mask.any():
                    active_logits = logits[mask]           # (N, C)
                    active_labels = labels[mask]           # (N, C)
                    loss = self.loss_fct(active_logits, active_labels)
                else:
                    loss = logits.sum() * 0.0  # keep graph connected if a batch is empty

            return TokenClassifierOutput(loss=loss, logits=logits,
                                         hidden_states=None, attentions=None)

        def save_pretrained(self, save_directory, **kwargs):
            import os
            os.makedirs(save_directory, exist_ok=True)
            self.config.save_pretrained(save_directory)
            torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        @classmethod
        def from_pretrained_dir(cls, load_directory):
            import os
            m = cls()
            state = torch.load(
                os.path.join(load_directory, "pytorch_model.bin"), map_location="cpu"
            )
            m.load_state_dict(state)
            return m

    return MultiLabelTokenClassifier()


def strategy_b_categories() -> list[str]:
    """The ordered category list that indexes the per-category heads."""
    return list(config.CUAD_CATEGORIES)
