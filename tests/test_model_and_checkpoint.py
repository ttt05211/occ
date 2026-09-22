from copy import deepcopy
from pathlib import Path

import torch

from causal_se2_occ.checkpoint import (
    load_model_checkpoint,
    migrate_legacy_checkpoint,
)
from causal_se2_occ.data.features import FEATURE_DIM
from causal_se2_occ.models.stwm import (
    FutureQueryBlock,
    ModelConfig,
    SourceCenteredSE2Predictor,
)
from causal_se2_occ.protocol import CHECKPOINT_PROTOCOL


def test_frozen_default_parameter_count_and_init():
    model = SourceCenteredSE2Predictor()
    assert FEATURE_DIM == 46
    assert sum(p.numel() for p in model.parameters()) == 2_073_220
    assert torch.count_nonzero(model.residual_head.weight) == 0
    assert torch.count_nonzero(model.yaw_head.weight) == 0


def test_forward_shapes():
    config = ModelConfig(
        d_model=16,
        semantic_dim=8,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=4,
    )
    model = SourceCenteredSE2Predictor(config)
    batch = 2
    output = model(
        torch.zeros(batch, FEATURE_DIM),
        torch.full((batch, 6, 4, 4), 17, dtype=torch.uint8),
        torch.zeros(batch, 6, 2),
        torch.zeros(batch, 6, 5),
        torch.zeros(batch, 6, 4, 4, dtype=torch.uint8),
    )
    assert output["residual_xy_m"].shape == (batch, 6, 2)
    assert torch.equal(
        output["residual_xy_m"],
        torch.zeros_like(output["residual_xy_m"]),
    )


def test_identity_migration(tmp_path: Path):
    config = ModelConfig(
        d_model=16,
        semantic_dim=8,
        heads=4,
        blocks=1,
        decoder_blocks=1,
        tube_hw=4,
    )
    model = SourceCenteredSE2Predictor(config)
    source = tmp_path / "a.pt"
    target = tmp_path / "b.pt"
    torch.save(
        {
            "protocol": CHECKPOINT_PROTOCOL,
            "arm": "Y",
            "model_config": config.__dict__,
            "state_dict": model.state_dict(),
            "epoch": 14,
            "global_step": 18410,
        },
        source,
    )
    migrate_legacy_checkpoint(source, target)
    _, restored = load_model_checkpoint(target)
    assert list(model.state_dict()) == list(restored.state_dict())
    assert all(
        torch.equal(value, restored.state_dict()[key])
        for key, value in model.state_dict().items()
    )


def _legacy_future_query_forward(block, query, context):
    z = block.self_norm(query)
    y, _ = block.self_attn(z, z, z, need_weights=False)
    query = query + y
    y, _ = block.cross_attn(
        block.cross_q_norm(query),
        block.cross_ctx_norm(context),
        block.cross_ctx_norm(context),
        need_weights=False,
    )
    query = query + y
    return query + block.ffn(block.ffn_norm(query))


def test_future_query_block_preserves_frozen_duplicate_context_norm_graph():
    torch.manual_seed(19)
    current = FutureQueryBlock(32, 4)
    reference = deepcopy(current)

    q_current = torch.randn(3, 6, 32, requires_grad=True)
    context_current = torch.randn(3, 17, 32, requires_grad=True)
    q_reference = q_current.detach().clone().requires_grad_(True)
    context_reference = context_current.detach().clone().requires_grad_(True)
    weight = torch.randn(3, 6, 32)

    out_current = current(q_current, context_current)
    out_reference = _legacy_future_query_forward(
        reference,
        q_reference,
        context_reference,
    )
    assert torch.equal(out_current, out_reference)

    (out_current * weight).sum().backward()
    (out_reference * weight).sum().backward()

    for (name_a, param_a), (name_b, param_b) in zip(
        current.named_parameters(),
        reference.named_parameters(),
    ):
        assert name_a == name_b
        assert torch.equal(param_a.grad, param_b.grad), name_a
