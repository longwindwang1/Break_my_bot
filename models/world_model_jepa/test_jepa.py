"""Latent world model tests: window gathering, causality, streaming vs batched prediction, EMA, scores, regularizers.

    ~/miniforge3/envs/env_isaaclab/bin/python -m pytest models/world_model_jepa -v
"""

import math

import numpy as np
import pytest
import torch

from models.world_model.dataset import WMData
from models.world_model.evaluate import Scorer, whitening
from models.world_model.surprisal import weights
from models.world_model.test_dataset import N, T, episode_steps, write_split

from .data import gather
from .model import LatentWM, effective_rank, vicreg_terms
from .predict import LATENT_COMBOS, predict_latent


def split_with_resets(tmp_path, name="s"):
    term = np.zeros((T, N), bool)
    trunc = np.zeros((T, N), bool)
    term[4, 0] = True  # env 0: new episode from frame 5
    trunc[8, 1] = True  # env 1: new episode from frame 9
    write_split(tmp_path / name, term, trunc, episode_step=episode_steps(term | trunc))
    return WMData(tmp_path / name, device="cpu")


def tiny_model(d: WMData, L=3, dim=8, M=3, seed=0) -> LatentWM:
    torch.manual_seed(seed)
    lay = d.layout
    m = LatentWM(state_dim=lay.state_dim, act_dim=lay.frame_dim - lay.state_dim, scan_dim=d.scan.shape[-1],
                 n_policy=len(lay.policies), H=lay.history_length, L=L, d=dim, M=M, enc_hidden=(16,), hidden=(16,))
    m.s_std.fill_(100.0)  # synthetic frames are 100 n + t
    with torch.no_grad():  # non-trivial predictor (the mean head starts at zero)
        m.head_mean.weight.normal_(0, 0.1)
        m.var_floor.fill_(1e-3)
    return m.eval()


def frame_id(states: torch.Tensor) -> torch.Tensor:
    """Synthetic frames hold 100 n + t in every dim."""
    return states[..., 0].round().long()


def test_gather_windows_fill_and_offsets(tmp_path):
    d = split_with_resets(tmp_path)
    H, L, K = d.layout.history_length, 3, 2
    n = torch.tensor([0, 0, 1, 2])
    t = torch.tensor([6, 2, 10, 9])
    g = gather(d, n, t, H, L, K)
    assert g["wins"].shape == (4, H + K, L, d.layout.state_dim)
    for b in range(4):
        start = int(d.ep_start[n[b], t[b]])
        for i in range(H + K):
            o = i - (H - 1)
            want = [100 * int(n[b]) + min(max(int(t[b]) + o - L + 1 + j, start), T - 1) for j in range(L)]  # clamped at the end
            assert frame_id(g["wins"][b, i]).tolist() == want, (b, i)
        # the context's prev-action frames are exactly WMData.history_frames (the ensemble's history)
        hist = d.history_frames(n[b : b + 1], t[b : b + 1])[0]
        assert torch.equal(g["prev_act"][b, :H], hist[:, d.layout.state_dim :])
    # env 0, t = 6: the episode started at frame 5, so offsets before it repeat frame 5
    assert frame_id(g["wins"][0, 0]).tolist() == [5, 5, 5]
    # validity follows train_ok along the rollout: env 1 times out in step 8 -> from t = 7 the 2-step prediction is invalid
    g2 = gather(d, torch.tensor([1, 1]), torch.tensor([6, 7]), H, L, 2)
    assert g2["valid"].tolist() == [[True, True], [True, False]]


def test_streaming_prediction_equals_batched(tmp_path):
    d = split_with_resets(tmp_path)
    m = tiny_model(d)
    P = predict_latent(m, d, keep_latents=True, keep_pred=True)
    H = m.H
    n, t = torch.arange(N).repeat_interleave(T - 1), torch.arange(T - 1).repeat(N)
    g = gather(d, n, t, H, m.L, 1)
    with torch.no_grad():
        z_ctx = m.encode(g["wins"][:, :H])
        x = m.predictor_input(z_ctx, g["prev_act"][:, :H], g["act"][:, 0], g["scan"])
        mean, lv, _ = m.step(x.unsqueeze(0).expand(m.M, -1, -1), z_ctx[:, -1].unsqueeze(0).expand(m.M, -1, -1))
        mu, var = LatentWM.moments(mean, lv)
        z_tg = m.encode(g["wins"][:, H - 1], target=True)
    torch.testing.assert_close(P["pred"][0][n, t], mu, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(P["pred"][1][n, t], var, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(P["latents"][n, t], z_tg, rtol=1e-5, atol=1e-5)


def test_prediction_is_causal(tmp_path):
    d = split_with_resets(tmp_path, "a")
    t0 = 6
    proprio = np.load(tmp_path / "a" / "proprio.npy")
    proprio[t0 + 1 :] += 7.0  # change the future only
    (tmp_path / "b").mkdir()
    for f in (tmp_path / "a").iterdir():
        (tmp_path / "b" / f.name).write_bytes(f.read_bytes())
    np.save(tmp_path / "b" / "proprio.npy", proprio)
    d2 = WMData(tmp_path / "b", device="cpu")
    m = tiny_model(d)
    P1, P2 = (predict_latent(m, x, keep_latents=True, keep_pred=True) for x in (d, d2))
    torch.testing.assert_close(P1["pred"][0][:, : t0 + 1], P2["pred"][0][:, : t0 + 1])
    torch.testing.assert_close(P1["latents"][:, : t0 + 1], P2["latents"][:, : t0 + 1])
    assert not torch.allclose(P1["latents"][:, t0 + 1 :], P2["latents"][:, t0 + 1 :])


def test_scores_by_hand(tmp_path):
    d = split_with_resets(tmp_path)
    m = tiny_model(d)
    with torch.no_grad():
        m.whiten.copy_(torch.linalg.qr(torch.randn(m.d, m.d))[0])
    P = predict_latent(m, d, keep_latents=True, keep_pred=True)
    mu, var = P["pred"]
    z = P["latents"]
    u = 5
    v = var[:, u - 1] + m.var_floor
    r = (z[:, u] - mu[:, u - 1]) / v.sqrt()
    want = {("norm", "all", 9.0): (0.5 * (r.pow(2).clamp(max=9.0) - 1)).mean(-1),
            ("nll", "all", 25.0): (0.5 * (r.pow(2).clamp(max=25.0) + torch.log(2 * math.pi * v))).mean(-1),
            ("norm", "maha_all", 25.0): (0.5 * ((r @ m.whiten.T).pow(2).clamp(max=25.0) - 1)).mean(-1)}
    for k, w in want.items():
        torch.testing.assert_close(P["scores"][k][:, u], w, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(P["spike"][:, u], r.abs().amax(-1), rtol=1e-4, atol=1e-5)
    assert set(P["scores"]) == set(LATENT_COMBOS)


def test_ema_update():
    m = LatentWM(state_dim=4, act_dim=2, scan_dim=3, n_policy=2, H=2, L=1, d=4, M=2, enc_hidden=(8,), hidden=(8,))
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(1.0)
    old = [p.clone() for p in m.target.parameters()]
    m.ema_update(0.9)
    for po, pt, p0 in zip(m.encoder.parameters(), m.target.parameters(), old):
        torch.testing.assert_close(pt, 0.9 * p0 + 0.1 * po)
    assert not any(p.requires_grad for p in m.target.parameters())


def test_vicreg_and_rank():
    torch.manual_seed(0)
    z = torch.randn(4096, 16)
    var, cov = vicreg_terms(z)
    assert var < 0.05 and cov < 0.05
    assert effective_rank(z) > 14
    collapsed = torch.randn(4096, 1) * torch.ones(1, 16) * 1e-3
    var_c, _ = vicreg_terms(collapsed)
    assert var_c > 0.9
    assert effective_rank(collapsed) < 1.5


def test_ensemble_scorer_interface(tmp_path):
    """Scorer.score with a W5 weighting is the old raw() path; maha whitening decorrelates its own residuals."""
    d = split_with_resets(tmp_path)
    S = d.layout.state_dim
    torch.manual_seed(0)
    P = {"delta_mean": torch.randn(N, T, S + 4) * 0.1, "delta_var": torch.rand(N, T, S + 4) + 0.5}
    sc = Scorer(d, P, torch.full((S,), 1e-3), warmup=0)
    for wname in ("all", "joints"):
        w = torch.as_tensor(weights(wname), dtype=torch.float32)
        torch.testing.assert_close(sc.score(wname, 25.0, "norm"), sc.raw(w, 25.0, "norm"))
    sc.z = torch.randn(N, 4000, S) @ torch.randn(S, S)  # correlated residuals
    wh = whitening(sc, torch.ones(N, 4000, dtype=torch.bool))
    dims, W = wh["maha_all"]
    u = sc.z.reshape(-1, S) @ W.T
    torch.testing.assert_close(torch.cov(u.T.double()).float(), torch.eye(S), atol=1e-3, rtol=0)
    assert len(wh["maha_joints"][0]) == int((weights("joints") > 0).sum())
    with pytest.raises(ValueError):
        sc.whiten = wh
        sc.score("maha_all", 25.0, "nll")
