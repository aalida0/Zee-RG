import os, sys, json, numpy as np, pytest
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R); sys.path.insert(0, R + '/gate1'); sys.path.insert(0, R + '/gate2'); os.environ.setdefault('ZEE_INPUTS', 'modern')
from zee import chargedbr as CB
B = json.load(open(R + '/results/benchmarks_input_superseded.json'))   # old muon-dominated light-scalar points (kept as regression)
NEW = json.load(open(R + '/results/benchmarks_input.json'))

def test_brs_sum_to_one_and_lepton_dominance():
    r = CB.widths_and_brs(B['BM1']['x'], B['BM1']['eta'])
    for br in r['BR']: assert sum(br.values()) == pytest.approx(1.0)
    assert r['BR'][0]['munu'] > 0.9 and r['width'][1] > 1.0     # light state mu nu, heavy state broad

def test_matches_higgstools():
    """Total widths AND branching ratios of both paths agree, including the extra h2+ -> h1+ h width (added through HiggsTools' scalar-pair channel;
    setTotalWidth alone would not rescale the BRs)."""
    pytest.importorskip('Higgs'); import hcoll
    for D in (B, NEW):
        for k in D:
            r = CB.widths_and_brs(D[k]['x'], D[k]['eta']); col, _ = hcoll.run(np.array(D[k]['x']), np.array(D[k]['eta']))
            for i, n in enumerate(('Hp1', 'Hp2')):
                assert r['width'][i] == pytest.approx(col[n + '_width'], rel=1e-6)
                for ch in ('enu', 'munu', 'taunu'):
                    assert r['BR'][i][ch] == pytest.approx(col[n + '_BR' + ch], rel=1e-6, abs=1e-6)   # HiggsTools returns 0 for BRs below ~1e-6


def test_new_benchmarks_decay_to_tb():
    for k in NEW:
        r = CB.widths_and_brs(NEW[k]['x'], NEW[k]['eta'])
        assert r['BR'][0]['tb'] > 0.9 and r['m'][0] > 450


def test_neutral_and_charged_extra_widths_rescale_brs():
    """H -> h h (and h2+ -> h1+ h) go in as scalar-pair channels: the BRs of the other channels of H must shrink by T/(T+Gamma)
    (BM2: Gamma(H->hh) = 0.36 GeV of 15 GeV), and the new channel carries Gamma/T."""
    pytest.importorskip('Higgs'); import hcoll; HP = hcoll.HP; WD = hcoll.WD
    cap = {}; orig = HP.Predictions
    class P(orig):
        def __init__(s, *a, **k): super().__init__(*a, **k); cap['p'] = s
    HP.Predictions = P; g0 = WD.gamma_Hhh
    try:
        x, eta = np.array(NEW['BM2']['x']), np.array(NEW['BM2']['eta'])
        WD.gamma_Hhh = lambda *a, **k: 0.0; hcoll.run(x, eta); H0 = cap['p'].particle('H'); T0, b0 = H0.totalWidth(), H0.br(HP.Decay.tt)
        WD.gamma_Hhh = g0; out, _ = hcoll.run(x, eta); H1 = cap['p'].particle('H')
        assert out['Gamma_H_hh'] > 0.1 and H1.totalWidth() == pytest.approx(T0 + out['Gamma_H_hh'], rel=1e-9)
        assert H1.br(HP.Decay.tt)/b0 == pytest.approx(T0/(T0 + out['Gamma_H_hh']), rel=1e-6)
        assert H1.br('h', 'h') == pytest.approx(out['Gamma_H_hh']/H1.totalWidth(), rel=1e-6)
    finally: HP.Predictions = orig; WD.gamma_Hhh = g0
