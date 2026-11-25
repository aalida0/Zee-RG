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
    """Total widths AND branching ratios of both paths agree, including the extra h2+ -> h1+ h width (HiggsTools' setTotalWidth does not
    rescale the BRs: gate2/hcoll._add_width does)."""
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
