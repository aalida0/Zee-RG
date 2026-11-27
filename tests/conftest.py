"""The archived campaign (prod1), its starting points and the benchmark records BM1/BM2/BP1/BP3 were produced with the printed-HG charged-charged term of T (hg_factor=True),
which the V13 referee report showed to be wrong (general Zee result: -2 w1 w2 F).  Tests that reproduce those archived objects therefore run the whole pytest process in the
named historical-reproduction mode below.  The corrected default is tested separately in subprocesses (tests/test_stu.py, tests/test_input_sensitivity.py) that unset this variable."""
import os
os.environ.setdefault('ZEE_STU_LEGACY_HG', '1')
