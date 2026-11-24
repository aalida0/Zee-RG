#!/bin/bash
# One-time setup on login.af.uchicago.edu. Usage: bash setup.sh   (zee_small.tar.gz must be in $HOME)
set -e
Z=$HOME/zee_env; mkdir -p $Z/out $Z/logs; cd $Z
[ -d repo ] || git clone https://github.com/aalida0/Zee-RG.git repo
(cd repo && git pull)
tar xzf ~/zee_small.tar.gz -C $Z
cp -r $Z/zee_data/_nufit_cache $Z/repo/zee/_nufit_cache
cd $Z/zee_data
[ -d hbdataset ] || git clone https://gitlab.com/higgsbounds/hbdataset.git
[ -d hsdataset ] || git clone https://gitlab.com/higgsbounds/hsdataset.git
git -C hbdataset checkout -q 3539f9adf3724f8752701208bf059d1658824cd2; git -C hsdataset checkout -q 3c6c2538d9ba0549ea0559cf9b38c18eb80c3af1
cd $Z
# HiggsTools needs a recent Python: pick the newest available (system, then CVMFS LCG views)
PY=""
for p in python3.12 python3.11 python3.10 python3.9 /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/bin/python3 /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc13-opt/bin/python3; do
  if command -v $p >/dev/null 2>&1 && $p -c 'import sys; sys.exit(sys.version_info < (3, 9))'; then PY=$(command -v $p); break; fi
done
[ -n "$PY" ] || { echo "no Python >= 3.9 found: send me the output of 'python3 --version; ls /usr/bin/python3*; cat /etc/os-release | head -2'"; exit 1; }
echo "using $PY ($($PY --version))"
[ -d venv ] || $PY -m venv venv
$Z/venv/bin/pip install -q --upgrade pip
# HiggsTools is not on PyPI: build the Python module from source (needs a C++ compiler and network access; takes several minutes)
$Z/venv/bin/pip install -q numpy scipy pandas matplotlib requests cmake ninja scikit-build-core pybind11
if ! $Z/venv/bin/python -c "import Higgs" 2>/dev/null; then
  [ -d higgstools ] || git clone https://gitlab.com/higgsbounds/higgstools.git
  (cd higgstools && (git checkout -q v1.3.0 2>/dev/null || git checkout -q 1.3.0 2>/dev/null || echo "WARNING: tag 1.3.0 not found, building the default branch"))
  export CMAKE_BUILD_PARALLEL_LEVEL=4 PATH=$Z/venv/bin:$PATH
  $Z/venv/bin/pip install ./higgstools
fi
$Z/venv/bin/python -c "import Higgs, numpy, scipy; print('env ok')"
