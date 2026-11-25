#!/bin/bash
# One-time setup on login.af.uchicago.edu. Usage: bash setup.sh   (zee_small.tar.gz must be in $HOME)
set -e
Z=$HOME/zee_env; mkdir -p $Z/out $Z/logs; cd $Z
# no shared mutable checkout: per-run frozen checkouts are made by make_run.sh RUN_ID COMMIT
tar xzf ~/zee_small.tar.gz -C $Z
cd $Z/zee_data
[ -d hbdataset ] || git clone https://gitlab.com/higgsbounds/hbdataset.git
[ -d hsdataset ] || git clone https://gitlab.com/higgsbounds/hsdataset.git
git -C hbdataset checkout -q 3539f9adf3724f8752701208bf059d1658824cd2; git -C hsdataset checkout -q 3c6c2538d9ba0549ea0559cf9b38c18eb80c3af1
cd $Z
# HiggsTools is compiled against Python: we need an interpreter WITH its headers (Python.h). System python3.x on the AF
# lacks python3.x-devel, so if none has headers we fetch a self-contained Python (python-build-standalone via uv) into $Z.
hdr='import sys,sysconfig,os; sys.exit(sys.version_info < (3, 9) or not os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h")))'
PY=""
for p in python3.12 python3.11 python3.10 python3.9; do
  if command -v $p >/dev/null 2>&1 && $p -c "$hdr"; then PY=$(command -v $p); break; fi
done
if [ -z "$PY" ]; then
  echo "no system Python with headers: installing a standalone Python 3.12 under $Z/pythons"
  B=$(command -v python3.12 || command -v python3)
  [ -d uvenv ] || $B -m venv uvenv
  uvenv/bin/pip install -q uv
  UV_PYTHON_INSTALL_DIR=$Z/pythons uvenv/bin/uv python install 3.12
  PY=$(UV_PYTHON_INSTALL_DIR=$Z/pythons uvenv/bin/uv python find 3.12 --managed-python)
  $PY -c "$hdr" || { echo "standalone Python has no headers either"; exit 1; }
fi
echo "using $PY ($($PY --version))"
# a venv built from a header-less Python must be rebuilt
[ -d venv ] && ! venv/bin/python -c "$hdr" 2>/dev/null && rm -rf venv higgstools/_skbuild
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
