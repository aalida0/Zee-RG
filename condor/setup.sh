#!/bin/bash
# One-time setup on login.af.uchicago.edu. Usage: bash setup.sh   (zee_small.tar.gz must be in $HOME)
set -e
Z=$HOME/zee; mkdir -p $Z/out $Z/logs; cd $Z
[ -d repo ] || git clone https://github.com/aalida0/Zee-RG.git repo
(cd repo && git pull)
tar xzf ~/zee_small.tar.gz -C $Z
cp -r $Z/zee_data/_nufit_cache $Z/repo/zee/_nufit_cache
cd $Z/zee_data
[ -d hbdataset ] || git clone https://gitlab.com/higgsbounds/hbdataset.git
[ -d hsdataset ] || git clone https://gitlab.com/higgsbounds/hsdataset.git
git -C hbdataset checkout -q 3539f9adf3724f8752701208bf059d1658824cd2; git -C hsdataset checkout -q 3c6c2538d9ba0549ea0559cf9b38c18eb80c3af1
cd $Z
[ -d venv ] || python3 -m venv venv
$Z/venv/bin/pip install -q numpy scipy HiggsTools
$Z/venv/bin/python -c "import Higgs, numpy, scipy; print('env ok')"
