# Formula notes from full-text reading (for the referee-v5 repair)

Each item: source → what we take → which audit item it fixes.

## C01 universality (HG 1701.05345 Eqs. 3.28–3.33; Crivellin et al. 2012.09845 Eq. 2)
- 1/M̃2² = sin²φ/m1² + cos²φ/m2²  (h1+ = c H+ − s h+ convention of HG Eq. 2.8)
- (G_Zee/G_SM)² = 1 + √2|f_eμ|²/(G_F M̃2²); g-ratios² = 1 + √2|f|²/(G_F M̃2²) with f_eτ (τ/μ), f_μτ (τ/e), f_μτ−f_eτ (μ/e).
- Code had eps = √2 M̃2²/G_F (inverted). Fix: eps = √2*Kf/GF, Kf = s²/m1² + c²/m2².
- Cross-check: Crivellin δ = |λ|² m_W²/(g² m²) with λ = 2f ⇒ (1+δ)² ≈ 1 + √2|f|²/(G_F m²). Same.
- Relaxed texture (f_eμ ≠ 0): divide every ratio by (1+eps|f_eμ|²) for μ→e normalisation, and add the CKM-unitarity shift V_us^β ≃ V_us[1 + (V_ud/V_us)² δ(μ→eνν)].
- Data (Crivellin/HFLAV): A(τ→μ)/A(μ→e) = 1.0029(14), A(τ→μ)/A(τ→e) = 1.0018(14), A(τ→e)/A(μ→e) = 1.0010(14); our UNIV table should be re-checked against HFLAV 2023 and given correlations.
- Babu et al. 1907.09498 §4.9: doublet Yukawas also give a non-interfering scalar g^s_RR = −Y_αe Y*_βτ sin²φ/(2√2 G_F m²); O(m⁻⁴), negligible for our Y2 ≲ 0.03, mention only.

## C04 / C07 charged and pseudoscalar couplings (Aristizabal Sierra–Restrepo hep-ph/0604012)
- −L ⊃ ν̄_L O e_R (c h1+ − s h2+) + ν^T C (2f) e_L (s h1+ + c h2+), with O = −√2 tβ M_ℓ/v + Π2/cβ.
- Γ(h_k+ → ℓ_j ν) = M_k/(16π) Σ_i [|O_ij|² W1k² + |2f_ij|² W2k²]: column norms of O (ν index first).
- Our D must therefore be the matrix with ν row index and e_R column index, i.e. Y2†/cβ − √2 tβ m_E/v in HG notation. hcoll/postfit use Y2/cβ (transpose): fix.
- They find μν can dominate τν for h1+; consistent with the corrected widths.
- Omura–Senaha–Tobe 1511.08880 Eq. 7: y_A = +iρ/√2 for leptons: A couplings are imaginary parts of ρ, so A→τμ uses Im, not Re.

## C03 one-loop dipoles with both chirality flips (Omura et al. 1511.08880 Eqs. 16–18, 30–32; Lavoura hep-ph/0302221 §4)
- T = e ε* ū_j m_i iσq (A_L P_L + A_R P_R) u_i; Br/Br(i→jνν) = 48π³α(|A_L|²+|A_R|²)/G_F².
- Neutral φ = h, H, A, sum over internal k:
  A_L^φ = 1/(16π² m_φ²) Σ_k y*_{φ k j}[(m_k/m_i) y*_{φ i k}(ln(m_φ²/m_k²) − 3/2) + y_{φ k i}/6]
  A_R^φ = 1/(16π² m_φ²) Σ_k y_{φ j k}[(m_k/m_i) y_{φ k i}(ln(m_φ²/m_k²) − 3/2) + y*_{φ i k}/6]
  (the 1/6 term is the external-flip piece; HG Eq. 3.13 has the same structure).
- Charged doublet: A_L^{H−} = −(ρ†ρ)_{ji}/(192π² m²), A_R = 0; for the Zee model weight by c² (h1) and s² (h2), plus f-loop (HG Eq. 3.19).
- Check: diagonal limit gives δa_μ = m_μ²(A_L + A_R) and reproduces their Eq. 15.
- Lavoura closed forms for general masses (t = m_F²/m_B²) available if m_k is not ≪ m_φ.

## C02 two-loop Barr–Zee (Omura et al. 1511.08880 Eq. 20, from Chang–Hou–Keung hep-ph/9302267)
- Fermion loops (t, b, τ) through φγγ and φZγ, with F_H, F_A, F̃_H, F̃_A.
- W loop: (α/16π³) g_φWW y*_{φij}/(m_i v)[3F_H + 23/4 F_A + 3/4 G + m_φ²/(2m_W²)(F_H − F_A) + Zγ part], g_hWW = s_βα, g_HWW = c_βα.
  Vanishes in exact alignment only for H; for h it is ∝ s_βα y_{hij} ∝ c_βα ρ: not negligible off alignment.
- φZγ adds ~10% (their statement).
- Charged-scalar loop (h1+, h2+): Ilisie 1502.04199 Eq. 20 for g−2: δa = α m_μ²/(8π³ m_φ²) Re(y_l) λ_{φH+H−} F⁽²⁾(m±²/m_φ²), F⁽²⁾(ω) = ½∫dx x(x−1)/(ω − x(1−x)) ln(ω/(x(1−x))). Map to LFV by m_μ y_l/v → off-diagonal y_{φij}; we already have g_{h h_i+ h_i−} for R_γγ.
- Heeck–Thapa 2303.13383 drop W and charged-scalar BZ (alignment limit); we cannot, since c_βα ≠ 0 on our points.
- Sign: 1502.07824 fn. says its one-loop/two-loop relative sign differs from Davidson–Grenier and Harnik–Kopp–Zupan. Fix the sign by the diagonal-limit test against the known g−2 results (scalar top-loop BZ negative, pseudoscalar positive for same-sign couplings).

## Vacuum / unitarity / BFB
- Barroso–Ferreira hep-ph/0507128: in the Zee model, unlike the 2HDM, CB minima can coexist with and lie below the normal minimum; the effect needs the trilinear μ (none at μ = 0). Typical region |μ| > 500 GeV, tβ ≲ 10. No closed mass criterion: justifies the numerical global-minimum check (vac.global_ok).
- Heeck–Thapa use a heuristic |μ| ≲ 4.1 m_heavier±; Florentino–Romão–Silva 2106.08332 use gradient descent from random starts (same as ours) and 19 unitarity matrices with max eig < 8π.
- Lopes–Milagre–Silva 2510.02434 §V.H (2HDM + charged + neutral singlet): closed eigenvalues to cross-check our 55×55 matrix: |2α_{11,11}| ≤ 8π; ½|δ11+δ22 ± √((δ11−δ22)² + 4|δ12|²)| ≤ 8π (drop the χ pieces). Use as unit tests (C10).
- Schmidt–Vandeleur 2512.17235: BFB only necessary 2HDM conditions plus λ_h ≥ 0, λ8 = λ9 = λ10 = 0, one-loop Matchete/RGBeta; no cutoff/instability scale. Our exact singlet elimination and cutoff are beyond this.

## RG scope
- Lepton Yukawas are omitted from our quartic β functions. On BM1–3: max|Y2| ≈ 0.028, |f| < 1e-3, so their contributions (∝ Y⁴/16π², λY²/16π²) are < 1e-6 per e-fold: negligible; state this with numbers.

## Collider (hcoll, postfit)
- b→sγ: sum over h1+, h2+ with weights c², s² (Florentino et al. Eq. for C7): our bsg does this.
- h2+ → h1+ Z with coupling (g/2c_W) sin φ cos φ: present in hcoll; consistent with 2106.08332.
- Babu et al. 1907.09498 §4.7: LEP limit ~100 GeV, LHC slepton recasts evaded for BR(τν) ≳ 0.4.
