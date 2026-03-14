import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import pycasso

np.random.seed(42)
n, d = 200, 50
X = np.random.randn(n, d)
b = np.array([1.5]*5 + [0]*(d-5))
Y_g = X @ b + np.random.randn(n)
Y_b = (np.random.rand(n) < 1/(1+np.exp(-X @ b))).astype(float)
# Y_p generated with offset so that true model is log(mu) = log(exposure) + X@b
_exposure = np.random.poisson(5, n) + 1
Y_p = np.random.poisson(_exposure * np.exp(X[:,:3] @ [0.4,-0.3,0.2])).astype(float)

# Step 1: dev_ratio
print("=== Step 1: dev_ratio ===")
s = pycasso.Solver(X, Y_g)
s.train()
assert 'dev_ratio' in s.result, "dev_ratio missing"
assert 'nulldev' in s.result, "nulldev missing"
assert np.all(s.result['dev_ratio'] >= 0), "dev_ratio negative"
assert np.all(s.result['dev_ratio'] <= 1), "dev_ratio > 1"
print(f"  gaussian dev_ratio range: [{s.result['dev_ratio'].min():.3f}, {s.result['dev_ratio'].max():.3f}]")
print("  PASS")

# Step 2: predict types
print("\n=== Step 2: predict types ===")
# gaussian link == response
p_link = s.predict(X[:5], type="link")
p_resp = s.predict(X[:5], type="response")
assert p_link.shape == (5,), f"link shape wrong: {p_link.shape}"
assert np.allclose(p_link, p_resp), "gaussian link != response"

# nonzero
nz = s.predict(X[:5], type="nonzero")
assert isinstance(nz, np.ndarray), f"nonzero not array: {type(nz)}"

# binomial
sb = pycasso.Solver(X, Y_b, family="binomial")
sb.train()
probs = sb.predict(X[:5], type="response")
assert probs.shape == (5,), f"binomial probs shape: {probs.shape}"
assert np.all(probs >= 0) and np.all(probs <= 1), "binomial probs out of range"
cls = sb.predict(X[:5], type="class")
assert set(cls).issubset({0, 1}), f"binomial class not binary: {set(cls)}"
link_b = sb.predict(X[:5], type="link")
assert link_b.shape == (5,), "binomial link shape"

# lam= parameter
p_lam = s.predict(X[:5], lam=s.lambdas[10])
p_idx = s.predict(X[:5], lambdidx=10)
assert np.allclose(p_lam, p_idx), "lam= exact match should equal lambdidx="
p_lam_interp = s.predict(X[:5], lam=(s.lambdas[10]+s.lambdas[11])/2)  # prints note
assert p_lam_interp.shape == (5,), "interpolated prediction shape"
print("  PASS")

# Step 3: assess
print("\n=== Step 3: assess / confusion ===")
a = sb.assess(X, Y_b)
assert 'deviance' in a, "assess missing deviance"
assert 'class_error' in a, "assess missing class_error"
assert len(a['deviance']) == sb.nlambda, "assess deviance length"
conf = sb.confusion(X, Y_b, lambdidx=[0, 5, 10])
assert len(conf) == 3, "confusion list length"
assert conf[0].shape == (2, 2), "confusion matrix shape"
print("  PASS")

# Step 4: offset
print("\n=== Step 4: offset ===")
# Y_p was generated as Poisson(_exposure * exp(X@b)), so true model uses log(_exposure) as offset
sp = pycasso.Solver(X, Y_p, family="poisson", offset=np.log(_exposure))
sp.train()
assert 'dev_ratio' in sp.result, "poisson with offset missing dev_ratio"
print(f"  poisson offset dev_ratio range: [{sp.result['dev_ratio'].min():.3f}, {sp.result['dev_ratio'].max():.3f}]")
assert sp.result['dev_ratio'].max() > 0.01, \
    f"Expected dev_ratio > 0.01 when signal is present, got {sp.result['dev_ratio'].max():.4f}"
# compare: without offset, dev_ratio should be lower (offset matters)
sp_no = pycasso.Solver(X, Y_p, family="poisson")
sp_no.train()
print(f"  without offset dev_ratio max: {sp_no.result['dev_ratio'].max():.3f}")
print(f"  with offset dev_ratio max:    {sp.result['dev_ratio'].max():.3f}")
# ValueError if offset used with wrong family
try:
    _ = pycasso.Solver(X, Y_g, family="gaussian", offset=np.ones(n))
    assert False, "Should have raised ValueError"
except ValueError:
    pass
print("  PASS")

# Step 5: cross_validate
print("\n=== Step 5: cross_validate ===")
cv = s.cross_validate(nfolds=5, type_measure="mse")
assert 'lambda_min' in cv, "cv missing lambda_min"
assert 'cvm' in cv, "cv missing cvm"
assert len(cv['cvm']) == s.nlambda, f"cvm length mismatch: {len(cv['cvm'])} != {s.nlambda}"
print(f"  lambda_min={cv['lambda_min']:.4f}, lambda_1se={cv['lambda_1se']:.4f}")
print("  PASS")

# Step 6: multinomial
print("\n=== Step 6: multinomial ===")
Y_mn3 = np.random.choice(3, n)
sm = pycasso.Solver(X, Y_mn3, family="multinomial")
sm.train()
assert sm.result['beta'].shape == (sm.nlambda, 3, d), \
    f"multinomial beta shape wrong: {sm.result['beta'].shape}"
probs_mn = sm.predict(X[:5], type="response")
assert probs_mn.shape == (5, 3), f"multinomial probs shape: {probs_mn.shape}"
assert np.allclose(probs_mn.sum(axis=1), 1.0, atol=1e-6), "multinomial probs don't sum to 1"
cls_mn = sm.predict(X[:5], type="class")
assert cls_mn.shape == (5,), f"multinomial class shape: {cls_mn.shape}"
assert set(cls_mn).issubset({0, 1, 2}), f"multinomial class out of range: {set(cls_mn)}"
nz_mn = sm.predict(X[:5], type="nonzero")
assert isinstance(nz_mn, list) and len(nz_mn) == 3, "multinomial nonzero"
link_mn = sm.predict(X[:5], type="link")
assert link_mn.shape == (5, 3), f"multinomial link shape: {link_mn.shape}"
print(f"  beta shape: {sm.result['beta'].shape}")
print(f"  probs sample: {probs_mn[0]}")
print("  PASS")

# __str__
print("\n=== __str__ ===")
print(str(s))

print("\nAll tests passed.")
