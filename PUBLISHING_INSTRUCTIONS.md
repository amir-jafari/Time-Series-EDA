# tseda — PyPI Release Guide

**Use this guide for every future release.**
Publishing is done entirely from your terminal. No GitHub Actions required.

Run all commands from:
`D:\Deep Learning Book\All_repo_source_file\Packages\TimeSeriesEDA`

PyPI account: **TimeSeriesToolBox@gmail.com**
PyPI project: **https://pypi.org/project/timeseries-eda/**

---

## First-Time Setup (do once)

### API Token

You already have an account at **TimeSeriesToolBox@gmail.com** with another package on it.
You have two options for the token:

**Option A — Reuse your existing account-wide token** (from `Token.txt`)
If that token was created with scope "Entire account", it will also work for `tseda`.
Test it by running Step 8 — if you get `403 Forbidden`, use Option B.

**Option B — Create a new project-scoped token for `tseda`**
On your **first upload ever** (PyPI creates the project automatically). After the first
successful upload, go to:
1. https://pypi.org/manage/account/token/
2. Click **Add API token**
3. Name it `tseda-release`
4. Scope: **Project → tseda**
5. Copy the token — it is shown only once

> **UI step required:** The token must be created in the PyPI web UI. There is no terminal API for this.

---

## Every Release — 8 Steps


---

### Step 1 — Bump the version

Edit **`pyproject.toml`** only:

```toml
version = "0.1.X"
```

Then refresh the editable install so the metadata picks up the new version:

```powershell
pip install -e .
```

Confirm it is correct:

```powershell
python -c "import tseda; print(tseda.__version__)"
```

Must print the new version. If it prints the old one, save the file and re-run `pip install -e .`.

> `tseda` reads its version from package metadata (`importlib.metadata`), so only
> `pyproject.toml` needs to be changed — there is no separate `__init__.py` version string.

---

### Step 2 — Run the tests

```powershell
pytest tests/ -v
```

All tests must pass before continuing.

---

### Step 3 — Commit and push

```powershell
git add pyproject.toml
git commit -m "Bump version to 0.1.X"
git push origin main
```

Wait for CI to go green: https://github.com/amir-jafari/Time-Series-EDA/actions

---

### Step 4 — Delete the dist folder contents

Open `dist\` in Explorer and delete everything inside it.
This prevents accidentally uploading old version files.

Or from the terminal:

```powershell
Remove-Item dist\* -Force
```

---

### Step 5 — Build the distribution

```powershell
python -m build
```

Two files appear in `dist/`:
- `timeseries_eda-0.1.X-py3-none-any.whl`
- `timeseries_eda-0.1.X.tar.gz`

> No rename step needed — setuptools normalizes the sdist filename automatically.

---

### Step 6 — Verify the packages

```powershell
python -m twine check dist/*
```

Both files must show `PASSED` with the correct version number. Fix any warnings before uploading.

---

### Step 7 — Upload to PyPI

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "your-api-token-here"
python -m twine upload dist/*
```

Replace `your-api-token-here` with your token (from `Token.txt`).

Both files will show 100% and you will see:

```
View at: https://pypi.org/project/timeseries-eda/0.1.X/
```

---

### Step 8 — Tag the release



```powershell
git tag v0.1.X
git push origin v0.1.X
```

---

## Version Number Rules

| Change type                        | Example             |
|------------------------------------|---------------------|
| Bug fix or small improvement       | `0.1.0` → `0.1.1`  |
| New feature (backward compatible)  | `0.1.1` → `0.2.0`  |
| Breaking change                    | `0.2.0` → `1.0.0`  |

---

## Troubleshooting

**`403 Forbidden` on upload**
Your existing token (from `Token.txt`) is scoped to the other package, not `tseda`.
Create a new project-scoped token via the PyPI web UI (see First-Time Setup above).

**`ERROR: File already exists`**
Cannot re-upload the same version. Bump the version in `pyproject.toml` and restart from Step 1.

**`python -c "import tseda; print(tseda.__version__)"` prints the wrong version**
The editable install metadata is stale. Run `pip install -e .` to refresh it.

**Tests fail in CI but pass locally**
`MPLBACKEND=Agg` should be set in `.github/workflows/ci.yml`. Check the Actions tab for
the specific error.