# Conda-forge Recipe for MechanicsDSL

This directory contains the conda-forge recipe for publishing MechanicsDSL to conda-forge.

## Submitting to conda-forge

1. **Fork the conda-forge/staged-recipes repository:**
   ```bash
   git clone https://github.com/conda-forge/staged-recipes
   cd staged-recipes
   ```

2. **Create a new branch:**
   ```bash
   git checkout -b mechanicsdsl
   ```

3. **Copy this recipe:**
   ```bash
   mkdir -p recipes/mechanicsdsl-core
   cp /path/to/mechanicsdsl-main/conda-recipe/meta.yaml recipes/mechanicsdsl-core/
   ```

4. **Update the SHA256 hash:**
   Download the tarball from PyPI and calculate the hash:
   ```bash
   pip download mechanicsdsl-core --no-deps --no-binary :all:
   sha256sum mechanicsdsl-core-*.tar.gz
   ```
   Update `meta.yaml` with the actual hash.

5. **Test locally (optional):**
   ```bash
   conda build recipes/mechanicsdsl-core
   ```

6. **Submit PR to staged-recipes:**
   - Push your branch
   - Open a pull request

## After Acceptance

Once accepted, the package will be available via:
```bash
conda install -c conda-forge mechanicsdsl-core
```

## Updating the Package

After the feedstock is created, update by:
1. Fork `conda-forge/mechanicsdsl-core-feedstock`
2. Update version in `meta.yaml`
3. Update SHA256 hash
4. Submit PR

## Dependencies

The recipe uses `matplotlib-base` instead of `matplotlib` to avoid pulling in Qt dependencies. This is standard practice for conda-forge recipes.
