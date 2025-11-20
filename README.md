# Scale-Dependent Decorrelation: RDT vs Collatz

(https://zenodo.org/badge/DOI/10.5281/zenodo.17487651.svg)](https://doi.org/10.5281/zenodo.17487651)

This repository contains the research paper and validation code demonstrating that Recursive Division Tree (RDT) depth and Collatz stopping time are fundamentally independent measures of integer complexity.

## Paper

**Title:** Scale-Dependent Decorrelation: RDT Depth and Collatz Stopping Time Independence Emerges at Large n

**Author:** Steven Reid  
**ORCID:** [0009-0003-9132-3410](https://orcid.org/0009-0003-9132-3410)

**Abstract:** We investigate the relationship between two integer complexity measures: Recursive Division Tree (RDT) depth and Collatz stopping time. Through systematic multi-scale analysis, we demonstrate that these measures are fundamentally independent: partial correlation controlling for log(n) yields r = 0.002, confirming orthogonality. The observed raw correlation exhibits scale-dependent decay (from r ≈ 0.20 for n < 10³ to r ≈ 0.03 for n > 5×10⁴), but this correlation is entirely spurious—arising from both measures exhibiting some dependence on integer magnitude.

## Key Finding

**RDT and Collatz are truly independent** (partial r = 0.002 after controlling for log(n)). The apparent raw correlation is entirely explained by both measures tracking integer magnitude to different degrees.

## Validation Scripts

Two Python scripts are provided to reproduce and validate the analysis:

### `rdt_vs_collatz.py`
Main analysis script that:
- Computes RDT depths and Collatz stopping times for n = 2 to 100,000
- Calculates correlation across the full range
- Performs stratified analysis by RDT depth
- Identifies extreme cases (high RDT/low Collatz and vice versa)
- Generates correlation statistics

**Usage:**
```bash
python rdt_vs_collatz.py
```

### `rdt_collatz_validation.py`
Comprehensive validation suite with 5 independent tests:

1. **Range Consistency Test** - Verifies correlation across different scales (reveals scale-dependence)
2. **Subsampling Stability Test** - Ensures results are robust to random sampling
3. **Control Comparison Test** - Compares to random baseline (proves RDT has real signal)
4. **Partial Correlation Test** - Controls for log(n) confounding (proves true independence)
5. **Monotonicity Check** - Tests for hidden nonlinear relationships

**Usage:**
```bash
python rdt_collatz_validation.py
```

**Requirements:** Python 3.8+, NumPy, SciPy, Matplotlib

## Citation

```bibtex
@article{reid2025collatz,
  author = {Reid, Steven},
  title = {Scale-Dependent Decorrelation: RDT Depth and Collatz Stopping Time 
           Independence Emerges at Large n},
  year = {2025},
  note = {Manuscript in preparation}
}
```

Original RDT algorithm:
```bibtex
@misc{reid2025rdt,
  author = {Reid, Steven},
  title = {Recursive Division Tree: A Log-Log Algorithm for Integer Depth},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.17487651}
}
```

## License

MIT License

## Acknowledgments

The author acknowledges the use of AI for assistance with code implementation and manuscript preparation. All theoretical insights, research methodology, and interpretations are the author's original work.
