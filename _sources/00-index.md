---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.0
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Functional alignment of fMRI data with fmralign

Materials presented at the [2026 MAIN Educational](https://main-educational.github.io) to introduce the software `fmralign` for aligning functional magnetic resonance imaging data (fMRI). 

No two brains are organized exactly alike, and anatomical registration alone does not bring functional responses into correspondence.

```{code-cell} ipython3
from IPython.display import IFrame

IFrame("https://docs.google.com/presentation/d/e/2PACX-1vSQyJnbhSdZTC9ZGYs93aTJo6ppLyNGQ4AXtCRFjUgPFpda1AO8Dm01ursEM5dFGev7227-zzYcE5aF/pubembed?start=false&loop=false&delayms=3000", "480px", "299px")
```


In this hands-on tutorial, we introduce practical implementations of functional alignment using `fmralign`, a Python library in the [`scikit-learn`](https://scikit-learn.org/stable/) ecosystem to align brain activity across human participants, using methods such as Procrustes, Shared Response Modeling (SRM), and Optimal Transport.
`fmralign` builds heavily on [`nilearn`](https://nilearn.github.io/stable/index.html) ; we therefore also spend some time introducing the `nilearn` library to handle fMRI data loading and feature extraction for alignment.