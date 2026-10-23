# functional alignment of fMRI data with fmralign

Materials presented at the [2026 MAIN Educational](https://main-educational.github.io) to introduce the software `fmralign` for aligning functional magnetic resonance imaging data (fMRI). 

No two brains are organized exactly alike, and anatomical registration alone does not bring functional responses into correspondence.

In this hands-on tutorial, we introduce practical implementations of functional alignment using `fmralign`, a Python library in the [`scikit-learn`](https://scikit-learn.org/stable/) ecosystem to align brain activity across human participants, using methods such as Procrustes, Shared Response Modeling (SRM), and Optimal Transport.
`fmralign` builds heavily on [`nilearn`](https://nilearn.github.io/stable/index.html) ; we therefore also spend some time introducing the `nilearn` library to handle fMRI data loading and feature extraction for alignment.

To build the book locally, make sure that you have the requirements installed :

```
pip install -r requirements.txt
```

Note that this site is still built with jupyter-book 1.0 !

Then :

```
jupyter-book build content/
```

Generated notebooks are then available in `content/_build/jupyter_execute`,
and the rendered html is in  `content/_build/html/index.html`.