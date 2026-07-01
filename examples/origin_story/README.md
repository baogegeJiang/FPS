# FPS-UDA Origin Story Prototype

This directory preserves the earliest FPS-UDA prototype as a historical
artifact. It is not the current benchmark pipeline, and it is not the dataset
level feature-bank format used by the released configs.

## Files

- `feature.h5`: the original feature-only H5 prototype with
  `src_feature`, `src_label`, `tgt_feature`, and `tgt_label`.
- `test2.ipynb`: the original exploratory notebook, saved unchanged.
- `test2_variant.ipynb`: another original exploratory notebook, saved
  unchanged with a filename that avoids parentheses.

## Notes

The notebooks are intentionally kept in their original form, including outputs,
development paths, device choices, and exploratory code. Some cells may reflect
nearby historical variants and may expect keys or files that are not present in
the included `feature.h5`.

Target labels in this prototype are used only for diagnostic or oracle-style
reporting during exploration. They are not part of the unsupervised training
signal in the current FPS-UDA benchmark workflow.
