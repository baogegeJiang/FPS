# TODO

## Feature Bank Release

- Use Hugging Face Dataset Hub as the primary hosting location for public feature-bank H5 files.
- Do not upload the current H5 banks as final artifacts yet; the exact bank schema/view set is still being finalized.
- After the H5 layout is locked:
  - regenerate the official banks;
  - compute and record SHA256 checksums;
  - upload banks to Hugging Face;
  - add/update a downloader script for `fps_h5cache/banks/`;
  - document the HF repo ID and download commands in `README.md`.
