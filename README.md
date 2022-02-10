# Automated Encoding Selection
### Reproducibility Repository

This repository contains source code and artifacts for the paper **Robust and Budget-Constrained Encoding Configurations for In-Memory Database Systems** (VLDB 2022).

In case you have any questions, please contact [Martin Boissier](https://hpi.de/plattner/people/phd-students/martin-boissier.html).

### Citation

This is a preliminary BibTeX entry. The paper has been accepted but it might move to another issue in case the camera ready version is not accepted.
<details><summary>BibTeX entry (click to expand)</summary>
```bibtex
@article{DBLP:journals/pvldb/Boissier22,
  author    = {Martin Boissier},
  title     = {Robust and Budget-Constrained Encoding Configurations for In-Memory Database Systems},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {15},
  number    = {4},
  pages     = {780--793},
  year      = {2022},
  url       = {http://www.vldb.org/pvldb/vol15/p499-boissier.pdf}
}
```
</details>

## Setup

The repository contains the `encoding_plugin`. This is a Hyrise plugin that manages the communication with the Hyrise server. Hyrise itself is a third party module within the plugin.
