# License and attribution status

This repository does not currently contain a repository-wide license grant.
Public availability is not a statement that every component has unrestricted
reuse rights. A blanket MIT/Apache/data license has not been inferred or added.

## Simulator and task ancestry

The environment derives from the Virtual Tools work by K. R. Allen, K. A.
Smith, and J. B. Tenenbaum, *Rapid trial-and-error learning with simulation
supports flexible tool use and physical reasoning* (PNAS, 2020).

- Original code and layouts: [k-r-allen/tool-games](https://github.com/k-r-allen/tool-games),
  inspected at `7ecea95efb3d61370779b274292db26728df53ae`.
- The vendored `simulator/runtime/pyGameWorld/conditions.py`, `gameTypes.py`,
  and `jsrun.py` match that upstream's Git blobs exactly. Other local simulator
  files have modifications; attribution applies independently of Git authorship.
- No license covering that repository's Python simulator or task layouts was
  found in its inspected tree. The only LICENSE files found there belong to
  nested JavaScript color dependencies and do not establish a license for the
  simulator or layouts.
- The separate [kasmith/virtual-tools-backbone](https://github.com/kasmith/virtual-tools-backbone)
  repository has an MIT license. It is a different source tree; that license
  has **not** been treated as a retroactive license for this vendored code.

Maintainers still need to establish applicable upstream permissions and choose
licenses for their own code and derived task assets. This document records the
unresolved issue; it is not a substitute permission grant.

## Installed dependencies

Dependencies in `requirements.txt` are installed separately, not relicensed by
this repository. See their distributed license files, including
[Pymunk's license](https://github.com/viblo/pymunk/blob/master/LICENSE.txt) and
[Pygame's license](https://github.com/pygame/pygame/blob/main/docs/LGPL.txt).
The pinned numerical reproduction environment is described in
`analysis/REPRODUCIBILITY.md`. An API subscription or provider model license is
not a license for the task assets.

## Outputs and participant data

Model responses are archived research outputs with their requested/returned
model IDs and provider provenance. No broader rights to provider services or
weights are granted here. Individual human participant records are excluded
from this public release. Upstream project references are attributions, not
imports of upstream participant records.

The original July 27 release discussion explicitly left license selection
unfinished. A later authorization to publish did not document upstream rights
or add a license file. The follow-up preserves that distinction.
