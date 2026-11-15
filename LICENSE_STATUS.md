# License status — November 14, 2026

## MIT publication authorized by the maintainer

The repository maintainer stated on November 14, 2026 that permission for the
simulator-wide license had already been obtained, and then expressly directed
publication under the **MIT License**. The exact MIT terms are now in
[LICENSE](LICENSE); attribution and dependency exceptions are in
[NOTICE.md](NOTICE.md).

The source of this publication authorization is the maintainer's instruction.
An independent copy of the upstream permission correspondence was not located
in the connected mailbox, recovered release conversations, local experiment
directories, or archive catalog. We therefore do **not** describe the upstream
authorization document as independently verified or quote terms that were not
found. The previous absence of a selected repository license is superseded by
the maintainer's explicit MIT selection.

## Scope

| Material | Status |
| --- | --- |
| Project-authored scripts, evaluation code, and documentation | MIT under the repository maintainer's grant. |
| Bundled simulator code in `simulator/runtime/pyGameWorld/` and `scripts/pyGameWorld/`, including local modifications | Published under MIT on the basis of the maintainer's stated simulator-wide permission. Original attribution is preserved in `NOTICE.md`. |
| Project-authored goal evaluators and rollout utilities in `simulator/goal_semantics/` and `simulator/runtime/` | MIT; embedded third-party notices, if any, remain applicable. |
| Original upstream task layouts in `132_base_environments/cells/*/environment.json`, legacy environment assets, and their rendered images | The simulator-code permission does not by itself document an asset-specific grant. These upstream-derived assets are not represented as independently cleared under MIT. |
| Generated task configurations, sweeps, and scene corrections | Project-authored code and documentation are MIT; underlying upstream layout rights remain as above. The corrected images do not create a new license grant for those layouts. |
| Archived API responses, provider models, and externally sourced material | Their original terms apply; the repository MIT notice does not grant rights held by a provider or another third party. |
| Installed dependencies | Their own licenses apply; see `NOTICE.md` and the packages' original notices. |

## Verified ancestry

The primary simulator ancestry is
[`k-r-allen/tool-games`](https://github.com/k-r-allen/tool-games), inspected at
`7ecea95efb3d61370779b274292db26728df53ae`. Three runtime files
(`conditions.py`, `gameTypes.py`, and `jsrun.py`) are exact Git-blob matches;
other files include local modifications. The inspected upstream tree did not
contain a root simulator/task license. The exact comparison remains in
[`analysis/upstream_code_comparison.json`](analysis/upstream_code_comparison.json).

The separate [`kasmith/virtual-tools-backbone`](https://github.com/kasmith/virtual-tools-backbone)
repository has an MIT license at the previously inspected commit
`4e5dcf8e3713fc34f06bf6899f7c28ebc811dd6d`, but it is a different source tree.
Its license is not the basis for this release's grant. The basis is the
maintainer's explicit instruction and stated permission.

The remaining documentation gap is the original authorization record and an
explicit scope statement for upstream task assets. No license is invented
for those assets, and no experimental record was changed by this license update.
