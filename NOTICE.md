# Attribution and third-party materials

The Virtual Tools simulator and task framework originate with Kelsey R. Allen,
Kevin A. Smith, and Joshua B. Tenenbaum, **Rapid trial-and-error learning with
simulation supports flexible tool use and physical reasoning**, PNAS (2020).
Upstream source: <https://github.com/k-r-allen/tool-games>.

This repository includes locally modified Python simulator code in
`simulator/runtime/pyGameWorld/`,
alongside project-specific runners, goal evaluation, and analysis code. The
upstream attribution is retained; the repository copyright notice does not
claim authorship of the original simulator.

The maintainer authorized MIT publication of the simulator on November 14,
2026, after stating that simulator-wide permission had been obtained. See
[LICENSE](LICENSE) and [LICENSE_STATUS.md](LICENSE_STATUS.md) for the scope of
the maintainer authorization and the remaining task-asset limitation.

Third-party dependencies keep their own licenses. In particular:

- Pymunk: MIT, <https://github.com/viblo/pymunk/blob/master/LICENSE.txt>.
- Pygame: LGPL, <https://github.com/pygame/pygame/blob/main/docs/LGPL.txt>.
- NumPy, SciPy, Pillow, Matplotlib, imageio, API SDKs, and other installed
  packages retain the notices and licenses distributed with those packages.
- Any separately included third-party license notice takes precedence for
  the component to which it applies. This MIT grant does not relicense a
  dependency, a provider's model, or third-party model-generated material.

Original upstream task layouts and derived scene assets are identified
separately in [LICENSE_STATUS.md](LICENSE_STATUS.md). Their asset-specific
permission scope has not been independently documented. Attribution is not
a substitute for a license grant.
