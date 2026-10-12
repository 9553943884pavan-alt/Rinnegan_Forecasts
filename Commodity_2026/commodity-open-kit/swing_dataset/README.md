# swing-dataset

The `swing_dataset` package the Commodity Open 2026 kit depends on: the point-in-time client
(`swing_dataset.pack.pit`), the 2.6 scorer (`swing_dataset.scoring2`, with `FORMATS2.md`) and
the 2.4 one it superseded (`swing_dataset.scoring`, with `FORMATS.md`), the kit helpers
(`swing_dataset.kit`: formats, features, baselines, the loop, the organizer helpers) and the
builder of the realized-swing inventory entries are scored against (`swing_dataset.swings`).
The organizers' data-build lanes are not part of this copy. The kit's
`requirements.txt` installs it from this directory; on its own, `pip install .` from here does the
same. Do not edit the files here: the organizers score with their own copy of the package.
