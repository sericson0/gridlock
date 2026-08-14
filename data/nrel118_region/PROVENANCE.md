# Provenance

Converted from **NREL-118** by `scripts/import_nrel118.py`.

- Upstream mirror: https://github.com/Sienna-Platform/PowerSystemsTestData (`118-Bus/`)
- Original: Peña, Brancucci Martinez-Anido & Hodge, "An Extended IEEE
  118-Bus Test System With High Renewable Penetration", IEEE Transactions
  on Power Systems 33(1):281-289, 2018. DOI 10.1109/TPWRS.2017.2695963
- Aggregation: `region` | Hours: 8784

## License — READ BEFORE REDISTRIBUTING

The upstream mirror carries **no license file**, so it is
all-rights-reserved by default. This converted dataset is for local
research only; do not redistribute it. Use RTS-GMLC (which carries an
explicit NREL grant) for anything published.

Note the original NREL download URLs are dead: NREL was renamed the
National Laboratory of the Rockies and `nrel.gov` no longer resolves.

## Conversion decisions

- Heat rates read directly: `Heat Rate Base (MMBTU/hr)` is a genuine
  no-load fuel input and maps onto gridlock's intercept with no fitting.
  Units with a single incremental band convert exactly; multi-band units
  are collapsed by a secant between minimum and maximum stable output.
- Fuel prices are **not** in any CSV. They are hard-coded in
  `data_118bus.jl` (natural gas 5.4, coal 1.8, oil 21, biomass 2.4,
  geothermal 0 $/MMBtu), and that file's unit-to-fuel rule (by prime
  mover and name prefix) is reproduced in the importer.
- Ramp rates converted from MW/min to MW/h; units with none recorded get
  a full-range hourly ramp.
- Minimum up/down times rounded up to whole hours.
- Hydro is modelled as profile-driven generation with no commitment
  state (gridlock has no reservoir model).
- Nodal demand = regional day-ahead load x the published per-bus
  participation factors (`Load/partfact.csv`).
- Line loss factors are zero (none published). No storage is included.
- Bus coordinates are zeroed: NREL-118 ships no geography, and gridlock
  only uses coordinates for reporting.

## Network caveat

Line ratings are thermal limits for a **DC power flow**. gridlock is a
transport model and ignores Kirchhoff's voltage law, so the nodal variant
lets flow route around congestion in ways the real system cannot.
`--aggregate region` collapses to the three WECC regions joined by summed
inter-region capacity, which is a defensible transport representation.
