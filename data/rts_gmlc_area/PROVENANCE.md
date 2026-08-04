# Provenance

Converted from **RTS-GMLC** by `scripts/import_rts_gmlc.py`.

- Upstream: https://github.com/GridMod/RTS-GMLC
- Aggregation: `area` | startup tier: `hot`
- Hours: 8784

## License

RTS-GMLC carries no SPDX license file. Its README contains an NREL "Data
Use Disclaimer Agreement" granting "the right, without any fee or cost, to
use, copy, and distribute these Data for any purpose whatsoever, provided
that this entire notice appears in all copies", with attribution and an
indemnity clause. Reproduce that notice alongside any redistribution of
this converted data.

## Conversion decisions

- Heat rates come from `SourceData/gen.csv` (average heat rate at minimum
  load plus incremental bands), collapsed to slope + intercept by the
  secant between minimum and maximum stable output. The bundled openTEPES
  export was **not** used: it zeroes the no-load term, VOM and shutdown
  cost for every unit and states startup cost in thousands of dollars.
- Startup cost = start heat (`hot` tier) x fuel price +
  `Non Fuel Start Cost $`.
- Ramp rates converted from MW/min to MW/h. Units with no rate recorded
  are given a full-range hourly ramp.
- Minimum up/down times rounded **up** to whole hours (the source has
  2.2 h and 4.5 h values; gridlock truncates, which would relax them).
- Hydro, run-of-river and CSP are modelled as profile-driven generators
  with no commitment state, since gridlock has no reservoir model.
  Synchronous condensers (0 MW) are dropped.
- Storage round-trip efficiency converted from percent; energy from
  `storage.csv` head volumes (GWh -> MWh).
- Line loss factors are a flat 1% (RTS-GMLC publishes none; matches the
  upstream openTEPES export).

## Network caveat

RTS-GMLC line ratings are thermal limits for a **DC power flow**. gridlock
is a transport model and ignores Kirchhoff's voltage law, so with
`--aggregate none` flows route around congestion in ways the real system
cannot and cost is optimistically low. `--aggregate area` collapses the
three areas to single nodes joined by summed tie capacities, which is a
defensible transport representation.
