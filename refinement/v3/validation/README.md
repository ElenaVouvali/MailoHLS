# Publication validation

All 50 offline tests passed: 19 V3, 15 V2 and 16 legacy. V3 uses labelled mock processes for execution tests. The native audit verified 11 legacy records, 9 V2 records and both Jacobi V3 records (one reused baseline and one new candidate). No new real HLS process was invoked during publication.

The executable V3 code matches the frozen checkpoint. The cross-version test was updated only to allow the publication documentation and Git evidence rules outside the V3 directory. The earlier checkpoint validation remains under `../replay/`.
