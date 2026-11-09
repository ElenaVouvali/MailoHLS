# GNN architecture and supervision

The V1 paper encoder is four TransformerConv layers with JK-max and concatenated first/JK states. RC3 V4b is two typed64D layers and a learned64-to-128 projection. Equal64x128 export shape does not establish compatibility. V1 attention weights must never read RC3 memory without fresh matched fusion training.

The corrected V4b frozen compiler checkpoint is authenticated by SHA ef61521989d8cf3282b28f881f0cd7673de2a5253f1d42b57542e21531019162. It is a legitimate experimental feature extractor, selected by an ADP predictor criterion. No evidence here selects it as the optimal latency Stage2 encoder. New hardware-message or FiLM rejection would be unjustified; see V4B_IMPLEMENTATION_REAUDIT.md.

RC3 graphs preserve action slots, loop hierarchy, affine accesses, SSA recurrence witnesses and explicit knownness. Edge witnesses in C are deterministic composites; a node embedding is not proof that every relation is recoverable. Static memory must contain neither candidate directives, budgets, objective labels nor measured QoR. Physical context may change hardware overlay embeddings; requested budget changes ranking eligibility, not fixed-design physical facts.

Archive reconciliation inherited from the prior branch:63 canonical variants /61 names,55 measured-source matches,40 curated V4 kernels,38 complete mappings,20 pilot packs. These are different denominators. Ambiguous variant names are preserved rather than silently overwritten. Hotspot repairs staged in the user's other checkout were not imported.

No depth/JK or latency-supervision training was launched. Test compiler/hardware/FiLM under matched latency feasible-ranking labels before considering depth. Auxiliary predictor superiority cannot select exported memory without downstream matched generated-design evaluation.
