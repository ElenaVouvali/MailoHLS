# Coroni guide reference code

Read [`docs/coroni_chatgpt_synthesis_guide.md`](../../docs/coroni_chatgpt_synthesis_guide.md)
and use `CHATGPT_PROMPT.md` with the actual requested JSONL. The files here preserve
the original guide/parser source bytes, source/action inventories, toolchain hash
receipt and nine small historical top-level XML report fixtures.

Run the parser tests without SSH or HLS:

```bash
python3 -m unittest discover -s experiments/coroni_synthesis_guide -p test_report_parser.py -v
```

The original private archive and full Coroni source/header snapshot remain in
Liono's ignored `artifacts/coroni_synthesis_guide_20261113/`; they are not duplicated
in Git. Obtain the authenticated application sources through an authorized Coroni
operator when building a synthesis batch. The inventory records the source version
and hashes; an inventory entry is not permission to assume source equivalence.

Historical fixture QoR values are parser test data, not results of a new queue.
Undefined latency remains unknown, utilization uses each report's own resource
capacities, and running synthesis still requires an authorized execution path.
