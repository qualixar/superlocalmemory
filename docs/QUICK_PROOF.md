# Store and recall one synthetic fact

Install using the [Quick Start](../README.md#quick-start), choose operating mode
A in `slm setup`, and run `slm doctor` before using the CLI.

```bash
slm remember "Synthetic demo: the release checklist requires a human approval after tests pass." --tags demo --json --sync
slm recall "release checklist human approval" --json
```

On 1 October 2026, source version 4.1.17 returned a durable `queryable` receipt
and recalled that exact synthetic sentence. Recall returned one result with
`calibration_status: uncalibrated` and `answer_confidence: null`. A ranking score
is not an accuracy measurement. This one-fact check is not a retrieval benchmark.

The verification used a separate `SLM_DATA_DIR`, `SLM_DAEMON_PORT=8893` and
`SLM_DISABLE_LEGACY_PORT=1` to avoid an existing memory service. It did not write
real customer data or change host hooks. Use an unused port and a dedicated
absolute data directory if you reproduce that isolation. Setup and integrations
remain explicit choices; review them before changing an existing workspace.
