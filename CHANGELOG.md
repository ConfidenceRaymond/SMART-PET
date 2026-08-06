# Changelog

## 0.3.0

- Licensed source, documentation, and distributed model weights under CC BY-NC-SA 4.0; commercial use is prohibited.
- Corrected the Ruff diagnostics found by the first Narval release validation.

- Added strict JSON configuration with command-line overrides.
- Added explicit configurable attention levels and checkpoint recording.
- Separated exact resume from weight-initialized fine-tuning.
- Added reusable inference engine and batch-inference CSV workflow.
- Added fixed-mask SUV evaluation CLI.
- Added portable single-GPU, DDP, and Narval launch examples.
- Removed institution-specific preprocessing, manifests, logs, caches, and temporary bundles from the public source tree.
- Documented differences from the 2024 published implementation and the validation-only status of current results.
