# System Architecture

Phase 1 defines the architecture for the end-to-end project. The executable NS-3 and ML integration will be added in later phases.

```text
NS-3 wireless network simulation
  -> dynamic network conditions
  -> raw CSV output
  -> preprocessing and validation
  -> feature engineering
  -> Random Forest / XGBoost channel-selection model
  -> optimization decision
  -> AI vs deterministic baseline evaluation
  -> metrics, figures, and report material
```

## Primary Optimization Decision

The first working version will optimize channel selection. This keeps the initial research scope narrow enough to verify scientifically while still exposing real wireless-network behavior.

## Baseline

The conventional baseline is deterministic lowest-interference channel selection. Ties are broken by higher SNR and then lower channel ID. This baseline is reproducible and technically meaningful for a multi-channel wireless scenario.

## Metrics

- Throughput: successfully received bits divided by observation time.
- Latency: mean end-to-end packet delay.
- Packet delivery ratio: received packets divided by transmitted packets.
- BER: bit errors divided by total transmitted bits, where the simulation design supports bit-level error accounting.
- Energy consumption: transmit power multiplied by active duration.
- Spectrum efficiency: throughput divided by bandwidth.

