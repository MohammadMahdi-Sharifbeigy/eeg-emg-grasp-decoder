# Performance Optimization Design

**Goal:** Eliminate GPU OOM on the current GPU and reduce epoch time substantially for EEG-to-EMG training without requiring new hardware.

## Objective

The current training path is too slow and too memory-hungry for practical iteration:
- around 2 hours per epoch for 3 participants
- 10+ hours per epoch when scaled further
- recurrent GPU OOM pressure

This design defines a staged optimization plan focused on the current GPU as a hard deployment constraint. It allows moderate changes to the training pipeline and model/runtime configuration when those changes materially improve memory use or throughput.

## Constraints

- Must fit and run on the current GPU
- Must reduce peak VRAM pressure before attempting larger experiments
- May introduce moderate architectural/runtime simplifications
- Should preserve the overall EEG -> EMG learning objective
- Should avoid blind tuning and instead use measured profiling checkpoints

## Assumed Primary Bottlenecks

Based on the current repository structure and training path, the likely bottlenecks are:

1. Too many windows generated from long continuous series
2. Repeated preprocessing/projection work that should be cached or precomputed
3. High sequence cost from window length and stride choices
4. Heavy memory pressure from sequence activations in the transformer
5. Slow or disabled mixed-precision paths in some parts of the loop
6. DataLoader or CPU-side preparation overhead that underfeeds the GPU
7. Expensive loss components or evaluation patterns that are disproportionate on the current hardware

The implementation plan should validate these assumptions with profiling instead of treating them as fixed truths.

## Strategy

The optimization plan should be executed in ordered stages. Each stage should have measurable outputs before proceeding.

### Stage 1: Establish Baseline and Profiling

Before changing the pipeline, capture a reproducible baseline on a small representative run.

Measure:
- epoch wall-clock time
- average batch time
- data-loading time vs model time
- GPU allocated and reserved memory
- CPU RAM pressure
- GPU utilization if available
- forward-pass vs backward-pass timing

Required baseline setup:
- fixed participant subset
- fixed batch size
- fixed max epochs or one-epoch profiling run
- fixed config snapshot

Required profiling outputs:
- one markdown log summarizing the bottleneck breakdown
- one table comparing `time per batch`, `samples per second`, `peak VRAM`, and `OOM status`

The purpose of Stage 1 is to identify the worst cost centers before making changes.

### Stage 2: Memory Stabilization First

OOM must be solved before speed improvements matter.

Priority interventions:

1. Ensure mixed precision is actually active end-to-end on CUDA.
2. Reduce activation footprint before increasing throughput.
3. Remove any avoidable tensor duplication or CPU/GPU round-trips.
4. Use smaller true micro-batches with optional gradient accumulation only if needed.
5. Prefer model-width and sequence-cost reductions over ad hoc cache-clearing workarounds.

Concrete actions:
- verify `torch.amp.autocast` and `GradScaler` are enabled during train steps
- confirm loss code paths remain numerically stable under AMP
- verify CCA projection stays on-device and does not trigger hidden host transfers
- tune `batch_size` downward only as much as required to fit
- introduce `gradient_accumulation_steps` only after measuring whether throughput remains acceptable
- reduce unnecessary retained tensors in logs, metrics, or checkpoint logic
- avoid accumulating large prediction arrays during training-time validation if not necessary

Expected result:
- a stable configuration that completes training without OOM on the current GPU

### Stage 3: Throughput Optimization

Once OOM is contained, optimize for samples processed per second.

#### 3.1 Precompute and Cache Expensive Work

Any deterministic preprocessing that does not need to happen every epoch should be moved out of the hot path.

Candidates:
- EEG preprocessing outputs
- EMG preprocessing outputs
- extracted `k_t`
- CCA-projected EEG features for train/val/test once the train-fit projection is available

Preferred direction:
- offline cache preprocessed arrays
- optionally cache post-CCA arrays split by participant and series
- keep runtime training focused on loading ready tensors instead of recomputing scientific transforms

Important caveat:
- training-set-only fitting rules must be preserved for CCA and normalization stats

#### 3.2 Fix Data Loading and Host-to-Device Transfer

The current loop should be profiled and tuned for loader throughput.

Actions:
- benchmark `num_workers` values instead of assuming one setting is best
- keep `pin_memory=True` on CUDA
- use `persistent_workers=True` when stable
- test `prefetch_factor` if worker mode is active
- ensure dataset objects are picklable and worker-safe
- minimize Python object overhead in `__getitem__`
- prefer contiguous arrays/tensors where possible

The goal is to keep the GPU fed consistently.

#### 3.3 Reduce Effective Sequence Cost

Long sequences dominate transformer memory and time.

The plan should explicitly evaluate:
- increasing stride to reduce window count
- shortening window length if performance remains acceptable
- using a coarser temporal representation before the transformer
- downsampling or patching the temporal input for the model path only

This is likely one of the highest-impact levers on current hardware.

#### 3.4 Replace Disproportionately Expensive Components

If a component has poor cost-benefit on the current GPU, the plan should allow it to be simplified.

Candidates:
- disable Soft-DTW during the stabilization phase
- train with pure MSE first, then optionally fine-tune with a hybrid loss
- defer expensive metrics or compute them less frequently
- reduce validation frequency if validation is too costly

This is acceptable because the user explicitly allows moderate changes for performance.

### Stage 4: Moderate Model Simplification

The current hardware likely cannot support the full cost of an aggressively sized temporal model across all subjects efficiently.

The optimization plan should evaluate targeted simplifications such as:
- lower `d_model`
- lower FFN width
- fewer transformer layers
- fewer attention heads where they are not justified
- temporal pooling or projection before the heaviest layers
- staged training on a smaller temporal representation

These changes should be ranked by:
- VRAM reduction
- speed gain
- likely effect on predictive quality

The goal is not arbitrary downsizing; it is cost-effective simplification.

### Stage 5: Validation and Regression Control

Each successful optimization step should be checked against:
- training stability
- mean validation loss
- mean Pearson correlation
- epoch time
- peak VRAM

Every major change should be recorded in a comparison table so the best tradeoff can be selected rather than guessed.

## Recommended Order of Execution

The implementation plan should recommend this order:

1. Profile one baseline run
2. Verify AMP and on-device projection behavior
3. Stabilize memory with batch/micro-batch control
4. Cache deterministic preprocessing outputs
5. Optimize DataLoader throughput
6. Reduce sequence cost
7. Disable or stage expensive loss components
8. Apply moderate model simplifications
9. Re-profile and compare against baseline

This order minimizes wasted work and avoids optimizing the wrong bottleneck first.

## Specific Techniques the Final Plan Should Cover

The step-by-step plan should explicitly address:
- mixed precision verification
- gradient accumulation tradeoffs
- pinned memory and worker tuning
- offline preprocessing caches
- optional post-CCA feature caches
- stride/window redesign
- sequence-length reduction
- transformer-width reduction
- selective validation/evaluation frequency
- profiling with `torch.profiler` or equivalent timing instrumentation
- memory summaries with `torch.cuda.memory_allocated` and `memory_reserved`

## Non-Goals

This design is not:
- a hardware purchasing plan
- a full model-research redesign
- a switch to a different dataset protocol
- a guarantee that all optimization comes only from hyperparameter tweaks

## Deliverable Shape

The follow-up markdown plan should be:
- step-by-step
- ordered by impact and dependency
- explicit about what to measure after each change
- focused on the current GPU
- concrete enough to execute without reinterpretation

## Success Criteria

The optimization effort is successful if it produces:
- one stable no-OOM training configuration on the current GPU
- a materially reduced epoch time compared with the current baseline
- a documented sequence of changes with measured impact
- a clear recommendation for the best runtime-quality tradeoff configuration
