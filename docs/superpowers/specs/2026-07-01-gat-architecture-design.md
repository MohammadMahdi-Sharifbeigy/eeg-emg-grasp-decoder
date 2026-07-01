# GAT Architecture Design

**Goal:** Define a clear architecture for adding a Graph Attention Network on top of the current transformer-based EEG-to-EMG pipeline, using the 5 EMG channels as graph nodes.

## Objective

The current runnable path is effectively:

`EEG -> preprocessing -> CCA -> Transformer -> linear EMG head`

This design introduces a structured graph reasoning layer between the transformer output and the final EMG predictions:

`EEG -> preprocessing -> CCA -> Transformer -> GAT over 5 EMG nodes -> EMG prediction`

The GAT should model task-dependent coordination among the five muscles while preserving the transformer as the temporal feature extractor.

## Scope

This is an architecture design, not an implementation plan.

It defines:
- how transformer outputs feed the graph
- what the graph nodes and edges represent
- how graph attention should work in this project
- what the GAT outputs should look like

It does not yet define:
- exact code edits
- training schedule details
- ablation order
- a full implementation task plan

## Node Definition

The graph contains exactly 5 nodes, one per EMG target channel:
- AD: anterior deltoid
- BR: brachioradialis
- FD: flexor digitorum
- ED: extensor digitorum
- FDI: first dorsal interosseus

These nodes are appropriate because:
- they match the project’s current prediction targets
- they are interpretable physiological endpoints
- they allow the GAT to model inter-muscle coordination directly

The graph is therefore a muscle-synergy graph rather than an EEG-channel graph or a multimodal entity graph.

## Inputs

### Transformer Output

Let the transformer produce:

`H_temp ∈ R^(B x T x D)`

where:
- `B` = batch size
- `T` = sequence length
- `D` = transformer hidden dimension

This representation is temporal and global; it does not yet contain explicit per-muscle node structure.

### Node-Feature Projection

Before the GAT, the transformer output must be transformed into per-node embeddings:

`H_nodes ∈ R^(B x T x N x F)`

where:
- `N = 5` muscle nodes
- `F` = node feature dimension

This can be produced by a learned projection:

`Linear(D -> N x F)`

followed by reshape:

`R^(B x T x D) -> R^(B x T x 5 x F)`

Interpretation:
- for each time step, the transformer latent state is decomposed into five muscle-specific node embeddings
- each node embedding becomes the feature vector for one muscle in the graph at that time step

### Optional Kinematic Conditioning

Kinematics should not become graph nodes in this design, but they may be used as auxiliary conditioning features for edge scoring.

Possible conditioning tensor:

`K_cond ∈ R^(B x T x K)`

where `K` is the kinematic feature dimension.

This allows graph attention to become context-sensitive without broadening the graph itself.

## Graph Structure

### Base Graph Topology

The default graph should be fully connected with self-loops:
- every muscle can attend to every other muscle
- each node also retains its own information

Reason:
- inter-muscle coupling is task-dependent
- a sparse hard-coded graph risks excluding useful coordination patterns
- the attention mechanism itself should learn which interactions matter at each time step

### Adjacency Representation

Adjacency can be represented in one of two ways:

1. Implicit full graph
- no explicit sparse edge list needed in the conceptual design
- attention scores are computed over all node pairs

2. Explicit adjacency mask
- a `5 x 5` mask with allowed edges and self-loops
- useful if the implementation later uses masked attention or graph libraries

The architecture design should assume a full `5 x 5` adjacency with self-loops as the default.

### Optional Structural Priors

The graph may optionally include priors from:
- anatomical knowledge
- known agonist/antagonist relationships
- synergy-derived similarity weights
- future NNMF-based connectivity

These priors should modulate or initialize the graph, not rigidly replace learned attention.

Recommended interpretation:
- use priors as bias terms, masks, or initialization aids
- keep final attention data-dependent

## Attention Mechanism

### Core Operation

At each time step `t`, the GAT operates over the 5 node embeddings:

`H_nodes[:, t, :, :] ∈ R^(B x 5 x F)`

For node `i`, attention is computed over nodes `j` in its neighborhood.

Standard learned attention form:

`e_ij = a(W h_i, W h_j)`

then normalized:

`alpha_ij = softmax_j(e_ij)`

and aggregated:

`h'_i = Σ_j alpha_ij * W h_j`

Interpretation in this project:
- the transformer captures temporal cortical structure
- the GAT redistributes that temporal information across a muscle-interaction graph
- attention coefficients represent context-dependent muscle coupling strength

### Multi-Head Attention

Multiple graph-attention heads should be used so different heads can capture different coordination patterns, such as:
- proximal vs distal muscle dependencies
- agonist vs antagonist relationships
- grasp-phase-sensitive couplings

Each head produces a distinct view of inter-muscle interaction.

Head outputs can then be:
- concatenated for richer representation, or
- averaged for a lighter architecture

The design should prefer concatenation in early or intermediate layers if memory permits, because the graph is small and interpretability benefits from richer head diversity.

### Timewise Operation

The GAT should operate independently at each time step over the 5-node graph.

This means:
- the transformer handles temporal modeling
- the GAT handles structured inter-muscle reasoning per time step

This division is clean and avoids asking the GAT to solve long-range temporal dependencies it is not designed to handle efficiently.

### Kinematic-Guided Attention Variant

An optional project-specific extension is to condition edge scoring on the kinematic state:

`e_ij^t = a(W h_i^t, W h_j^t, U k_t)`

This means:
- the graph remains the 5-muscle graph
- attention weights become dependent on the current movement state

This is well aligned with the project, because inter-muscle coordination during grasp, load, hold, and release is not static.

Recommended role of this extension:
- define it as the preferred advanced variant
- keep a plain GAT as the simpler baseline

## Outputs

### GAT Output Tensor

After one or more GAT layers, the output should remain node-aligned:

`H_gat ∈ R^(B x T x 5 x F_out)`

Each node embedding now represents a refined muscle-specific latent state after inter-muscle interaction modeling.

### Prediction Head

The final prediction target is the EMG envelope for each of the 5 muscles over time:

`Y_hat ∈ R^(B x T x 5)`

Recommended decoding:
- apply a small learned per-node projection from `F_out -> 1`
- squeeze the last dimension to recover the 5-channel time series

Equivalent implementation choices:
- shared linear applied across nodes
- independent per-node heads

Recommended design:
- start with a shared projection shape applied node-wise for simplicity and regularization
- keep independent heads as an ablation option

### Use in Prediction

The GAT output directly refines the regression target path:
- transformer output provides temporal context
- GAT output injects muscle-relationship structure
- decoder maps refined node embeddings to EMG amplitude predictions

This preserves the continuous regression objective already used in the project.

## Recommended Layering

The architecture should be staged like this:

1. EEG preprocessing
2. CCA projection
3. Transformer temporal encoder
4. Node projection into 5 muscle embeddings
5. One or more GAT layers over the 5-node muscle graph
6. Node-wise EMG decoder

This keeps module responsibilities clean and makes future ablations straightforward.

## Baseline and Variant Definitions

The eventual implementation should distinguish at least two variants:

### Baseline GAT

- transformer output projected to 5 nodes
- fully connected 5-node graph
- standard multi-head GAT
- node-wise decoder to EMG

### Kinematic-Guided GAT

- same node definition
- same 5-node graph
- attention score computation additionally conditioned on kinematics

This gives a simple baseline and a project-specific enriched version.

## Why This Design Fits the Project

This architecture is appropriate because:
- the project already predicts 5 EMG channels, so the graph nodes map naturally to outputs
- the transformer already provides sequence modeling, so the GAT can focus on muscle interaction structure
- the small graph size keeps graph computation tractable on limited hardware
- the resulting attention maps remain interpretable in physiological terms

## Success Criteria

The architecture design is successful if it provides:
- a clear tensor flow from transformer output to graph input
- a precise definition of the 5-node muscle graph
- a project-specific explanation of how attention should work
- a clear output definition mapping back to EMG predictions
- a clean separation between baseline GAT and kinematic-guided GAT variants
