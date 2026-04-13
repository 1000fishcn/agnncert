# Paper-overlap benchmark analysis

- Total completed runs: 45
- Total failed runs: 2
- Paper datasets without matching checkpoints: Amazon2M, Big-Vul, MUTAG
- Average speedup: 0.8187x
- Average accuracy delta: +0.005086
- Average macro-F1 delta: +0.003308
- Average direct-route ratio: 0.1456
- Average realized early-stop ratio: 0.9929

## Grouped means

- Task `graph`: speedup 1.5721x, accuracy delta +0.000031, macro-F1 delta +0.000158, direct-route ratio 0.0948.
- Task `node`: speedup 0.3165x, accuracy delta +0.008456, macro-F1 delta +0.005409, direct-route ratio 0.1794.

- Robust mode `e`: speedup 0.9925x, accuracy delta +0.009221, macro-F1 delta +0.006338, direct-route ratio 0.2254.
- Robust mode `n`: speedup 0.6797x, accuracy delta +0.001778, macro-F1 delta +0.000885, direct-route ratio 0.0817.

- Split `graph` + `e`: speedup 1.6899x, accuracy delta -0.000257, macro-F1 delta -0.000576, direct-route ratio 0.0981.
- Split `graph` + `n`: speedup 1.4544x, accuracy delta +0.000319, macro-F1 delta +0.000891, direct-route ratio 0.0915.
- Split `node` + `e`: speedup 0.4219x, accuracy delta +0.016976, macro-F1 delta +0.011994, direct-route ratio 0.3296.
- Split `node` + `n`: speedup 0.2440x, accuracy delta +0.002599, macro-F1 delta +0.000881, direct-route ratio 0.0762.

## Highlights

- Fastest gain: AIDS e T=10 with 3.0756x speedup and accuracy delta -0.002311.
- Best accuracy gain: computers e T=400 with accuracy delta +0.071239 and speedup 0.8216x.
- Best macro-F1 gain: PubMed e T=400 with macro-F1 delta +0.039593.
- Fast dataset mean: AIDS averages 2.1042x speedup and -0.000257 accuracy delta.
- Fast dataset mean: DD averages 1.3249x speedup and +0.000350 accuracy delta.
- Fast dataset mean: PROTEINS averages 1.2874x speedup and +0.000000 accuracy delta.
- Slow dataset mean: CiteSeer averages 0.1780x speedup and +0.000105 accuracy delta.
- Slow dataset mean: Cora-ML averages 0.2651x speedup and +0.004804 accuracy delta.
- Slow dataset mean: PubMed averages 0.2779x speedup and +0.011877 accuracy delta.
- Most runs still rely mainly on subgraph voting; with route_confidence=0.85, direct original-graph routing is often too strict for these checkpoints.
- The main runtime win comes from early stopping inside subgraph voting rather than direct routing.

## Remaining failures

- Some checkpoints are present on disk but cannot be reproduced with the current dataset build; the most common issue is checkpoint-model shape mismatch.
- PROTEINS n T=20: 	size mismatch for conv1.lin.weight: copying a param with shape torch.Size([32, 3]) from checkpoint, the shape in current model is torch.Size([32, 4]). (log: D:\python\1\AGNNCert\paper_overlap_outputs\graph_n_PROTEINS_GCN_T20\benchmark.log)
- CiteSeer e T=400:  (log: D:\python\1\AGNNCert\paper_overlap_outputs\node_e_CiteSeer_GCN_T400\benchmark.log)
