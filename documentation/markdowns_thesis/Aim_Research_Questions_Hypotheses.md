# 4 Aim, Research Questions and Hypotheses

The research gap identified in Section 3.2 has three parts: a discrete SDR substrate, inhibition resolved into two interneuron types with distinct operations, and a test of whether those types contribute differently to encoding and to retrieval. This chapter states the aim of the thesis as the answer to that gap, breaks it into three research questions, and sets out the four hypotheses through which the questions are tested.

## 4.1 Aim

The aim of this thesis is to design, implement and evaluate a laminar cortical-column model that replaces the homogeneous k-winners-take-all inhibition of the HTM tradition with two distinct interneuron operators, a PV-like divisive normalisation and an SST-like dendritic gate, and to test whether these operators contribute differently to sparse encoding and to cue-driven retrieval in an SDR associative-memory column.

Each element of the aim answers one part of the gap. The SDR substrate, with its minicolumns, dendritic segments and local continual learning, is taken over from the HTM tradition (Section 2.2). The two operators follow the compartmental division of inhibition described in Section 1.3: the PV-like operator acts on the summed drive that reaches the output compartment, and the SST-like operator acts on the apical input before it is combined with basal drive. The test of differential contribution is the question that neither tradition has asked of this substrate. As in the hippocampal models reviewed in Section 2.1, which mode is active is set from outside the column, by presenting it with an encoding or a retrieval condition; what the thesis examines is how the two operators share the work within each mode.

The term *associative-memory column* is used here for a column that stores SDR assemblies and later recalls them from a cue, in the sense of the associative extensions of the HTM tradition reviewed in Section 2.3. It does not refer to Kanerva's Sparse Distributed Memory as a specific formalism, although the principle of sparse, similarity-based storage and recall is shared.

## 4.2 Research Questions

The aim is broken into three research questions.

**RQ1.** Can cell-type-differentiated inhibition be introduced into an SDR column while preserving sparse, unsupervised, continual learning? Local, continuous learning without a global error signal is what makes the HTM substrate attractive (Section 2.2.4), and a change to its inhibition is only useful if that property survives it.

**RQ2.** Does differentiated inhibition yield a dissociation between encoding and retrieval, in the sense that the two operators have their strongest effects in different modes? The hippocampal precedent suggests that such a division of labour is biologically plausible (Section 2.1); whether it appears in a neocortical SDR column is open.

**RQ3.** Is that dissociation functionally distinct from what homogeneous kWTA can produce? Strict kWTA bounds the active set by rank (Section 2.2.5), whereas a normalising operator with a threshold recruits units according to the strength of their input (Section 2.6.1). RQ3 is therefore addressed first, as a gate on the rest of the work: before the hypotheses below are tested, the PV-like operator must be shown to differ from kWTA in practice. Concretely, assembly size must vary with total input drive, an empty assembly must be reachable for sufficiently weak or ambiguous input, and the graded pre-threshold response must be retained rather than discarded. If the chosen parameters reach these behaviours only in a regime that silences every input, this is reported as not yet demonstrating a functional difference from kWTA.

## 4.3 Hypotheses

Four hypotheses make the research questions testable. H1 and H2 concern the two modes separately, H3 concerns what happens when both kinds of input arrive together, and H4 concerns how the two operators compare across modes.

**H1 (Encoding).** During encoding, feedforward input relayed through layer 4 recruits a sparse L2/3 assembly, and fast PV-mediated perisomatic inhibition, modelled as divisive normalisation, is the principal mechanism controlling the size of that assembly (H1a). PV inhibition additionally influences the precision of activation timing, once a coarse temporal dimension within each processing step, such as activation order or phase, is added to the L2/3 population (H1b). H1 rests on the perisomatic targeting of PV cells and their control of the gain and timing of spike output (Section 1.3), and on divisive normalisation as a canonical model of that control (Section 2.6.1). H1b requires an extension of the substrate. Should that extension prove infeasible within the project, H1 is evaluated on assembly size alone and timing is reported as a defined direction for future work.

**H2 (Retrieval).** During retrieval, a deep-layer or top-down cue reactivates the learned L2/3 assembly, and SST-mediated dendritic inhibition regulates the spread and persistence of that reactivation, preventing irrelevant or competing input from destabilising the recalled pattern. H2 rests on the apical targeting of SST cells (Section 1.3) and on the role of SST-type cells in protecting recall in the hippocampal model of Cutsuridis et al. [1]. The route of the cue is deliberately left generic, so that H2 can be tested without committing to a specific thalamic or corticocortical pathway.

**H3 (Apical–basal arbitration).** When basal (feedforward) and apical (retrieval) signals arrive together and disagree, apical–basal integration produces a graded outcome: a blend when the two are consistent, and suppression of, or a bias toward, the more strongly supported signal when they conflict, rather than a hard binary switch. The arbitration in H3 concerns which representation the column expresses, not which mode it is in. It rests on apical amplification at the cellular level [2] and on circuit models in which SST-like interneurons cancel predictable top-down input so that only the mismatch remains [3] (Section 2.6.2). Because the predicted outcome is graded, the measure of consistency between basal-driven and apical-driven candidate assemblies, for example their SDR overlap, is fixed before the test is run.

**H4 (Differential contribution).** PV and SST inhibition contribute differentially, not exclusively: PV inhibition has its strongest effect on encoding sparsity, and SST inhibition has its strongest effect on the selectivity and stability of cue-driven retrieval, with neither effect being zero on the other's measure. The claim is deliberately weaker than a double dissociation. The arithmetic of each interneuron class depends on stimulus and network state (Section 1.3), and in the hippocampal precedent the classes active in each mode overlap rather than separate cleanly [1]. H4 can therefore only be tested by varying the strength of each operator in graded steps, not by switching either on or off.

Table 4.1 summarises how each hypothesis, and the gate that precedes them, is to be tested. The simulations are specified in the methodology.

**Table 4.1.** How the research questions and hypotheses are to be tested.

| **Claim** | **Test** | **Main measure** |
|---|---|---|
| PV vs. kWTA gate (RQ3) | Encoding condition at varied input strength, run first | Assembly size against total drive; empty assembly reachable |
| H1a | Encoding condition; PV strength varied | L2/3 sparsity |
| H1b | Encoding condition with a within-step temporal dimension | Timing precision |
| H2 | Retrieval condition, cue only | Overlap with stored assembly; persistence over repeated cycles |
| H3 | Feedforward and recalled input together, consistency varied | Graded outcome on a pre-declared consistency measure |
| H4 | Graded sweep of PV × SST strength | Encoding and retrieval measures at every combination |

## Bibliography

[1] Vassilis Cutsuridis, Stuart Cobb, and Bruce P. Graham. "Encoding and Retrieval in a Model of the Hippocampal CA1 Microcircuit." *Hippocampus* 20.3 (2010), pp. 423–446. doi: 10.1002/hipo.20661.

[2] Matthew E. Larkum. "A Cellular Mechanism for Cortical Associations: An Organizing Principle for the Cerebral Cortex." *Trends in Neurosciences* 36.3 (2013), pp. 141–151.

[3] João Sacramento et al. "Dendritic cortical microcircuits approximate the backpropagation algorithm." *Advances in Neural Information Processing Systems* 31 (2018).
