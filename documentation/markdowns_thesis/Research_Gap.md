# 3 Research Gap

The two foundations answer different questions. The biological foundation describes the circuit the model follows: a laminar column whose inhibition is divided among interneuron classes that act on different compartments of the pyramidal cell. The computational foundation describes what existing models do with sparse coding, interneuron classes and associative memory, and places them on three axes: substrate, inhibition model, and the arbitration between encoding and retrieval. This chapter draws the two together. It first sets out what is already established and is therefore not claimed here (Section 3.1), then shows where these strands fall short and states the gap (Section 3.2), and finally answers the objections that a reader familiar with the field is most likely to raise (Section 3.3).

## 3.1 What Is Established

Most of the components of this thesis are not new, and the claim made here depends on keeping them apart from what is.

The compartmental division of inhibition is established biology. PV basket and chandelier cells inhibit the perisomatic region and the axon initial segment, SST Martinotti cells inhibit the apical tuft, and VIP cells disinhibit the tuft by inhibiting SST cells [1] (Section 1.3). That PV and SST cells play different roles in a circuit is also established in models. Rate models with multi-compartment pyramidal cells have shown how the VIP, SST and PV motif gates pathways and redistributes inhibition between soma and dendrite [2, 3], and laminar spiking models now place PV, SST and VIP populations in every layer of a column [4, 5].

The two operators that the thesis assigns to these classes are established as well. Divisive normalisation is a canonical neural computation [6], and its link to PV cells has direct, if contested, experimental support [7]. Dendritic gating of top-down input at the apical tuft is established at the level of single cells and circuits [8, 9, 10], and has been turned into a learning principle in which SST-like interneurons act on the apical compartment [11, 12].

The use of interneuron classes inside a memory network is not new either. In the attractor and BCPNN tradition, basket cells implement competition between minicolumns, Martinotti cells set how long an attractor lasts, and double bouquet cells make learned inhibition compatible with Dale's principle [13, 14]. Finally, the substrate is taken over unchanged from the HTM tradition: sparse distributed representations over minicolumns with dendritic segments [15, 16, 17], and the separation of encoding and retrieval as a functional requirement of memory has a clear precedent in the hippocampal model of Hasselmo et al. [18]. Cutsuridis et al. [19] have shown, in a model of hippocampal CA1, how PV-type and SST-type interneurons can divide the work of the two modes between them.

## 3.2 Where the Strands Fall Short

Each of these strands, then, is well developed on its own terms. What they have not done is come together. Table 3.1 shows this by reducing Table 2.1 to a single row per tradition, set against the three parts of the gap. No tradition covers more than two of them. The one model that does test how interneuron classes share encoding and retrieval is a biophysical model of hippocampal CA1, not a neocortical SDR column.

**Table 3.1.** The research gap at a glance. Each row condenses one tradition from Table 2.1; the columns are the three parts of the gap stated in the text.

| **Tradition** | **(i) Discrete SDR substrate** | **(ii) Operator-distinct PV and SST** | **(iii) Differential PV/SST roles in encoding and retrieval tested** |
|---|---|---|---|
| Hippocampal CA1 model | No: biophysical spiking cells | Partly: PV-type and SST-type cells, not formal operators | Yes, with the mode set by septal theta |
| HTM, CLA and SDR models | Yes | No: homogeneous kWTA | No: single inhibition rule |
| SDM and NAA | Yes (NAA); abstract binary (SDM) | No | No: single inhibition rule |
| Interneuron-resolved circuit models | No: rate or spiking populations | Yes | No: no memory task |
| Associative-memory columns | Partly: minicolumns, no learned SDRs | Partly: several classes, roles in retrieval dynamics | No: storage and recall roles not compared |
| Operator mechanisms | No | Partly: operators defined, cell attribution contested | No: no memory task |
| **This thesis** | **Yes** | **Yes** | **Tested, with the mode set externally** |

Put together, the gap has three parts: (i) a discrete SDR substrate in the HTM tradition; (ii) inhibition split into two interneuron types with distinct operations, a PV-like divisive normalisation on the output compartment and an SST-like dendritic gate on the apical input compartment; and (iii) an explicit test of whether these two operations contribute differently to encoding and to retrieval. This thesis sets out to hold all three at once. In practice, that means taking a biological distinction that is already well established, carrying it into a substrate that has so far reduced inhibition to a single sorting rule, and asking whether the move changes anything that homogeneous kWTA could not. As in the hippocampal models, which mode is active is set from outside the column, by the encoding or retrieval condition presented to it. What the thesis tests is how the two interneuron types share the work within each mode, and, where feedforward and recalled signals arrive together, how the column integrates them.

The contribution is therefore a modest one. The thesis does not offer a new account of cortical inhibition or a new memory model, but a controlled substitution inside an existing one, and the outcome is useful either way. If the two operations do divide encoding and retrieval between them, the SDR substrate gains a cell-type-specific division of labour between encoding and retrieval, of the kind Cutsuridis et al. describe for CA1. If they do not, that is evidence that, at least in this setting, the homogeneous rule loses less than the biology would suggest.

## 3.3 Anticipated Objections

**This has already been done in the hippocampus.** Cutsuridis et al. [19] assign storage and recall to different interneuron classes in CA1 and test those roles by removing the classes one at a time. This is the closest precedent for the thesis, and the thesis follows it in letting the mode be set from outside the circuit. The model, however, is a compartmental spiking model of a hippocampal region whose inputs arrive in separate dendritic layers from entorhinal cortex and CA3, and its interneurons are detailed cell models rather than operators that could replace a selection rule. The question of what happens when the single inhibition rule of an SDR column is replaced by two such operators, in a neocortical laminar setting, does not arise there.

**PV and SST have already been dissociated.** Hertäg and Sprekeler [3] model PV cells on the soma and SST cells on the apical dendrite, and show that the circuit can switch persistently between integrating and cancelling top-down input. This is the closest precedent for a circuit-internal switch and the thesis treats it as such. Its substrate, however, is a rate model with no sparse distributed representations and no learned associations, and the switch concerns how feedback is processed, not whether a pattern is stored or recalled.

**Interneuron classes have already been studied in memory networks.** Krishnamurthy et al. [13, 20] show that basket and Martinotti cells shape attractor memory differently, with Martinotti cells setting dwell time and producing surround suppression. Here the memory setting is real, but the roles concern the dynamics of a pattern that is already being recalled, in a spiking attractor network whose patterns were not learned as SDRs, and the change of mode in this tradition comes from neuromodulation [21].

**The mechanisms are borrowed rather than discovered.** This is correct and intended. The thesis does not propose a new form of inhibition. Its claim concerns the consequence of placing two known operators in a substrate that lacks them, which is a question the source models cannot answer because they do not use that substrate. A negative result, in which the two operators behave no differently from kWTA with respect to encoding and retrieval, would be as informative as a positive one.

**PV is not purely divisive and SST is not purely subtractive.** The arithmetic of each class depends on stimulus and network state [22, 23], and the attribution of division to PV cells is contested [24, 25]. The thesis therefore makes its claim about operators, not about cell classes. The operators are called PV-like and SST-like because their targets follow the compartmental targeting described in Section 1.3, which is robust, while the mapping of each operator to a single class is a modelling assumption on which the results about the operators do not depend.

**The HTM tradition already names its inhibitory cells.** Hawkins and Ahmad [16] identify fast-spiking basket cells as the likely source of the inhibition their model needs, and the author of NAA leaves the inhibition model explicitly open [26]. Naming a cell type is not implementing it: in both cases the computation remains a homogeneous k-winners-take-all rule (Section 2.2.5).

The aim of the thesis follows directly from this gap and is stated in the following chapter.

## Bibliography

[1] Robin Tremblay, Soohyun Lee, and Bernardo Rudy. "GABAergic Interneurons in the Neocortex: From Cellular Properties to Circuits." *Neuron* 91.2 (2016), pp. 260–292.

[2] Guangyu Robert Yang, John D. Murray, and Xiao-Jing Wang. "A dendritic disinhibitory circuit mechanism for pathway-specific gating." *Nature Communications* 7 (2016), p. 12815. doi: 10.1038/ncomms12815.

[3] Loreen Hertäg and Henning Sprekeler. "Amplifying the redistribution of somato-dendritic inhibition by the interplay of three interneuron types." *PLoS Computational Biology* 15.5 (2019), p. e1006999. doi: 10.1371/journal.pcbi.1006999.

[4] Yazan N. Billeh et al. "Systematic Integration of Structural and Functional Data into Multi-scale Models of Mouse Primary Visual Cortex." *Neuron* 106.3 (2020), pp. 388–403.e18. doi: 10.1016/j.neuron.2020.01.040.

[5] Giulia Moreni et al. "Cell-type-specific firing patterns in a V1 cortical column model depend on feedforward and feedback-driven states." *PLoS Computational Biology* 21.4 (2025), p. e1012036. doi: 10.1371/journal.pcbi.1012036.

[6] Matteo Carandini and David J. Heeger. "Normalization as a canonical neural computation." *Nature Reviews Neuroscience* 13.1 (2012), pp. 51–62. doi: 10.1038/nrn3136.

[7] Nathan R. Wilson et al. "Division and subtraction by distinct cortical inhibitory networks in vivo." *Nature* 488.7411 (2012), pp. 343–348. doi: 10.1038/nature11347.

[8] Matthew E. Larkum, J. Julius Zhu, and Bert Sakmann. "A new cellular mechanism for coupling inputs arriving at different cortical layers." *Nature* 398.6725 (1999), pp. 338–341. doi: 10.1038/18686.

[9] Matthew E. Larkum. "A Cellular Mechanism for Cortical Associations: An Organizing Principle for the Cerebral Cortex." *Trends in Neurosciences* 36.3 (2013), pp. 141–151.

[10] Masanori Murayama et al. "Dendritic encoding of sensory stimuli controlled by deep cortical interneurons." *Nature* 457 (2009), pp. 1137–1141. doi: 10.1038/nature07663.

[11] Robert Urbanczik and Walter Senn. "Learning by the dendritic prediction of somatic spiking." *Neuron* 81.3 (2014), pp. 521–528. doi: 10.1016/j.neuron.2013.11.030.

[12] João Sacramento et al. "Dendritic cortical microcircuits approximate the backpropagation algorithm." *Advances in Neural Information Processing Systems* 31 (2018).

[13] Pradeep Krishnamurthy, Gilad Silberberg, and Anders Lansner. "A cortical attractor network with Martinotti cells driven by facilitating synapses." *PLoS ONE* 7.4 (2012), p. e30752. doi: 10.1371/journal.pone.0030752.

[14] Nikolaos Chrysanthidis, Florian Fiebig, and Anders Lansner. "Introducing double bouquet cells into a modular cortical associative memory model." *Journal of Computational Neuroscience* 47 (2019), pp. 223–230. doi: 10.1007/s10827-019-00729-1.

[15] Subutai Ahmad and Jeff Hawkins. "How do neurons operate on sparse distributed representations? A mathematical theory of sparsity, neurons and active dendrites." 2016. arXiv: 1601.00720 [q-bio.NC].

[16] Jeff Hawkins and Subutai Ahmad. "Why Neurons Have Thousands of Synapses, a Theory of Sequence Memory in Neocortex." *Frontiers in Neural Circuits* 10 (2016), p. 23. doi: 10.3389/fncir.2016.00023.

[17] Yuwei Cui, Subutai Ahmad, and Jeff Hawkins. "The HTM Spatial Pooler: A Neocortical Algorithm for Online Sparse Distributed Coding." *Frontiers in Computational Neuroscience* 11 (2017), p. 111. doi: 10.3389/fncom.2017.00111.

[18] Michael E. Hasselmo, Clara Bodelon, and Brad P. Wyble. "A Proposed Function for Hippocampal Theta Rhythm: Separate Phases of Encoding and Retrieval Enhance Reversal of Prior Learning." *Neural Computation* 14.4 (2002), pp. 793–817.

[19] Vassilis Cutsuridis, Stuart Cobb, and Bruce P. Graham. "Encoding and Retrieval in a Model of the Hippocampal CA1 Microcircuit." *Hippocampus* 20.3 (2010), pp. 423–446. doi: 10.1002/hipo.20661.

[20] Pradeep Krishnamurthy, Gilad Silberberg, and Anders Lansner. "Long-range recruitment of Martinotti cells causes surround suppression and promotes saliency in an attractor network model." *Frontiers in Neural Circuits* 9 (2015), p. 60. doi: 10.3389/fncir.2015.00060.

[21] Erik Fransén and Anders Lansner. "A model of cortical associative memory based on a horizontal network of connected columns." *Network: Computation in Neural Systems* 9.2 (1998), pp. 235–264. doi: 10.1088/0954-898X_9_2_006.

[22] Sami El-Boustani and Mriganka Sur. "Response-dependent dynamics of cell-specific inhibition in cortical networks in vivo." *Nature Communications* 5 (2014), p. 5689. doi: 10.1038/ncomms6689.

[23] Bryan A. Seybold et al. "Inhibitory Actions Unified by Network Integration." *Neuron* 87.6 (2015), pp. 1181–1192.

[24] Seung-Hee Lee et al. "Activation of specific interneurons improves V1 feature selectivity and visual perception." *Nature* 488.7411 (2012), pp. 379–383. doi: 10.1038/nature11312.

[25] Bassam V. Atallah et al. "Parvalbumin-expressing interneurons linearly transform cortical responses to visual stimuli." *Neuron* 73 (2012), pp. 159–170.

[26] Damir Dobric. "Investigation and Modelling of a Cortical Learning Algorithm in the Neocortex." PhD thesis. University of Plymouth, 2024. doi: 10.24382/5155.
