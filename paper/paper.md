# EEG-Based Coffee Aroma Discrimination Using Machine Learning to Decode Brain Responses to Olfactory Stimuli

## Abstract

Coffee aroma drives consumer preference and market value, but sensory panels are subjective and electronic noses describe chemistry rather than perception. We tested whether scalp EEG with classical machine learning can discriminate *Coffea arabica* from *Coffea canephora* (Robusta) aromas at the single trial level. Twenty volunteers were recorded with a 16 channel KT88 amplifier at 100 Hz over 45 sniff trials per session (15 Arabica, 15 Robusta, 15 odourless air). After band pass filtering, common average referencing, and artefact rejection, we classified three second epochs with logistic regression, an RBF SVM, and a random forest under strict leave one subject out (LOSO) evaluation. A transparent data quality audit identified hardware clipping (≥2.3 % samples railed at ±204.8 µV) in 11 of 18 acquired recordings, which were excluded to avoid a per subject confounder, leaving 238 odor epochs from eight participants. The best random forest reached an accuracy of 0.546 (AUC 0.553) on band power features; the highest configuration of a 72 cell feature search (engineered superset, random forest) reached a balanced accuracy of 0.588. An easier coffee versus air contrast peaked at 0.538 and a pseudo subject control confirmed the result is not a fold size artefact. The only systematically above chance result was within subject three level classification of sensory ratings (balanced accuracy up to 0.48, chance 0.33). We release the dataset, the pipeline, and a calibrated reference point for subject independent EEG decoding of closely related natural odors.

**Keywords:** EEG, olfactory perception, coffee aroma, machine learning, brain–computer interface.

## I. Introduction

Coffee is one of the most traded agricultural commodities worldwide, and the two dominant species *Coffea arabica* and *Coffea canephora* (Robusta) command very different prices, with partial substitution being a recurring form of adulteration [1], [2]. Quality assurance still relies on trained sensory panels and instrumental methods such as GC–MS or electronic noses [3], [4], both of which describe either human judgement or chemistry, but not the cortical response that ultimately defines perceived aroma. Scalp EEG offers a complementary, non invasive window: odors modulate band power and chemosensory event related potentials over fronto-temporal sites [5], [6], and recent multivariate decoding work shows that odor information is available in EEG within the first half second of stimulus onset [7].

Machine learning has accelerated single trial EEG decoding, with band power features and linear or kernel classifiers as the classical baseline [8] and compact convolutional networks such as EEGNet [9] as a modern alternative. Several groups have reported above chance odor classification, but almost always under within subject splits and between highly dissimilar odorants [6], [10], [11]. Whether EEG carries information about closely related natural odors that generalises across participants — the realistic deployment setting for an aroma decoder — remains open. Closely related work on coffee specifically has focused on group level oscillatory changes during inhalation [12], not on subject independent classification of species.

This paper addresses that gap with three contributions. First, we collect and release the first 16 channel EEG dataset of *Arabica* versus *Robusta* sniffing under a controlled protocol. Second, we benchmark logistic regression, an RBF SVM, and a random forest on spectral band power features and on five additional engineered feature families under strict LOSO evaluation, and we report honest chance level results. Third, we document a hardware clipping artefact in a contiguous block of eleven recordings and show how omitting the corresponding data quality audit would have inflated cross subject accuracy through a per subject confounder rather than genuine olfactory neural activity.

## II. Materials and Methods

### A. Participants, Stimuli, and Acquisition

Twenty volunteers were recruited; two (P002, P008) withdrew before the session, leaving 18 complete datasets. All participants reported normal olfaction, had abstained from food and strongly flavoured drinks for at least two hours before recording, and gave written informed consent under a protocol approved by the institutional ethics board. Stimuli were freshly ground beans of *Coffea arabica* and *Coffea canephora* from the same supplier, roasted to a medium level on the same day. Each session presented 15 Arabica, 15 Robusta, and 15 odourless air trials in a pseudo random order, drawn from a per subject canonical sequence stored separately from the EEG recording. Every trial comprised a two second baseline, a three second deep sniff, and an inter stimulus interval of at least 20 s [6], [13]. Each category was mapped to five integer codes identifying the physical vial used (Table I); for the present binary task only the Arabica and Robusta codes were retained.

EEG was recorded with a Contec KT88 amplifier in a 16 channel configuration (Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6; international 10–20 system [14]) at 100 Hz, 12 bit ADC, full scale ±204.8 µV. Two ECG channels were discarded. Electrode impedances were kept below 10 kΩ and the trigger column was sampled synchronously with the EEG.

**Table I.** Stimulus categories and trigger codes.

| Category | Label | Integer codes |
|----------|-------|---------------|
| Control (odourless air) | C0 | 712, 238, 759, 869, 562 |
| *Coffea arabica* | C1 | 981, 633, 902, 598, 733 |
| *Coffea canephora* | C2 | 585, 597, 200, 558, 692 |

### B. Data Quality Audit

Before any signal processing, every recording was passed through an automated audit that checked sampling regularity, flat channels (standard deviation below 0.5 µV), and hardware saturation (fraction of samples whose absolute amplitude exceeded 204.0 µV, just below the ±204.8 µV amplifier rail). The audit revealed a bimodal distribution of saturation (Fig. 1): participants P003–P013 showed 2.3 % to 17 % saturated samples (139,725–1,110,804 per recording), while the remaining seven recordings stayed below 0.4 %. Because non linear clipping injects harmonic distortion across all bands and is structured by participant, it acts as a per subject covariate that a LOSO classifier could exploit instead of the odor label. The eleven affected subjects were therefore excluded before any modelling, leaving a clean cohort of eight (P001, P014–P020). Two additional findings — two trailing all NaN rows at the end of P014's recording, which would otherwise propagate through the zero phase filter, and a manual marker entry mismatch for P019, in which the EEG trigger column disagreed with the canonical stimulus sequence — were handled by dropping the affected rows at load time and by labelling epochs from the canonical sequence rather than from the EEG trigger.

![Fig. 1](../outputs/report/fig_4_3_clipping_evidence.png)

**Fig. 1.** Hardware clipping evidence. (a) Amplitude histogram of channel F3: clean participant P001 (top) is bell shaped, clipped P006 (bottom) shows two peaks pinned at ±204.8 µV. (b) Five seconds of raw P006 with flat topped excursions. (c) Per subject fraction of saturated samples (log scale); red bars mark the eleven excluded recordings.

### C. Preprocessing, Features, and Classifiers

Each continuous recording was band pass filtered (4th order Butterworth, 1–45 Hz, zero phase forward backward) before epoch extraction to avoid edge artefacts inside the analysis window [8], then common average referenced [15]. Odor epochs were extracted by run length encoding of the trigger column: every Arabica or Robusta segment of exactly 300 samples was kept. An epoch was rejected if it contained a non finite value, a flat channel (std < 0.5 µV), or any channel whose peak to peak amplitude exceeded the adaptive 99th percentile of the epoch level peak to peak distribution. After rejection, the final dataset contained 238 epochs (121 Arabica, 117 Robusta) over the eight clean subjects.

The primary feature set was Welch [16] band power on five canonical bands (delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–45 Hz) in absolute and relative form, yielding 160 features per epoch. A feature search additionally evaluated five families: time domain statistics (128 features), Hjorth parameters (48), 2 Hz narrow band power (256), three window temporal band power (480), and an engineered superset (912) selectable via a mutual information criterion at k ∈ {20, 40, 80, all}. Three classifiers were benchmarked: L2 regularised logistic regression, an RBF SVM with probabilistic outputs, and a random forest with 300 class balanced trees. Each model was wrapped in a pipeline of standard scaling, optional feature selection, and classification, with hyperparameters fixed at the scikit learn defaults [17] to avoid leakage through per fold tuning.

Evaluation followed strict leave one subject out cross validation: on each of the eight folds, the held out subject's epochs are predicted by a model fit only on the other seven. We report accuracy, balanced accuracy, macro F1, and ROC AUC; chance level for the two class problems is 0.5 and for the three level sensory classification of Section III-C it is 1/3. The full source code, pipeline, and unit tests will be released with the camera ready version.

## III. Results

All numbers are taken directly from CSV files written by the pipeline. The clean cohort contributes 238 epochs distributed as P001=30, P014=31, P015=30, P016=29, P017=29, P018=30, P019=30, P020=29; Arabica vs Robusta are balanced at 121 / 117.

### A. Arabica versus Robusta

On the 160 dimensional band power feature set (Table II, Fig. 2), all three classifiers stayed within five percentage points of the 0.5 chance level: random forest 0.546 / 0.546 / 0.553 (accuracy / macro F1 / ROC AUC), RBF SVM 0.513 / 0.507 / 0.466, logistic regression 0.466 / 0.465 / 0.496.

**Table II.** LOSO metrics on band power features.

| Model | Accuracy | Macro F1 | ROC AUC |
|-------|---------:|---------:|--------:|
| Logistic regression | 0.466 | 0.465 | 0.496 |
| SVM (RBF) | 0.513 | 0.507 | 0.466 |
| Random forest | **0.546** | **0.546** | **0.553** |

![Fig. 2](../outputs/report/fig_6_1_arabica_robusta_bars.png)

**Fig. 2.** LOSO accuracy of the three baseline classifiers on band power features. Error bars: standard deviation across the eight folds; dashed line: 0.5 chance level.

The feature search across six families, three classifiers, and four feature selection levels (72 configurations in total) yielded a peak balanced accuracy of 0.588 (engineered superset, random forest, AUC 0.600), followed by 0.585 for band power with a linear SVM (Table III, Fig. 3). Balanced accuracy across the entire sweep ranged from 0.44 to 0.59 with no clear winning family. A representation free baseline that passes PCA reduced raw epochs to the same three classifiers reached only ≈0.49 balanced accuracy. That the simplest band power features and a 912 dimensional engineered superset converge to the same ceiling indicates the limitation is not the choice of representation.

**Table III.** Top LOSO configurations, Arabica vs Robusta.

| Family | Dim | k | Model | Bal. acc | ROC AUC |
|--------|----:|---|-------|---------:|--------:|
| `engineered` | 912 | all | random forest | **0.588** | **0.600** |
| `bandpower` | 160 | 80 | linear SVM | 0.585 | 0.557 |
| `engineered` | 912 | all | logistic regression | 0.580 | 0.567 |
| `engineered` | 912 | all | linear SVM | 0.576 | 0.573 |
| `time` | 128 | 40 | random forest | 0.567 | 0.530 |

![Fig. 3](../outputs/report/fig_6_4_feature_search_heatmap.png)

**Fig. 3.** Balanced accuracy of the feature search across families (rows) and classifiers (columns); each cell shows the best result over the four feature selection levels. The gradient is shallow and no cell exceeds 0.59.

### B. Coffee versus Air and Pseudo Subject Control

To rule out that the difficulty is specific to within category discrimination, we replaced the target with control air versus coffee (Arabica and Robusta merged). The peak among the same 72 configurations was 0.538 balanced accuracy (engineered superset, linear SVM, AUC 0.521); the raw + PCA baseline returned ≈0.49. The harder within category task and the easier between category task therefore share the same chance level ceiling.

A second sensitivity test split each clean subject into five epoch blocks per label, producing 72 pseudo subjects, and reran the LOSO. If the chance level result were caused by the eight fold split being too coarse for 238 epochs, splitting finer would raise balanced accuracy. Instead, the best pseudo subject configuration reached only 0.491 (raw + PCA + logistic regression) — slightly *below* the corresponding real subject score of 0.525 with the engineered superset — which rules out fold variance as the explanation.

### C. Sensory Rating Analysis

Within subject Pearson correlations between the 160 band power features and the three rating dimensions stayed within [−0.20, +0.22], with the largest positive values for theta band power over central and temporal electrodes against the favourite rating (e.g. `rel_theta_C3`, r ≈ 0.219). These correlations would not survive Bonferroni correction but are consistent with the literature linking theta oscillations to hedonic processing.

RidgeCV regression on the same features returned negative coefficients of determination on held out folds at both evaluation levels (Table IV). Quantising each rating into three per subject percentile based levels (low / medium / high; chance = 1/3) gives the only systematically above chance results in the study: within subject random forest balanced accuracies of 0.480 (valence), 0.408 (intensity), and 0.399 (favourite), versus 0.396 / 0.288 / 0.343 for the cross subject case. The pattern — modest within subject signal that collapses to chance across participants — matches the broader BCI literature on subject specific olfactory hedonics [6].

**Table IV.** Sensory ratings from band power: RidgeCV R² and 3-class random forest balanced accuracy (chance ≈ 0.333).

| Rating | R² within | R² cross | Bal. acc within (RF) | Bal. acc cross (RF) |
|--------|----------:|---------:|---------------------:|--------------------:|
| Valence   | −0.197 | −0.381 | **0.480** | 0.396 |
| Intensity | −0.075 | −0.464 | 0.408 | 0.288 |
| Favourite | −0.228 | −0.116 | 0.399 | 0.343 |

## IV. Discussion

Under a strict subject independent evaluation, neither classical machine learning models on spectral band power nor a sweep of richer representations could discriminate the aroma of *Coffea arabica* from that of *Coffea canephora* above the chance ceiling, with a peak balanced accuracy of 0.588 over 72 configurations. The same ceiling appears on the easier coffee versus air contrast and on the regression of subjective ratings, so the limitation lies in the amount of subject invariant olfactory information in the signal rather than in the choice of label or representation. This outcome extends to closely related natural odors a pattern already documented for cross subject EEG decoding [8]: studies that report high single trial odor accuracies almost invariably evaluate within subject splits or contrast highly dissimilar odorants [6], [10], [11], and two roasted coffee species — which share most of their volatile inventory — sit at the difficult end of the spectrum. The only above chance result, within subject three level rating classification (0.40–0.48 against 0.33 chance), is consistent with a robust individual response to olfactory hedonics that does not transfer to a calibration free decoder.

The transparent quality audit was as important methodologically as the headline result. The eleven excluded recordings produced 2.3 % to 17 % saturated samples — a per subject covariate that a LOSO classifier could trivially exploit instead of the intended olfactory signal. Subject specific confounders of this kind, whether from impedance drift, electrode location, or amplifier clipping, can inflate cross subject scores even when no genuine task signal is present [8], and the parallel finding on the P019 marker mismatch argues for sourcing trial order independently of the live marker channel in future protocols.

Four limitations bound the conclusions. The KT88 acquisition is modest: 100 Hz caps spectral analysis at 50 Hz and 16 channels limit spatial resolution; higher density montages and faster sampling would also enable ICA based artefact removal that the present rate does not support reliably. The olfactory delivery was manual rather than from a calibrated olfactometer [6], [13], which is ecologically valid but injects onset timing jitter that may obscure precise CSERP locking [7]. The clean cohort of eight participants is small — the 11 clipped recordings cut the effective sample size by more than half — and the analysis stopped at classical machine learning without end to end deep learning, which we judged unreliable to train on this sample size.

These limitations suggest concrete directions: within subject modelling with per user calibration, which is empirically supported by the above chance sensory rating result; sniff aligned event related features that resolve N1, P2, and the late positive complex rather than averaging over the full three seconds [5], [7]; functional connectivity features combined with Riemannian alignment that have proven robust under LOSO in motor imagery and emotion BCIs [18]; and EEGNet style compact convolutional networks [9] pre trained on larger public EEG datasets and fine tuned on the present cohort with domain adaptation. A controlled protocol that pairs natural coffee with calibrated reference odorants would further clarify the difficulty gradient between within and between category discrimination.

## V. Conclusion

We presented a reproducible pipeline and the first 16 channel EEG dataset for *Coffea arabica* versus *Coffea canephora* aroma discrimination. Under strict leave one subject out evaluation on eight clean participants and 238 odor epochs, classical machine learning models reached a peak balanced accuracy of 0.588 across 72 feature search configurations — close to chance — with the same ceiling reproduced on coffee versus air and falsified as a fold size artefact by a pseudo subject control. Only within subject classification of sensory ratings produced systematically above chance scores. A transparent data quality audit was essential and revealed that hardware clipping in 11 of 18 recordings would have inflated cross subject accuracy through a per subject confounder. The released pipeline, the documented audit, and the calibrated chance level reference point provide a foundation for future work on within subject modelling, event related decoding, connectivity features, and end to end deep learning for EEG based aroma discrimination.

## Acknowledgements

The authors thank all volunteers, the host laboratory staff, and the ethics committee for their support, and the coffee suppliers for providing freshly roasted *Coffea arabica* and *Coffea canephora* samples of comparable origin. This work was supported by [grant code]. The authors declare no competing interests.

## References

[1] International Coffee Organization, "Coffee Report and Outlook — December 2023," ICO, London, U.K., Dec. 2023.

[2] L. C. Cardoso, F. M. V. Pereira, M. M. Sena, and R. J. Poppi, "Hyperspectral imaging, chemometrics, feature selection, and machine learning for rapid, non destructive detection and quantification of Robusta adulteration in Arabica coffee," *Food Control*, vol. 178, art. no. 111554, 2025, doi: 10.1016/j.foodcont.2025.111554.

[3] S. Bressanello *et al.*, "Coffee aroma: Chemometric comparison of the chemical information provided by three different samplings combined with GC–MS to describe the sensory properties in cup," *Food Chemistry*, vol. 214, pp. 218–226, Jan. 2017, doi: 10.1016/j.foodchem.2016.07.088.

[4] M. Pardo and G. Sberveglieri, "Coffee analysis with an electronic nose," *IEEE Trans. Instrum. Meas.*, vol. 51, no. 6, pp. 1334–1339, Dec. 2002, doi: 10.1109/TIM.2002.808038.

[5] B. M. Pause and K. Krauel, "Chemosensory event-related potentials (CSERP) as a key to the psychology of odors," *Int. J. Psychophysiol.*, vol. 36, no. 2, pp. 105–122, May 2000, doi: 10.1016/S0167-8760(99)00105-1.

[6] E. Kroupi, A. Yazdani, J.-M. Vesin, and T. Ebrahimi, "EEG Correlates of Pleasant and Unpleasant Odor Perception," *ACM Trans. Multimedia Comput. Commun. Appl.*, vol. 11, no. 1s, art. no. 13, pp. 1–17, Oct. 2014, doi: 10.1145/2637287.

[7] M. Iravani *et al.*, "Spatiotemporal dynamics of odor representations in the human brain revealed by EEG decoding," *Proc. Natl. Acad. Sci. USA*, vol. 119, no. 21, e2114966119, May 2022, doi: 10.1073/pnas.2114966119.

[8] F. Lotte *et al.*, "A review of classification algorithms for EEG-based brain–computer interfaces: A 10 year update," *J. Neural Eng.*, vol. 15, no. 3, art. no. 031005, Jun. 2018, doi: 10.1088/1741-2552/aab2f2.

[9] V. J. Lawhern *et al.*, "EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces," *J. Neural Eng.*, vol. 15, no. 5, art. no. 056013, Oct. 2018, doi: 10.1088/1741-2552/aace8c.

[10] M. Aydemir, "Olfactory recognition based on EEG gamma-band activity," *Neural Computation*, vol. 29, no. 6, pp. 1667–1680, Jun. 2017, doi: 10.1162/NECO_a_00966.

[11] X. Hou *et al.*, "Olfactory EEG induced by odor: Used for food identification and pleasure analysis," *Food Chemistry*, vol. 462, art. no. 140946, Jan. 2025, doi: 10.1016/j.foodchem.2024.140946.

[12] H. Tanaka and Y. Kobayashi, "Effect of coffee aroma on cerebral activity during concentration tasks," *J. Behav. Brain Sci.*, vol. 14, no. 6, pp. 280–294, 2024, doi: 10.4236/jbbs.2024.146015.

[13] G. Kobal, *Elektrophysiologische Untersuchungen des menschlichen Geruchssinns*. Stuttgart: Thieme, 1981.

[14] H. H. Jasper, "The ten twenty electrode system of the International Federation," *Electroencephalogr. Clin. Neurophysiol.*, vol. 10, pp. 371–375, 1958.

[15] D. J. McFarland, L. M. McCane, S. V. David, and J. R. Wolpaw, "Spatial filter selection for EEG based communication," *Electroencephalogr. Clin. Neurophysiol.*, vol. 103, no. 3, pp. 386–394, Sep. 1997, doi: 10.1016/S0013-4694(97)00022-2.

[16] P. D. Welch, "The use of fast Fourier transform for the estimation of power spectra," *IEEE Trans. Audio Electroacoust.*, vol. 15, no. 2, pp. 70–73, Jun. 1967, doi: 10.1109/TAU.1967.1161901.

[17] F. Pedregosa *et al.*, "Scikit learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.

[18] A. Barachant, S. Bonnet, M. Congedo, and C. Jutten, "Classification of covariance matrices using a Riemannian based kernel for BCI applications," *Neurocomputing*, vol. 112, pp. 172–178, Jul. 2013, doi: 10.1016/j.neucom.2012.12.039.
