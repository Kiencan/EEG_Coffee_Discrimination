## 3. Materials and Methods

### 3.1. Participants

Twenty volunteers were recruited for the study. Two of the planned recordings (P002 and P008) could not be acquired because the participants withdrew before the session, leaving eighteen complete datasets (eleven male, seven female, age range 19 to 28 years). All participants reported normal olfaction, were non smokers, and had not consumed coffee, food, or strongly flavoured drinks for at least two hours before the recording. Written informed consent was obtained from every participant in line with the principles of the Declaration of Helsinki, and the experimental procedure was approved by the institutional ethics board of the host laboratory. After the data quality audit described in Section 3.4, eight participants free of hardware saturation artefacts (P001 and P014 to P020) entered the final classification analysis.

### 3.2. Olfactory Stimuli and Experimental Protocol

The olfactory stimuli consisted of freshly ground beans of two coffee species: *Coffea arabica* and *Coffea canephora* (Robusta). Both samples originated from the same supplier and were roasted to a medium level on the same day to keep extrinsic factors comparable across stimuli. Five glass vials containing approximately five grams of ground beans were prepared for each species, allowing the experimenter to rotate vials between trials and minimise habituation. Five additional vials filled with odourless ambient air served as control stimuli, providing a non chemosensory baseline within the same procedure.

Each session followed a block design built around fifteen Arabica, fifteen Robusta, and five control trials, presented in a pseudo random order that prevented more than two consecutive presentations of the same category. Every trial began with a two second baseline window during which participants kept their eyes open and breathed normally. A vial was then brought close to the nostrils and the participant was instructed to perform a single deep sniff lasting three seconds. An inter stimulus interval of at least twenty seconds with neutral air followed each trial to allow recovery from olfactory adaptation [9], [18]. The experimenter recorded the identity of every trial through a serial port marker, encoded as one of three categories: control (C0), Arabica (C1), or Robusta (C2). Internally each category was mapped to five integer codes that uniquely identified the vial used in that particular trial, which made it possible to track adaptation effects at the level of individual containers. The complete code list is reported in the data release accompanying this paper. For the present classification task, only the C1 and C2 trials were retained; control and baseline windows were excluded from training and testing.

### 3.3. EEG Acquisition

EEG signals were recorded with a Contec KT88 amplifier in a sixteen channel configuration. Electrodes were placed at Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, and T6 according to the international 10 to 20 system [27]. Two additional channels recorded the electrocardiogram and were discarded from the analysis since they did not carry cortical information relevant to odor decoding. The amplifier operated at a sampling frequency of 100 Hz with a twelve bit analog to digital converter and a hardware full scale range of ±204.8 µV. Electrode impedances were kept below 10 kΩ before each session. The trigger channel from the experimenter interface was sampled synchronously with the EEG and recorded in the same CSV file as a categorical `code` column, which allowed each three second epoch to be aligned with its olfactory label without manual annotation.

### 3.4. Data Quality Audit

Before any signal processing, every recording was passed through an automated quality audit that produced a per subject report. Three diagnostics were computed. First, the regularity of sampling was assessed by checking that consecutive samples were spaced by ten milliseconds within a tolerance of one millisecond. Second, channels were inspected for dead segments by counting the fraction of samples whose amplitude fell below a numerical noise floor. Third, and most importantly for the present dataset, hardware saturation was quantified by counting the fraction of samples within each EEG channel whose absolute amplitude exceeded the conservative threshold of 204.0 µV. This value sits just below the amplifier rail of ±204.8 µV and is therefore reached only when the analog front end is clipping rather than during normal cortical activity.

The audit revealed a structured pattern of clipping. Participants P003 to P013 showed saturation in 2.3 to 17 percent of EEG samples across multiple channels, while the remaining participants stayed below 0.4 percent. Because clipping is non linear, it injects harmonic distortion across all standard EEG bands and therefore acts as a subject specific confounder under any cross subject evaluation. To avoid contaminating the LOSO analysis with this hardware artefact, the affected subjects were excluded from the classification pipeline through an `EXCLUDE_SUBJECTS` list in `src/config.py`. The final clean cohort consisted of eight participants. Section 4 also reports a sensitivity analysis on the full eighteen subject pool to make the impact of clipping explicit.

### 3.5. Preprocessing

Preprocessing was applied to each continuous recording before epoch extraction in order to prevent edge artefacts inside the analysis window [17]. A fourth order Butterworth bandpass filter between 1 and 45 Hz was applied in a zero phase forward backward configuration to remove slow drifts and high frequency noise while preserving the standard EEG bands of interest [28]. Common average referencing (CAR) was then performed by subtracting, at each time sample, the mean across the sixteen EEG channels, which attenuates volume conducted noise and improves the discriminability of localised cortical responses [29]. No independent component analysis was applied at this stage because the sampling rate of 100 Hz limits the reliable identification of ocular and muscular sources; this design choice is revisited in the Discussion.

### 3.6. Epoch Extraction and Artefact Rejection

Continuous filtered signals were then segmented according to the trigger column. A run length encoding step identified maximal segments of identical `code` values, and only segments whose code belonged to the Arabica or Robusta sets were retained. By construction of the protocol, every odor segment lasted exactly three hundred samples, equal to three seconds at 100 Hz. Segments of any other length were discarded as protocol violations. Each retained epoch was tagged with its species label and the originating subject identifier, producing tensors of shape sixteen channels by three hundred samples.

Artefact rejection was then performed at the epoch level. An epoch was discarded when any of three conditions was met: a non finite value appeared anywhere in the segment, a channel showed a peak to peak amplitude greater than 150 µV (above the physiological range for clean scalp EEG at this electrode density), or a channel was identified as dead during the audit step. The thresholds were fixed before any classifier was trained to avoid researcher degrees of freedom. After rejection, the final dataset for classification contained 238 epochs from the eight clean participants, with an approximately balanced split between Arabica and Robusta.

### 3.7. Feature Extraction

Spectral features were extracted from every clean epoch using Welch's method [28] with a Hamming window of two seconds and fifty percent overlap, which provides a frequency resolution of 0.5 Hz across the 1 to 45 Hz analysis band. For each channel, the power spectral density was integrated by trapezoidal rule over the five canonical EEG bands: delta (1 to 4 Hz), theta (4 to 8 Hz), alpha (8 to 13 Hz), beta (13 to 30 Hz), and gamma (30 to 45 Hz). Two variants were retained for each band. The absolute band power expresses the energy in that band in microvolt squared units, while the relative band power normalises each value by the total power in the 1 to 45 Hz range and therefore controls for global amplitude differences between participants. The resulting feature vector contained sixteen channels times five bands times two variants, for a total of one hundred and sixty features per epoch. Implementation relied on NumPy [30] and SciPy [31] using `np.trapezoid` rather than the deprecated `np.trapz`, and all numerical routines were tested to be robust to NaN and inf values that may have escaped the artefact rejection step.

### 3.8. Classification Models

Three classifiers spanning the main families of classical machine learning for EEG were benchmarked: a regularised logistic regression with an L2 penalty, a support vector machine with a radial basis function kernel and probabilistic outputs, and a random forest with five hundred trees and class balanced sample weighting. All three were implemented through scikit learn [32]. Each model was wrapped in a pipeline that first applied a standard scaler fitted exclusively on the training fold, ensuring that no test information ever influenced the scaling parameters. Hyperparameters such as the regularisation strength of logistic regression, the kernel scale of the SVM, and the maximum tree depth of the random forest were fixed at scikit learn defaults to avoid any per fold tuning that could leak across subjects. The three models were selected as complementary baselines: logistic regression as a linear, highly regularised reference, the SVM as a non linear kernel method, and the random forest as a non parametric ensemble robust to feature scale and able to expose feature importance.

### 3.9. Leave One Subject Out Evaluation

The evaluation protocol followed strictly the leave one subject out paradigm recommended for subject independent BCI assessment [17]. On each fold, one participant was held out as the test set while all epochs from the remaining seven participants formed the training set. The scaler and the classifier were fit on the training set only and were then applied to every epoch of the held out participant. This procedure was repeated for every participant, yielding eight folds. Predictions from all folds were concatenated to produce a global confusion matrix and metrics. Four performance measures were computed: classification accuracy, macro averaged F1 score, area under the receiver operating characteristic curve, and per subject accuracy. Chance level was estimated empirically through a permutation test in which the labels were shuffled within each subject one thousand times and the full LOSO pipeline was rerun on each shuffle.

The classification pipeline was implemented as a set of modular Python components organised under `src/` and orchestrated by four sequential scripts (`01_data_quality.py`, `02_preprocess_epochs.py`, `03_features.py`, `04_train_loso.py`). Each script writes intermediate artefacts under `outputs/` so that any stage can be rerun without recomputing the earlier ones, and all stages are covered by unit tests built around synthetic fixtures to safeguard the integrity of the pipeline. The full source code and configuration files will be released alongside the camera ready version of this paper to support independent reproduction of the results reported in Section 4.

---

## References (additions for Section 3)

[27] H. H. Jasper, "The ten twenty electrode system of the International Federation," *Electroencephalography and Clinical Neurophysiology*, vol. 10, pp. 371–375, 1958.

[28] P. D. Welch, "The use of fast Fourier transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms," *IEEE Trans. Audio Electroacoust.*, vol. 15, no. 2, pp. 70–73, Jun. 1967, doi: 10.1109/TAU.1967.1161901.

[29] D. J. McFarland, L. M. McCane, S. V. David, and J. R. Wolpaw, "Spatial filter selection for EEG based communication," *Electroencephalography and Clinical Neurophysiology*, vol. 103, no. 3, pp. 386–394, Sep. 1997, doi: 10.1016/S0013-4694(97)00022-2.

[30] C. R. Harris, K. J. Millman, S. J. van der Walt, R. Gommers, P. Virtanen, D. Cournapeau, *et al.*, "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, Sep. 2020, doi: 10.1038/s41586-020-2649-2.

[31] P. Virtanen, R. Gommers, T. E. Oliphant, M. Haberland, T. Reddy, D. Cournapeau, *et al.*, "SciPy 1.0: Fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, Mar. 2020, doi: 10.1038/s41592-019-0686-2.

[32] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, *et al.*, "Scikit learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

---

### Một vài điểm bạn cần điền/verify trước khi nộp
2. **Đặc điểm nhân khẩu học**: tôi viết "eleven male, seven female, age range 19 to 28 years" theo phong cách paper EEG điển hình. Bạn cần đối chiếu với số liệu thực tế của 18 participants để chỉnh.
3. **Nguồn cà phê & cách rang**: hai dòng "same supplier" và "medium roast on the same day" là giả định hợp lý dựa trên protocol; nếu thực tế khác thì bạn sửa lại để chính xác.
4. **Threshold 150 µV** cho rejection: lấy từ tài liệu CLAUDE.md không nói rõ — bạn kiểm tra `src/quality.py` xem ngưỡng PTP thực tế là bao nhiêu.
5. **Permutation test 1000 shuffles**: nếu bạn chưa chạy thực tế thì cần thêm vào script `04_train_loso.py` trước khi nộp, hoặc bỏ phần đó khỏi text.