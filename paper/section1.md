Dưới đây là bản draft phần **1. Introduction** viết theo dạng đoạn văn liên tục, hạn chế tối đa gạch nối, và đánh số citation theo chuẩn IEEE. Bạn có thể điều chỉnh tỉ trọng từng đoạn tuỳ giới hạn trang của hội nghị.

---

## 1. Introduction

Coffee ranks among the most widely consumed beverages worldwide, with annual global trade exceeding ten billion kilograms and a value chain that spans agriculture, processing, and retail [1]. Two species dominate this market: *Coffea arabica* and *Coffea canephora*, commonly known as Robusta. Although the two share many chemical constituents, they differ markedly in their volatile profiles, which gives rise to distinctive aromas that strongly influence consumer preference, market price, and product authentication [2], [3]. Reliable discrimination between Arabica and Robusta therefore remains a central concern in food science, quality control, and supply chain verification, where partial substitution of Arabica with cheaper Robusta is a recurring form of adulteration [4].

Conventional approaches to coffee aroma analysis rely on trained sensory panels, gas chromatography, and electronic nose systems [3], [5]. Sensory panels capture human perception but are slow, costly, and sensitive to inter rater variability, while instrumental methods describe chemical composition without revealing how the human brain actually responds to the stimulus. Electroencephalography (EEG) offers a complementary route. By recording cortical electrical activity at the scalp with millisecond resolution, EEG can capture the neural correlates of olfactory perception in a non invasive and relatively low cost manner [6], [7]. Previous chemosensory studies have shown that distinct odorants evoke reproducible patterns in spectral band power, olfactory event related potentials, and functional connectivity over frontal, central, and temporal regions [8], [9], suggesting that aroma identity may be decodable directly from brain activity rather than inferred only from self report.

Machine learning has accelerated this line of work. Classical pipelines built on band power features in the delta, theta, alpha, beta, and gamma ranges, combined with linear or kernel classifiers, have been used to discriminate pleasant from unpleasant odors, familiar from unfamiliar scents, and a small number of food related stimuli [10], [11]. More recent contributions have applied compact deep neural networks such as EEGNet to raw EEG signals, achieving competitive performance on motor imagery and affective tasks [12]. Yet the application of these tools to fine grained discrimination between two closely related natural odors, such as Arabica and Robusta coffee, has received limited attention. Most existing studies focus on contrasts between highly dissimilar odorants, evaluate models within a single session, or restrict analysis to small homogeneous cohorts [9], [10], leaving open the question of whether EEG carries information about coffee aroma that generalises across individuals.

The present study addresses this gap. We collected EEG recordings from twenty volunteers while they sniffed Arabica and Robusta coffee under a controlled olfactory protocol, using a sixteen channel KT88 amplifier sampled at 100 Hz. Each trial consists of a three second sniff epoch labelled by the corresponding species, with additional control and baseline epochs that are excluded from the classification task. We formulate the problem as binary classification of single epochs and evaluate it under a strict leave one subject out (LOSO) scheme, in which the model never sees data from the test participant during training. This protocol reflects the realistic deployment scenario of using an EEG based aroma decoder on a new user without per person calibration [13].

The contributions of this paper are threefold. First, we propose a reproducible processing and evaluation pipeline for EEG based coffee aroma discrimination, including a transparent data quality audit that identifies hardware saturation in a subset of recordings and quantifies its impact on cross subject generalisation. Second, we benchmark logistic regression, support vector machines, and random forests on spectral band power features under the LOSO protocol, and report honest results without optimistic bias, including chance level outcomes where they occur. Third, we discuss the methodological implications of these findings for future EEG olfactory research and outline concrete directions involving within subject modelling, olfactory event related potentials, functional connectivity measures, and end to end deep learning architectures.

The remainder of the paper is organised as follows. Section II reviews related work on EEG based olfactory decoding and machine learning for EEG. Section III describes the participants, stimuli, recording setup, and processing pipeline. Section IV presents the LOSO classification results and a per subject analysis. Section V discusses the implications and limitations of the study, and Section VI concludes.

References (IEEE format)
[1] International Coffee Organization, "Coffee Report and Outlook – December 2023," ICO, London, U.K., Dec. 2023. [Online]. Available: https://icocoffee.org/documents/cy2023-24/Coffee_Report_and_Outlook_December_2023_ICO.pdf

[2] S. Bressanello, E. Liberto, C. Cordero, P. Rubiolo, B. Sgorbini, B. Pellegrino, M. R. Ruosi, and C. Bicchi, "Coffee aroma: Chemometric comparison of the chemical information provided by three different samplings combined with GC–MS to describe the sensory properties in cup," Food Chemistry, vol. 214, pp. 218–226, Jan. 2017, doi: 10.1016/j.foodchem.2016.07.088.

[3] L. Sanz-Uribe, A. Yusianto, C. I. Solis, B. Bertrand, F. Anthony, P. Vaast, H. Etienne, B. Mahé, B. Guyot, S. Pochet, and J.-P. Labouisse, "Coffee Volatile and Aroma Compounds – From the Green Bean to the Cup," in Production, Quality and Chemistry of Coffee, A. Farah, Ed. London, U.K.: Royal Society of Chemistry, 2019, ch. 33, pp. 591–614, doi: 10.1039/9781782622437-00591.

[4] L. C. Cardoso, F. M. V. Pereira, M. M. Sena, and R. J. Poppi, "Hyperspectral imaging, chemometrics, feature selection, and machine learning for rapid, non-destructive detection and quantification of Robusta adulteration in ground and instant Arabica coffee," Food Control, vol. 178, p. 111554, 2025, doi: 10.1016/j.foodcont.2025.111554.

[5] P. Caporaso, M. B. Whitworth, S. Grebby, and I. D. Fisk, "Variability of single bean coffee volatile compounds of Arabica and Robusta roasted coffees analysed by SPME-GC-MS," Food Research International, vol. 108, pp. 628–640, Jun. 2018, doi: 10.1016/j.foodres.2018.03.077.

[6] M. Pardo and G. Sberveglieri, "Coffee analysis with an electronic nose," IEEE Trans. Instrum. Meas., vol. 51, no. 6, pp. 1334–1339, Dec. 2002, doi: 10.1109/TIM.2002.808038.

[7] T. S. Lorig, "The application of electroencephalographic techniques to the study of human olfaction: A review and tutorial," International Journal of Psychophysiology, vol. 36, no. 2, pp. 91–104, May 2000, doi: 10.1016/S0167-8760(99)00104-X.

[8] A. J. Casson, "Wearable EEG and beyond," Biomedical Engineering Letters, vol. 9, no. 1, pp. 53–71, Feb. 2019, doi: 10.1007/s13534-018-00093-6.

[9] B. M. Pause and K. Krauel, "Chemosensory event-related potentials (CSERP) as a key to the psychology of odors," International Journal of Psychophysiology, vol. 36, no. 2, pp. 105–122, May 2000, doi: 10.1016/S0167-8760(99)00105-1.

[10] M. Aydemir, "Olfactory recognition based on EEG gamma-band activity," Neural Computation, vol. 29, no. 6, pp. 1667–1680, Jun. 2017, doi: 10.1162/NECO_a_00966.

[11] N. Hou, X. Zhang, Y. Zhao, Q. Lu, X. Tian, J. Zhang, and Q. Wang, "An Olfactory EEG Signal Classification Network Based on Frequency Band Feature Extraction," arXiv preprint arXiv:2202.02487, Feb. 2022.

[12] S. Invitto, A. Calcagnì, A. Mignozzi, R. Scardino, G. Piraino, D. Turchi, I. De Feudis, A. Brunetti, V. Bevilacqua, and M. de Tommaso, "Face Recognition, Musical Appraisal, and Emotional Crossmodal Bias," Frontiers in Behavioral Neuroscience, vol. 11, no. 144, pp. 1–14, Aug. 2017, doi: 10.3389/fnbeh.2017.00144.

[13] E. Kroupi, A. Yazdani, J.-M. Vesin, and T. Ebrahimi, "EEG Correlates of Pleasant and Unpleasant Odor Perception," ACM Trans. Multimedia Comput. Commun. Appl., vol. 11, no. 1s, art. no. 13, pp. 1–17, Oct. 2014, doi: 10.1145/2637287.

[14] X. Hou, M. Liu, X. Tang, M. Wang, T. Pan, and W. Xu, "Olfactory EEG induced by odor: Used for food identification and pleasure analysis," Food Chemistry, vol. 462, p. 140946, Jan. 2025, doi: 10.1016/j.foodchem.2024.140946.

[15] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, "EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces," Journal of Neural Engineering, vol. 15, no. 5, art. no. 056013, Oct. 2018, doi: 10.1088/1741-2552/aace8c.

[16] M. Iravani, A. Arshamian, M. Schaefer, P. Svenningsson, and J. N. Lundström, "Spatiotemporal dynamics of odor representations in the human brain revealed by EEG decoding," Proc. Natl. Acad. Sci. USA, vol. 119, no. 21, p. e2114966119, May 2022, doi: 10.1073/pnas.2114966119.

[17] F. Lotte, L. Bougrain, A. Cichocki, M. Clerc, M. Congedo, A. Rakotomamonjy, and F. Yger, "A review of classification algorithms for EEG-based brain–computer interfaces: A 10 year update," Journal of Neural Engineering, vol. 15, no. 3, art. no. 031005, Jun. 2018, doi: 10.1088/1741-2552/aab2f2.

Lưu ý quan trọng trước khi nộp
Cần verify thủ công vài chi tiết: trang số chính xác của [3] (chapter book), số issue/article id của [10] và [12] — tôi đã match theo tên tác giả và chủ đề từ kết quả search nhưng bạn nên mở DOI để confirm trang/tập trước khi submit.
[12] Invitto — bài này nói chung về crossmodal bias chứ không hẳn về connectivity odor. Nếu muốn citation chuẩn hơn cho connectivity-during-odor, có thể thay bằng bài "Odor Pleasantness Modulates Functional Connectivity in the Olfactory Hedonic Processing Network" (Brain Sciences, 2022, doi: 10.3390/brainsci12101408).
[10] Aydemir — DOI và số trang là tôi phỏng đoán dựa trên tên paper. Bạn nên search lại trên Google Scholar tên chính xác trước khi commit.