# FMVPatchTST: Variable-Modality Aware Patch Transformer Network for Multivariate Non-stationary Time Series Prediction + Scalable

Preface
In the task of multivariate time series prediction, raw sequences often exhibit characteristics such as non-stationarity, strong noise, multivariate coupling, and multiscale changes. Traditional predictive models typically directly input multivariate sequences into models like Transformer, LSTM, TCN, or PatchTST for modeling. Although they can learn some temporal dependencies, for complex non-stationary sequences, the trend component, periodic component, perturbation component, and noise component in the original signal often overlap with each other, leading to difficulties in fully capturing dynamic features at different frequency scales.

Multivariate Variational Mode Decomposition (MVMD) provides an effective approach to this issue. It decomposes a multivariate sequence into multiple modes with different frequency characteristics, thereby breaking down the complex sequence into low-frequency trends, mid-frequency fluctuations, high-frequency perturbations, and residual components. We propose FMVPatchTST: Frequency-aware Mode-Variable PatchTST Multivariate Prediction Network. This model is designed for causal MVMD rolling decomposition of multivariate modal tensors, focusing on the "variable—modal—time" three-dimensional structure. By employing frequency-aware Patch embeddings, frequency prototype alignment, mode attention, variable attention, and goal-guided fusion, it achieves structured prediction for multivariate non-stationary sequences. (Provides explanatory framework analysis)

We also provide detailed materials, tutorial documents, and video explanations, including how to replace your own dataset, parameter tuning guides, etc. The code is commented line by line, with detailed introductions to parameters:
● Datasets: Electricity datasets, wind power dataset from a wind farm, wind speed datasets, etc.
1. Innovation Model Introduction
1.1 Overall Process of the Model
The core idea of FMVPatchTST is:
- The multivariate sequence after causal MVMD rolling decomposition is represented as a "variable—modality—time" three-dimensional tensor, and through frequency-aware patch embedding, modality alignment, temporal patch encoding, modality attention, variable attention, and target guidance fusion, it achieves structured prediction for non-stationary multivariate sequences.
For the original multivariate time series:
- After causal MVMD rolling decomposition, the data is no longer a regular two-dimensional time series but becomes a three-dimensional tensor with clear structural meaning:
  -
  - C represents the number of variables, K represents the number of modalities, L represents the length of the historical window.
  - If simply concatenating (C * K) modalities into ordinary channels and then feeding them into PatchTST, although it can complete the prediction, it will destroy the semantics of variables and modalities, and also cannot explicitly model the hierarchical relationship between different variables, different modalities, and different time segments.
The overall prediction process of FMVPatchTST is as follows:
- Original multivariate time series X -> causal MVMD rolling decomposition -> variable-modality-time three-dimensional tensor U ∈ R^{B×C×M×L}
- Frequency-aware patch embedding -> frequency prototype alignment / modality alignment
Temporal PatchTST Encoder->Mode Attention + Variable Attention
Target-guided Fusion -> Prediction Head-> Future Sequence Prediction
The focus of this model is not merely a simple combination of "MVMD + PatchTST", but rather, it designs a specialized deep modeling mechanism for the structural features after MVMD rolling decomposition.
1.2 Model Innovations
Innovation Point 1: Variable-Modality-Time Three-Dimensional Structure Modeling
Unlike traditional methods that directly flatten the causal MVMD rolling decomposition results into a regular multi-channel sequence, FMVPatchTST retains a three-dimensional structure and explicitly models variables, modalities, and time dimensions:
This design avoids the loss of variable and modal semantics, which is the core difference between this model and the common FMVMD-PatchTST concatenated method.
Innovation Point 2: Frequency-Aware Patch Embedding
FMVPatchTST introduces variable, modal, frequency, and position embeddings in the patch embedding module. This module enables the model to clearly identify the variable source, modal source, and frequency attributes of tokens, which helps to enhance the structural expressiveness of data after causal MVMD rolling decomposition. To address the issue of potential drift in different window modal number in causal rolling MVMD, FMVPatchTST introduces a frequency prototype alignment mechanism. This mechanism maps the modalities in different windows to a stable frequency prototype space, enhancing the consistency of modalities under rolling decomposition scenarios.
Innovation Point 3: Separation of Modality Attention and Variable Attention Modeling
FMVPatchTST designs modality attention and variable attention separately. Modality attention is used to model the dependencies between different modalities within the same variable, while variable attention is employed to model the dependencies between different variables within the same modality scale. This decoupled modeling approach is more in line with the data structure of causal MVMD rolling decomposition than direct global attention.
2. Electric Power Transformer Data Set Introduction
2.1 Import Data
2.2 Data Causal Rolling MVMD Decomposition
 (1)Decomposition Visualization
 (2)Causal Rolling MVMD Decomposition
Traditional decomposition prediction methods often involve first performing a full sequence decomposition before splitting the training samples. This approach carries the risk of future information leakage. FMVPatchTST adopts a causal rolling decomposition strategy, which means for any prediction moment (t), only using the historical window, the sequence is divided into fixed-length sample windows to provide the basis for decomposition.
This treatment ensures that the model input meets strict temporal causality.
2.3 Data Set Production and Preprocessing
Details are provided in the provided documentation!
3. Highly Innovative Prediction Model Based on FMVPatchTST
3.1 Define the FMVPatchTST prediction network model
3.2 Set parameters and train the model
50 epochs, MSE is 0.004928, the prediction effect of FMVPatchTST is significant, causal \tMVMD rolling \tdecomposition breaks complex sequences into different frequency modality, \tPatchTST \tfurther models the temporal dependence within the modalities, \tthrough frequency \tembedding \tand frequency prototype alignment, enhances the causal \tMVMD rolling \tdecomposition\tunderlying modal consistency, \tMode Attention \tmodels cross-modal relations, \tVariable Attention \tmodels cross-variable relations, thereby achieving more accurate, stable, and interpretable transformer oil temperature prediction.
4 Results Visualization and Model Evaluation
4.1 Prediction Results Visualization
4.2 Model Evaluation
FMVPatchTST is not a simple "MVMD + PatchTST", but rather a structured predictive network for multivariate modalities tensor after causal rolling \tMVMD rolling \tdecomposition. It explicitly models complex relationships between variables, modalities, and time through frequency-aware Patch Embedding, frequency prototype alignment, mode attention, variable attention, and target-guided fusion, providing a deep learning framework with structural priors and interpretability for multivariate non-stationary time series prediction. It is not only applicable to transformer oil temperature prediction but can also be extended to tasks such as photovoltaic power prediction, power load prediction, equipment status prediction, wind power prediction, etc.
1. From the Mode Attention heatmap, it is evident that the attention distribution of each frequency prototype exhibits a clear diagonal dominance characteristic, indicating that the model can maintain relative independence among different frequency modalities. Meanwhile, there is still some degree of interaction between adjacent frequency prototypes, such as higher attention weights for frequency prototypes 1 and 2, and 3 and 4, while the attention between distant frequency prototypes is weaker. This suggests that the model has learned a local interaction pattern for frequencies, namely, there is a stronger dependence between closely spaced frequency components, while the mixing of low-frequency and high-frequency modalities is not uniform.

2. The Variable Attention results indicate that different variables are not simply modeled independently. Each query variable generally assigns higher attention to MUFL and HUFL, especially for the target variable OT, where its attention weight to MUFL and HUFL is higher than its own attention weight to OT, suggesting that the model can utilize load-related variables to provide effective exogenous information for target prediction. At the same time, the overall distribution of variable attention is relatively smooth, indicating that the model does not rely entirely on a single variable but rather integrates multiple variable information through weighted fusion. These results verify the rationality of decoupling mode attention and variable attention in this study.

Click to download: Full original data, Python code

Previously interesting content:
Informer prediction model collection: Adding feature importance analysis!
Fast Fourier Transform violent growth points! A high-innovation time series classification model based on time-frequency feature fusion
STL-VMD secondary decomposition + Informer-LSTM parallel prediction model
Continue updates! Attention mechanism + lithium battery remaining life prediction collection
Direct writing in one area! CEEMDAN decomposition + Informer-LSTM +XGBoost combination prediction model
Comparative collection of photovoltaic power generation forecasting models based on Informer-SENet! Six sets of comparative forecasting models, directly written for graduation thesis or minor thesis!
Highly innovative model for predicting the remaining lifespan of lithium batteries using ICEEMDAN+SHAP for explainability analysis!
Predictive model based on Informer-BiGRUGATT-CrossAttention.
Highly innovative! Comprehensive predictive model based on ICEEMDAN+MSCNN-BiGRU-Attention.
Novel prediction model for air quality PM2.5 concentration! Comprehensive predictive model based on VMD rolling decomposition + attention mechanism.
End.
\t2. Friends who like this can click to follow, give the article a "like", share it in WeChat Moments or discuss groups.

## Images

<!-- PAYMENT_BLOCK -->

## Get this dataset

- Payment link (Stripe): [Buy now](https://buy.stripe.com/3cs8yP7sY87d0vu9AB)
- Payment QR code: ![Payment QR code](pay.png)

<!-- /PAYMENT_BLOCK -->
