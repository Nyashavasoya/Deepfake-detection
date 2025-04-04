# Deepfake-detection

## Part 1: Research and Analysis

---

### Spoofing Attacker Also Benefits from Self-Supervised Pretrained Model


Self-supervised learning (SSL) models like wav2vec 2.0, HuBERT, and WavLM represent a significant leap in anti-spoofing by leveraging large-scale pretraining on diverse datasets such as LibriSpeech, GigaSpeech, and VoxPopuli. These models extract high-level speech representations that generalize well to unseen spoofing attacks, making them highly robust. In experiments, they achieved an equal error rate (EER) of just 0.44%, demonstrating their effectiveness in distinguishing genuine from synthetic speech.

However, a critical challenge arises: attackers can also exploit SSL models, neutralizing the defender’s advantage. Since these pretrained models are publicly available, malicious actors can fine-tune them to generate more convincing deepfakes, eroding detection efficacy. While SSL-based systems excel in generalization, this symmetry between attackers and defenders calls for additional safeguards, such as adversarial training or model watermarking, to maintain an edge.

Another limitation is the requirement for fixed-length input alignment (64,600 samples), which can be restrictive for real-world applications with variable-duration audio.This issue is easy to address—techniques like dynamic padding, or processing chunks can enable flexible input handling without significant architectural changes.

Ultimately, SSL-based anti-spoofing offers a powerful, scalable solution, but its open availability to attackers means defenders must continuously innovate to stay ahead. Future work could explore ensemble methods (combining multiple SSL models) or real-time adversarial detection to mitigate these risks.



### END-TO-END ANTI-SPOOFING WITH RAWNET2

RawNet2 introduces a end-to-end approach to spoofing detection by processing raw audio waveforms directly, bypassing the need for handcrafted features like spectrograms. Its architecture employs fixed sinc filters in the initial layer—a deliberate design choice to prevent overfitting, especially on small datasets such as ASVspoof 2019 LA, which contains only six known attack types. Further processing is handled by residual blocks and GRU layers, which capture long-term temporal dependencies, while Feature Map Scaling (FMS) dynamically enhances the most discriminative features.

In testing, RawNet2 demonstrated best performance over baseline models, particularly when compared to configurations using learned filters.. Its efficiency stems from eliminating feature extraction steps, reducing preprocessing overhead, and maintaining robustness even with limited training data.

Same issue of static input size.

Overall, RawNet2 offers a lightweight, efficient solution for real-time anti-spoofing, particularly in scenarios where computational resources are limited. While its fixed input length can be adapted, its front-end signal processing approach remains optimized as-is, making it a strong candidate for deployment in practical applications. Future enhancements could explore hybrid architectures that combine RawNet2’s raw waveform processing with supplementary feature extractors for even greater robustness.



### Fully Automated End-to-End Fake Audio Detection


The Light-DARTS approach represents a significant advancement in automated anti-spoofing by combining neural architecture search (DARTS) with wav2vec's powerful feature extraction capabilities. This innovative framework eliminates the need for manual hyperparameter tuning, instead automatically discovering optimal network architectures through differentiable architecture search. The system leverages pre-trained wav2vec models to extract high-level speech representations, which are then processed by a lightweight yet effective network discovered through the DARTS process.

Performance results are impressive, with the system achieving a 77.26% relative reduction in Equal Error Rate (EER) compared to baseline methods. This substantial improvement demonstrates the effectiveness of combining automated architecture search with self-supervised learning features. The approach is particularly promising for adaptive deployment scenarios, where the system can automatically optimize itself for different environments or attack types without requiring extensive manual intervention.

This method does face some challenges. The architecture search process is computationally intensive, requiring resources during the training phase. Unfortunately, this limitation is not easily addressable without resorting to distributed training systems or working with predefined, constrained search spaces that may limit the optimal architectures discovered.

Another consideration is the system's reliance on wav2vec features, which may not perform equally well across all languages, particularly low-resource ones. Light-DARTS presents a good solution for organizations seeking cutting-edge, automated anti-spoofing capabilities. Its ability to self-optimize makes it particularly valuable in environments where attack methods may evolve over time. Future improvements could focus on developing more efficient architecture search algorithms or creating hybrid approaches that combine the benefits of automated search with hand-designed, domain-specific optimizations.



### AASIST: AUDIO ANTI-SPOOFING USING INTEGRATED SPECTRO-TEMPORAL

AASIST advances spoofing detection by combining spectral and temporal analysis through heterogeneous graph attention (HS-GAL) and max graph operations (MGO), achieving state-of-the-art results on ASVspoof 2021. Its graph pooling reduces nodes by 50%, enabling near-real-time performance despite its complexity. While computationally intensive, optimizations like layer simplification or quantization could improve efficiency. 


#### FINAL:
RawNet2 and SSL models are prioritized for real-time feasibility (simple architectures, minor adjustments).

AASIST is ideal for high-accuracy scenarios despite complexity, with partial optimizations.

Light-DARTS suits automated deployments but requires significant infrastructure.



## Part 2: Code

---

Code files present in this repo, here the AASIST model architecture with its pre-trained weight(trained on ASVspoof) is used to train in-the-wild dataset for deepfake AI detection. the model with updated weights after training and evaluating, was then saved and used for testing purpose. 



## Part 3: Implementation and Analysis of AASIST on In-The-Wild Dataset

---

Accuracy-wise, AASIST performs better than simpler architectures like RawNet2. Additionally, it works well on perturbed (noisy) data due to its attention mechanism. While it may take a few milliseconds longer than RawNet2, it provides better performance for real-time inference.
Spectro-temporal graph attention detects unnatural harmonics and Heterogeneous graph pooling disrupts gradient-based attacks for adversarial examples

### 1. Implementation Process

#### Challenges Encountered
The dataset required reverberation and noise addition to better simulate real-world conditions and improve model robustness

Faced memory issues due to the large dataset size

Training was computationally expensive, requiring high GPU power, memory, and processing capabilities

#### How Challenges Were Addressed
Continuously cleared cache and removed unnecessary files from memory to manage RAM usage

Leveraged AASIST's pre-trained weights to avoid training from scratch

Ensured data preprocessing (e.g., padding, mono conversion) was memory-efficient and batched properly

#### Assumptions Made
Background noise in audio samples does not significantly correlate with spoofing artifacts

Real-world audio (even noisy or compressed) can be effectively analyzed using AASIST due to its design and accuracy

### 2. Model Selection and Analysis

#### Why AASIST
Achieved state-of-the-art performance on the ASVspoof 2021 LA track:

Equal Error Rate (EER): 0.83%, outperforming RawNet2's 1.12%.

Specifically designed to handle raw waveform input, reducing preprocessing steps.

Robust against compressed or low-quality recordings, a common scenario in deepfake detection in-the-wild.

Effective spectro-temporal feature extraction using graph attention mechanisms for both frequency and time domains.

### 3. Architecture Components of AASIST

#### 3.1 Input Processing
Input Shape: Raw audio waveform
Shape: (batch_size, 1, num_samples)
Example: (32, 1, 64000) for 4-second audio at 16kHz

Optional Augmentation: Frequency masking (only during training)

#### 3.2 Time-Domain Processing
Sinc Convolution Layer:

Applies 40 Mel-scaled bandpass filters

Preserves phase information, crucial for spoof detection

Spectral Smoothing:

Operations: Absolute value → Max pooling → BatchNorm → SELU activation

#### 3.3 Encoder Network
Purpose: Detect both local glitches and global anomalies

Structure: 6 Residual Blocks with filter sizes: [32, 32, 64, 64, 64, 64]

Each residual block includes:
Conv2D (2×3) → BatchNorm → SELU

Conv2D (2×3) → Residual Skip Connection

MaxPool (1×3)

#### 3.4 Dual Graph Attention Paths
##### 3.4.1 Spectral Graph Path (GAT-S)
Goal: Learn interactions between frequency bins

Max pooling along time dimension

Positional Embedding: Learnable pos_S added to 23 frequency bins

Graph Attention Layer:

Nodes: Frequency bins

Projects to dimension: gat_dim = 128

Retains top 50% most informative nodes

##### 3.4.2 Temporal Graph Path (GAT-T)
Goal: Capture long-range temporal dependencies

Max pooling along frequency dimension

Graph Attention Layer:

Nodes: Time steps

Projects to dimension: gat_dim = 128

Retains top 50% most informative nodes

#### 3.5 Heterogeneous Graph Fusion (HS-GAL)
Input:

Spectral graph: (batch, 12, 128)

Temporal graph: (batch, T//2, 128)

Cross-Domain Attention Mechanism:

Separate projection layers for temporal and spectral inputs

Three types of attention:

Spectral ↔ Spectral

Temporal ↔ Temporal

Spectral ↔ Temporal

Master Node: Global context vector updated with attention from both paths

Output: Refined features capturing interactions across both domains

#### 3.6 Hierarchical Pooling & Readout
Pooling Strategy: Multi-stage, node-pruning at 50%, 50%, and 70%

Aggregation:

Max and Average pooling across final spectral and temporal graphs

Concatenated with master node output

Final vector shape: (batch, 640)

#### 3.7 Classification Head
Linear layer projects the 640-dimensional feature vector to 2 output classes:

Bona-fide (real)

Spoof (fake)

Outputs:

Raw classification logits

Optionally, intermediate 640-D feature vector for further analysis or transfer learning

### 4. Performance Evaluation and Analysis

#### Performance Results
The In-The-Wild dataset was chosen to evaluate AASIST on a different dataset than the one it was originally trained on.

Achieved test accuracy: 99.56%

#### Observed Strengths and Weaknesses
##### Strengths
Robust to noise:

Performs well even with background noise, making it suitable for real-time applications where environmental sounds cannot be controlled.

End-to-End Processing:

Works directly on raw waveform input, eliminating the need for complex preprocessing steps like spectrogram conversion.

##### Weaknesses
Complex architecture:

Compared to RawNet2, AASIST is significantly more complex, making it harder to interpret and fine-tune.

Fixed input length (4s):

The model requires fixed 4-second audio segments, which may not always align naturally with spoken content.

#### Suggestions for Future Improvements
Make input length dynamic:

Instead of using strictly 4s audio chunks, allow for variable-length inputs or implement sliding window processing (with overlap) to handle continuous audio streams more effectively.

Improve real-time efficiency:

Optimize computational efficiency for faster inference, especially for live-streamed audio.

### 5. Reflection Questions

#### Question 1: Implementation Challenges
The implementation was not very challenging because the model was trained using official code from the AASIST paper.

The primary effort went into adapting it to the In-The-Wild dataset and evaluating its generalization.

#### Question 2: Real-World Performance
Research datasets (e.g., ASVspoof) contain clean audio, often recorded in controlled conditions.

Real-world environments involve noisy, compressed, and lossy recordings, which may cause a higher EER.

#### Question 3: Additional Data Requirements
Compressed audio samples:

Training with low-bitrate and heavily compressed deepfake audio to improve robustness.

Harder-to-detect spoof attacks:

Collecting/generated deepfake audio specifically designed to bypass AASIST.

Environmental noise augmentation:

Including real-world noise from crowds, streets, and office spaces to improve adaptation.

#### Question 4: Production Deployment Strategy
Backend API for Audio Streaming → FastAPI

Handles real-time audio input and inference.

Containerization & Orchestration → Docker & Kubernetes

Ensures scalable deployment with load balancing.

Monitoring & Logging → Grafana

Tracks real-time inference latency, detection accuracy, and system health.

Database Storage

Stores processed audio samples and metadata for future analysis or retraining.

## References

### Academic Papers

1. **RawNet2: End-to-End Anti-spoofing**
   - Authors: Tak, H., Patino, J., Todisco, M., Nautsch, A., Evans, N., & Larcher, A.
   - Conference: ICASSP 2021
   - DOI: [10.1109/ICASSP39728.2021.9414234](https://doi.org/10.1109/ICASSP39728.2021.9414234)

2. **Fully Automated End-to-End Fake Audio Detection**
   - Authors: Wang, C., Yi, J., Tao, J., Sun, H., Chen, X., et al.
   - Year: 2022
   - arXiv: [2208.09618](https://arxiv.org/abs/2208.09618)

3. **AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks**
   - Authors: Jung, J., Heo, H., Tak, H., Shim, H., Chung, J.S., et al.
   - Year: 2021
   - arXiv: [2110.01200](https://arxiv.org/abs/2110.01200)

4. **Spoofing Attacker Also Benefits from Self-Supervised Pretrained Model**
   - Authors: Ito, A., & Horiguchi, S.
   - Year: 2023
   - arXiv: [2305.15518](https://arxiv.org/abs/2305.15518)

### Implementation Reference

Official AASIST Implementation: [github.com/clovaai/aasist](https://github.com/clovaai/aasist)

