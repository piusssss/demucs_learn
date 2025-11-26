MUSIC SOURCE SEPARATION WITH BAND-SPLIT ROPE TRANSFORMER
Wei-Tsung Lu∗
, Ju-Chiang Wang∗
, Qiuqiang Kong, Yun-Ning Hung
SAMI, ByteDance
{weitsung.lu, ju-chiang.wang, kongqiuqiang, yunning.hung}@bytedance.com
ABSTRACT
Music source separation (MSS) aims to separate a music recording
into multiple musically distinct stems, such as vocals, bass, drums,
and more. Recently, deep learning approaches such as convolutional
neural networks (CNNs) and recurrent neural networks (RNNs) have
been used, but the improvement is still limited. In this paper, we
propose a novel frequency-domain approach based on a Band-Split
RoPE Transformer (called BS-RoFormer). BS-RoFormer relies on
a band-split module to project the input complex spectrogram into
subband-level representations, and then arranges a stack of hierar￾chical Transformers to model the inner-band as well as inter-band
sequences for multi-band mask estimation. To facilitate training the
model for MSS, we propose to use the Rotary Position Embedding
(RoPE). The BS-RoFormer system trained on MUSDB18HQ and
500 extra songs ranked the first place in the MSS track of Sound
Demixing Challenge (SDX’23). Benchmarking a smaller version of
BS-RoFormer on MUSDB18HQ, we achieve state-of-the-art result
without extra training data, with 9.80 dB of average SDR.
Index Terms— Music source separation, band-split, rotary po￾sition embedding, Transformer, BS-RoFormer, SDX’23.
1. INTRODUCTION
Music source separation (MSS) [1, 2] is a task of separating a mu￾sic recording into musically distinct sources. As defined in the 2015
Signal Separation Evaluation Campaign (SiSEC) [3], researchers are
focused on the 4-stem setting: vocals, bass, drums, and other. The
MUSDB18 dataset [4] has been used to benchmark the performance.
MSS is considered to be a more challenging audio separation task
given the complexity of input signals (i.e., stereo, 44.1k Hz sampling
rate) and more sources to separate (i.e., 4 or more stems). MSS can
benefit various downstream MIR tasks, such as vocal pitch estima￾tion [5], music transcription [6], and so on. MSS may also enable
applications such as karaoke-version generation, intelligent music
editing and remixing [7], and other music inspired derivative works.
In recent years, many deep learning approaches have been pro￾posed to tackle the MSS problem. They are typically categorized
into frequency-domain and time-domain approaches. Frequency￾domain approaches reply on Fourier transform to derive a time￾frequency (T-F) representation for input. Then, models such as
fully connected neural networks [8], convolutional neural networks
(CNNs) [9, 10, 11], and recurrent neural networks (RNNs) [12] are
applied. On the other hand, time-domain approaches such as Wave￾U-Net [13], ConvTasNet [14], and Demucs [15] build their neural
networks directly on the waveform input. Recently, Hybrid Trans￾former Demucs (HTDemucs) [16] proposes to use a cross-domain
Transformer to combine the frequency- and time-domain models.
∗ Equal contribution
Speaking of frequency-domain approaches, most existing mod￾els do not make assumptions on weighting the frequency bins based
on prior knowledge. Models like CNNs use the same kernels across
all frequency bins for all target sources, expecting the models can
learn the band-pass mechanism from the data with raw frequency
bins. However, this might not be efficient, since different frequency
bands may have different patterns that preferably characterize dif￾ferent instrument sources. Recently, band-split mechanism is intro￾duced at the front-end in Band-split RNN (BSRNN) [17] to force
the model to learn band-wise features. BSRNN splits the T-F rep￾resentation into non-overlapping subbands and arranges a stack of
interleaved RNNs to simultaneously model the inner-band and inter￾band sequences. This design has demonstrated state-of-the-art result
on Musdb18. However, its use of RNN could be still sub-optimal,
given that Transformer [18] has constantly shown the superiority in
many relevant tasks of sequential data modeling.
The idea of modeling the T-F representation of music audio us￾ing interleaved Transformers has been explored recently [19, 20].
For example, SpecTNT [19] consists of two Transformer encoders
in a hierarchical order to model the frequency and time sequences
respectively, and then uses the frequency class tokens to connect be￾tween the two Transformers. This or similar ideas have shown state￾of-the-art performance in various music transcription related tasks
[20, 21, 22], but have not been studied for MSS. In this paper, we
propose a novel MSS approach based on Band-Split RoPE Trans￾former (termed as BS-RoFormer) that originates from combining the
ideas of band-split and hierarchical Transformer architecture. Mak￾ing it work, however, is non-trivial, mainly because the model is
large, and training it effectively is very memory and time consum￾ing. We introduce the use of Rotary Position Embedding (RoPE)
[23] in Transformer significantly improves the performance. Other
efforts such as checkpointing, mixed precision, and flash attention
[24] could also help the training efficiency. We submitted the BS￾RoFormer system to Sound Demixing Challenge 2023 (SDX’23)1
,
the Music Separation track. Our system ranked the first place and
outperformed the second best by a large margin in SDR. In ablation
study, we demonstrate the importance of RoPE and that a smaller
BS-RoFormer model trained solely on MUSDB18HQ can achieve
state-of-the-art performance compared to existing models.
2. SYSTEM OVERVIEW
Fig. 1 depicts the system. Let x ∈ R
C×L
denote the audio wave￾form of the input mixture, where C and L are the numbers of chan￾nels and audio samples, respectively. Our frequency-domain MSS
system uses the complex spectrogram as input, where x is trans￾formed into a time-frequency (T-F) representation X ∈ C
C×T ×F
by a short-time Fourier transform (STFT), where T and F are the
1
https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/
arXiv:2309.02612v2 [cs.SD] 10 Sep 2023
Fig. 1. The framework of the Band-Split RoFormer system.
the numbers of frames and frequency bins, respectively. Let fθ de￾note a T-F neural network with a set of learnable parameters θ, the
output of fθ can be linear spectrums, ideal binary masks (IBMs),
ideal ratio mask (IRMs), or complex IRMs (cIRMs) [25]. We adopt
the cIRMs, denoted as Mˆ ∈ C
C×T ×F
. In this work, the goal of fθ
is to estimate the cIRMs: Mˆ = fθ(X), so that both the magnitude
and phase signals of the target source are attained.
Then, the separated complex spectrogram Yˆ ∈ C
C×T ×F
is ob￾tained by multiplying the cIRM by the input complex spectrogram:
Yˆ = Mˆ ⊙ X. Finally, an inverse STFT (iSTFT) is applied to Yˆ to
recover the separated signal yˆ in the time-domain. Mean absolute er￾ror (MAE) loss between the reference y and output yˆ is used to train
fθ. Specifically, the objective loss includes both the time-domain
MAE and the multi-resolution complex spectrogram MAE [26]:
loss = ||y − yˆ|| +
S−1 X
s=0
||Y
(s) − Yˆ (s)
||, (1)
where S = 5 multi-resolution STFTs are used with the window sizes
of [4096, 2048, 1024, 512, 256] and a fixed hop size of 147, which
is equivalent to 300 frames per second.
3. BAND-SPLIT ROPE TRANSFORMER
BS-RoFormer consists of a band-split module, RoPE Transformer
blocks, and a multi-band mask estimation module.
3.1. Band-Split Module
In BSRNN [17], it has shown that the band-split strategy can help the
frequency-domain approach. One rationale is that the input mixture
audio X is a full-band signal, so splitting the bands based on a prior
setting can foster the model to purify the learned representations at
different bands, gaining the robustness against cross-band vague￾ness. Following [17], we split X into N uneven non-overlapping
subbands along the frequency axis and apply individual multi-layer
perceptions (MLPs) to each subband. We denote the output of each
subband as Xn ∈ C
C×T ×Fn , where Fn is the number of frequency
bins in the n-th subband. All subbands Xn constitute the entire com￾plex spectrum X, and there is P N
n=1 Fn = F.
Each MLP consists of a RMSNorm layer [27] followed by a
linear layer. The RMSNorm layer regularizes the summed inputs to a
neuron in one layer according to root mean square. RMSNorm is an
efficient replacement to the LayerNorm normalization [27]. The n-th
linear layer contains a learnable matrix with a shape of (C×Fn)×D
and a learnable bias with a shape of D, where D is the number of
features. The transformed output of each subband is denoted as Hn
0
with a shape of T × D. We stack all Hn
0
, n = 1, . . . , N, along the
subband axis to obtain a stacked H0 with a shape of T × N × D as
the input to the RoPE Transformer blocks.
Algorithm 1 Hierarchical Transformer for a time-frequency input,
with parameters B: batch size, T: frames number, N: subband num￾ber, and D: latent dimension.
1: Input: Hl with a shape of B × T × N × D
2: Permute Hl
to (B × N) × T × D
3: Apply a Transformer along T.
4: Rearrange the above output to (B × T) × N × D
5: Apply a Transformer along N.
6: Output: Permute the above output to Hl+1 with a shape of B ×
T × N × D.
3.2. RoPE Transformer Blocks
Suppose there are L blocks in RoPE Transformer blocks, each
block’s output is denoted as Hn
l ∈ R
T ×N×D. The band-split output
Hn
0
serves as the input for the 1-st Transformer block. Different
from conventional Transformer encoder that applies self-attention
to one-dimensional sequence (e.g., time), our Transformer structure
is hierarchical, where interleaved Transformer layers are in turn
applied to the time and frequency axes in a Transformer block. The
former Transformer, called time-Transformer, models the inner￾band (local) temporal sequence, while the latter Transformer, called
subband-Transformer, handles the inter-band (global) spectral se￾quence to ensure the information across bands is exchangeable. The
process of a Transformer block is presented in Algorithm 1. For
time-Transformer, all mini-batches (with a batch size of B) and
subbands are stacked; for subband-Transformer, all mini-batches
and frames are stacked. All Transformer layers use the same archi￾tecture. Fig. 2 depicts a Transformer layer, which consists of an
attention module (with rotary position embedding) and a feedfor￾ward module.
3.2.1. Attention Module with Rotary Position Embedding
In an attention module, we first rearrange the input to a shape as
needed (see Algorithm 1), followed by a RMSNorm [27]. Then, we
apply the query, key, and value layers to predict the query, key, and
value [18]: Q = HlWq, K = HlWk, and V = HlWv, where Q,
K, and V have the shapes of D × Z, and Z is the latent dimension.
Learnable matrices Wq, Wk, and Wv have the shapes of D×Z. The
Q, K, and V are split into multiple heads. For positional encoding,
we propose to use the rotary position embedding (RoPE) [23], which
are applied as:
Qˆ = Rot(Q)
Kˆ = Rot(K)
(2)
where Rot(·) is the RoPE encoder [23] shared throughout the
entire Transformer blocks. The time-Transformer and subband￾Transformer have their own RoPE encoders that apply rotation
Fig. 2. Diagram of a Transformer layer. This example is a time￾Transformer along the temporal sequence and B
′ = B × N.
matrices on each embedding in Q and K based on its position of the
corresponding sequence. Then, we apply the attention operation by
using the processed query, key, and value as follows [18]:
output = Dropout Sof tmax(p
Qˆ
Z/h
Kˆ T
)
! V, (3)
where h is the number of heads. Equation (3) is the core part that
requires the most computation. To speed up, we employ the FlashAt￾tention technique [24]. After the attention module, a fully-connected
layer with Dropout is used. A residual addition is applied between
the input and output of the attention module.
We believe the use of RoPE is crucial in this work. Given the
proposed hierarchical Transformer structure, the positional embed￾dings for the time- and subband-Transformers should be robust to
the alternating self-attention operations between the time and sub￾band axes. Directly adding learnable absolute positional embeddings
to encode the positions may fail to maintain the scale of norm after
repetitive transposed self-attention processes. Since RoPE encodes
the relative position by multiplying the context representations with
a rotation matrix, we argue this would help preserve the positional
information for both time and subband sequences.
3.2.2. Feedforward Module
The feedforward module consists of a RMSNorm layer, a fully con￾nected layer with GeLU activation [28], and a dropout. Then, one
more fully connected layer with dropout is applied. Similarly, we
apply a residual addition between the input and output of the feed￾forward module.
3.3. Multi-band Mask Estimation Module
We denote the outputs from the RoPE Transformer blocks as
Hn
L
, n = 1, . . . , N. Similar to the band-split module, we apply
N individual MLP layers to each subband Hn
L
. Each MLP layer
consists of a RMSNorm layer, a fully connected layer followed
by a Tanh activation, and a fully connected layer followed by a
gated linear unit (GLU) layer [29]. The n-th MLP layer outputs
the subband mask ˆMn with a shape of (2 × C) × T × Fn, which
contains a mask of real values and a mask of imaginary values. All
the outputs are concatenated along the frequency axis to derive a
cIRM Mˆ ∈ C
C×T ×F
.
4. EXPERIMENTS
This section first presents our implementation details of the winning
system in Sound Demixing Challenge 2023 (SDX’23). Then, we
describe the ablation study on the a smaller version of BS-RoFormer
and the effectiveness of Rotary position embedding.
4.1. Dataset
The Musdb18HQ dataset [4] contains 100 and 50 songs for training
and evaluation, respectively. All recordings are stereo with a sam￾pling rate of 44.1k Hz. Each recording contains four stems: vocals,
bass, drums, and other. For MDX’23 submission, we also use an
“In-House” dataset containing 500 songs: 450 songs are added to
the training set, and 50 songs are for validation. Each song has four
stems with a sampling rate of 44.1k Hz following [4].
4.2. Evaluation Metrics
We adopt the signal-to-distortion ratio (SDR) [30] implemented by
museval [31] as the evaluation. The SDR score is calculated by:
SDR(y, yˆ) = 10 log10
||y||2
||sˆ− s||2
(4)
Higher SDR indicates better separation quality.
4.3. Data Augmentation
Our model takes a segment of 8-seconds waveform for input and
output. For better training efficiency, we maintain a dynamic pool
of 8-seconds stem-level segments in the memory. At each training
step, the dataloader samples a batch of random 4-stem tuples from
the pool. Each stem audio is processed with a random gain in a
range of ±3 dB and has a chance of 10% to be replaced with a si￾lence waveform. The four stems of a tuple, which can originate from
different songs, are mixed by linear addition to generate a training
example. This random-mixing strategy will produce examples that
are not musically aligned.
The dynamic pool has a size (e.g., 512 segments for each stem)
larger than the batch size to facilitate wider diversity for a batch,
and is updated in a first-in-first-out manner at each training step to
remain its size. To add samples to the pool, we crop an 8-seconds
segment of 4 stems at a random moment from a full-length song. A
stem-level segment must satisfy a loudness level larger than -50 dB.
4.4. Implementation Details
Band-split Module. We apply a Hann window size of 2048 and a
hop size of 10 ms for STFT to compute the complex spectrogram
of 8-seconds long input. We use the following band-split scheme:
2 bins per band for frequencies under 1000 Hz, 4 bins per band be￾tween 1000 Hz and 2000 Hz, 12 bins per band between 2000 Hz
and 4000 Hz, 24 bins per band between 4000 Hz and 8000 Hz, 48
bins per band between 8000 Hz and 16000 Hz, and the remaining
bins beyond 16000 Hz equally divided into two bands. This results
The ⋆ symbol indicates the proposed BS-RoFormer.
System Vocals Bass Drums Other Mean
1. SAMI-ByteDance⋆ 11.36 11.15 10.27 7.08 9.97
2. ZFTurbo [33] 10.51 9.94 9.53 7.05 9.26
3. kimberley jensen 10.40 10.06 9.47 6.80 9.18
4. kuielab [34] 10.01 9.72 9.43 6.72 8.97
5. alina porechina 9.07 9.92 9.29 6.23 8.63
(Baseline) BSRNN [17] 7.98 5.63 6.53 4.43 6.14
Table 1. SDX’23 Leaderboard C final results (in global SDR).
in a total of 62 bands. All bands are non-overlapping. We derive
this setting referring to [17]. In our pilot study, slightly varying the
band-slplit setting does not show significant difference in results.
Configuration. We use D = 384 for feature dimension, L = 12 for
the number of Transformer blocks, 8 heads for each Transformer,
and a dropout rate of 0.1. The multi-band mask estimation module
utilizes MLPs with a hidden layer dimension of 4D. The result￾ing model has 93.4M parameters. We use Pytorch-Lightning 2.0
and Exponential Moving Averaging (EMA) with a decay 0.999. For
training, we adopt the AdamW optimizer [32] with a learning rate
(LR) 5 × 10−4
, and reduce LR by 0.9 every 40k steps. To optimize
the GPU memory usage, we employ the checkpointing technique as
well as the mixed precision, where the STFT and iSTFT modules
use FP32 and all the others use FP16.
We trained three separation models respectively for vocals, bass,
and drums using In-House and the Musdb18HQ training set. For the
“other” stem, we subtracted the vocals, bass, and drums signals from
the input mixture in the time domain. For each model, the training
process lasted for 4 weeks using 16 Nvidia A100-80GB GPUs with a
total batch size of 128 (i.e., 8 for each GPU). The model checkpoint
with the best validation result was selected.
Enframe & Deframe. We use a hop size of 4 seconds for
segment enframing. Two deframing methods are studied: “trun￾cate&concat” (TC) and “overlap&average” (OA). In TC, we discard
the front and rear 2-seconds signals of each segment-level output
and concatenate all truncated segments. In OA, each output segment
has a 4-second overlap by the next segment at the rear, and the over￾lapped part is averaged.
4.5. SDX’23 MSS Results
SDX’23 MSS track featured three leaderboards, A: label noise, B:
bleeding, and C: standard music separation. The first two leader￾boards set some requirements about robustness, so the submitted
systems can be trained only on the provided dataset. Our system
participated the standard music separation track, where the submit￾ted systems can be trained on any data with no limitation, so it is
more desirable. Table 1 presents the final results on the organizer’s
private set for the top-5 systems of SDX’23 MSS leaderboard C [35].
We used TC deframing for the submission.
Our system (SAMI-ByteDance) outperforms the second best
(ZFTurbo) by a large margin (0.71 dB on average), demonstrating
the effectiveness of the proposed BF-RoFormer. When comparing
the results of vocals, bass, and drums (since we did not train the
“other” separator), the average SDR difference is 0.93 dB. In terms
of listening experience, we found our model outputs are highly
accurate, meaning they contain less residues from the background.
However, the resulting spectrogram may look sharp and less foggy
compared to that generated by existing CNN-based models, and this
might not be a favorable feature to some audiences. According to
The † symbol indicates MSS systems trained with extra data.
The ‡ symbol indicates the results are evaluated on non-HQ version.
Vocals Bass Drums Other Avg.
Conv-TasNet [14]‡ 6.81 5.66 6.08 4.37 5.73
Spleeter [36]†‡ 6.86 5.51 6.71 4.55 5.91
ResUNet [10]‡ 8.98 6.04 6.62 5.29 6.73
HDemucs [37]† 8.13 8.76 8.24 5.59 7.68
BSRNN [17] 10.01 7.22 9.01 6.70 8.24
BSRNN [17]† 10.47 8.16 10.15 7.08 8.97
Sparse HT Demucs [16]† 9.37 10.47 10.83 6.41 9.27
BS-Transformer (L=6, OA) 9.15 6.11 3.08 4.77 5.78
BS-RoFormer (L=6, TC) 10.68 11.28 9.41 7.68 9.76
BS-RoFormer (L=6, OA) 10.66 11.31 9.49 7.73 9.80
BS-RoFormer (L=12, OA)† 12.72 13.32 12.91 9.01 11.99
Table 2. Median SDRs of different MSS models.
[35], our model outputs gained more preference from musicians and
educators than from music producers in the listening test of SDX23.
4.6. Ablation Study
We investigate the effects of three aspects: 1) the number of Trans￾former blocks; 2) RoPE or absolute positional encoding; 3) TC or
OA deframing methods. To this end, we implement three smaller
variants of the proposed model with L=6. In one variant, we re￾move the RoPE and add learnable absolute positional embeddings
in the attention module, and call it “BS-Transformer,” since it uses
a standard Transformer. We train a dedicated separation model for
the “other” stem, except “BS-RoFormer (L=12, OA),” which is our
MDX’23 submission. The numbers of parameters for BS-RoFormer
and BS-Transformer with L=6 are 72.2M and 72.5M, respectively.
Models with L=6 are trained solely on the Musdb18HQ training set
using 16 Nvidia V100-32GB GPUs. We do not use the In-House
dataset for ablation study. The effective batch size is 64 (i.e., 4 for
each GPU) using accumulate grad batches=2.
Table 2 presents the comparison between our proposed models
and existing models. We report the median SDR across the median
SDRs over all 1 second chunks of each test song in Musdb18HQ fol￾lowing prior works. First, BS-RoFormer with L=6 is still very com￾petitive and can achieve state-of-the-art performance compared to
models trained without extra training data. “BS-RoFormer†
(L=12,
OA)” outperforms all existing models by a large margin (over 2 dB
on average). Second, BS-Transformer without RoPE does not seem
to work given the low SDRs, demonstrating that RoPE is crucial
in our proposed architecture as discussed in Section 3.2.1. Accord￾ing to our observations, the training progress of BS-Transformer is
very slow, and it still remains low SDRs after two weeks of training
on Musdb18HQ. Instead, BS-RoFormer models with L=6 get con￾verged within a week. Lastly, the OA deframing shows better per￾formance than TC except vocals. Qualitatively, OA offers smoother
song-level quality that can improve the overall listening experience.
5. CONCLUSION
We have presented the BS-RoFormer model, which is based on a
novel hierarchical RoPE Transformer architecture. Its outstanding
performance may shed the light on the development of next genera￾tion MSS systems. For future work, we would focus on improving
the qualitative performance. The sharp sound quality may be im￾proved by introducing overlapping band projection at the front-end.
6. REFERENCES
[1] Z. Rafii, A. Liutkus, F.-R. Stoter, S. I. Mimilakis, D. FitzGer- ¨
ald, and B. Pardo, “An overview of lead and accompaniment
separation in music,” IEEE/ACM Trans. Audio Speech Lang.
Process., vol. 26, no. 8, pp. 1307–1335, 2018.
[2] Y. Mitsufuji, G. Fabbro, S. Uhlich, F.-R. Stoter, A. D ¨ efossez, ´
M. Kim, W. Choi, C.-Y. Yu, and K.-W. Cheuk, “Music demixing challenge 2021,” Frontiers in Signal Processing, 2022.
[3] A. Liutkus, F.-R. Stoter, Z. Rafii, D. Kitamura, B. Rivet, N. Ito, ¨
N. Ono, and J. Fontecave, “The 2016 signal separation evaluation campaign,” in 13th International Conference on Latent
Variable Analysis and Signal Separation, 2017, pp. 323–332.
[4] Z. Rafii, A. Liutkus, F.-R. Stoter, S. I. Mimilakis, and R. Bit- ¨
tner, “The MUSDB18 corpus for music separation,” Dec.
2017, https://doi.org/10.5281/zenodo.1117372.
[5] T. Nakano, K. Yoshii, Y. Wu, R. Nishikimi, K. W. E. Lin, and
M. Goto, “Joint singing pitch estimation and voice separation
based on a neural harmonic structure renderer,” in IEEE WASPAA, 2019, pp. 160–164.
[6] L. Lin, Q. Kong, J. Jiang, and G. Xia, “A unified model for
zero-shot music source separation, transcription and synthesis,” in ISMIR, 2021.
[7] M. Moon, “TikTok-owner ByteDance debuts Ripple music creation app,” https://www.engadget.com/tiktok-ownerbytedance-debuts-ripple-music-creation-app-130023602.html.
[8] E. M. Grais, M. U. Sen, and H. Erdogan, “Deep neural networks for single channel source separation,” in IEEE ICASSP,
2014, pp. 3734–3738.
[9] P. Chandna, M. Miron, J. Janer, and E. Gomez, “Monoaural ´
audio source separation using deep convolutional neural networks,” in Latent Variable Analysis and Signal Separation
(LVA/ICA), 2017, pp. 258–266.
[10] Q. Kong, Y. Cao, H. Liu, K. Choi, and Y. Wang, “Decoupling
magnitude and phase estimation with deep resunet for music
source separation,” in ISMIR, 2021.
[11] A. Jansson, E. Humphrey, N. Montecchio, R. Bittner, A. Kumar, and T. Weyde, “Singing voice separation with deep U-Net
convolutional networks,” in ISMIR, 2017.
[12] S. Uhlich, M. Porcu, F. Giron, M. Enenkl, T. Kemp, N. Takahashi, and Y. Mitsufuji, “Improving music source separation
based on deep neural networks through data augmentation and
network blending,” in IEEE ICASSP, 2017, pp. 261–265.
[13] D. Stoller, S. Ewert, and S. Dixon, “Wave-U-Net: A multiscale neural network for end-to-end audio source separation,”
arXiv preprint arXiv:1806.03185, 2018.
[14] Y. Luo and N. Mesgarani, “Conv-TasNet: Surpassing ideal
time–frequency magnitude masking for speech separation,”
IEEE/ACM Trans. Audio Speech Lang. Process., vol. 27, no.
8, pp. 1256–1266, 2019.
[15] A. Defossez, N. Usunier, L. Bottou, and F. Bach, “Music ´
source separation in the waveform domain,” arXiv preprint
arXiv:1911.13254, 2019.
[16] S. Rouard, F. Massa, and A. Defossez, “Hybrid transformers ´
for music source separation,” in IEEE ICASSP, 2023.
[17] Y. Luo and J. Yu, “Music Source Separation With Band-Split
RNN,” IEEE/ACM Trans. Audio Speech Lang. Process., vol.
31, pp. 1893–1901, 2023.
[18] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones,
A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all
you need,” NeurIPS, vol. 30, 2017.
[19] W.-T. Lu, J.-C. Wang, M. Won, K. Choi, and X. Song,
“SpecTNT: A time-frequency transformer for music audio,” in
ISMIR, 2021.
[20] W.-T. Lu, J.-C. Wang, and Y.-N. Hung, “Multitrack music transcription with a time-frequency perceiver,” in IEEE ICASSP,
2023.
[21] Y.-N. Hung, J.-C. Wang, X. Song, W.-T. Lu, and M. Won,
“Modeling beats and downbeats with a time-frequency transformer,” in IEEE ICASSP, 2022, pp. 401–405.
[22] J.-C. Wang, Y.-N. Hung, and J. B. Smith, “To catch a chorus,
verse, intro, or anything else: Analyzing a song with structural
functions,” in IEEE ICASSP, 2022, pp. 416–420.
[23] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, “Roformer: Enhanced transformer with rotary position embedding,” arXiv preprint arXiv:2104.09864, 2021.
[24] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAt- ´
tention: Fast and memory-efficient exact attention with ioawareness,” in NeurIPS, 2022.
[25] D. Wang and J. Chen, “Supervised speech separation based on
deep learning: An overview,” IEEE/ACM Trans. Audio Speech
Lang. Process., vol. 26, no. 10, pp. 1702–1726, 2018.
[26] E. Guso, J. Pons, S. Pascual, and J. Serr ´ a, “On loss functions `
and evaluation metrics for music source separation,” in IEEE
ICASSP, 2022, pp. 306–310.
[27] B. Zhang and R. Sennrich, “Root mean square layer normalization,” NeurIPS, vol. 32, 2019.
[28] D. Hendrycks and K. Gimpel, “Gaussian error linear units
(GeLUs),” arXiv preprint arXiv:1606.08415, 2016.
[29] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language
modeling with gated convolutional networks,” in ICML, 2017,
pp. 933–941.
[30] E. Vincent, R. Gribonval, and C. Fevotte, “Performance mea- ´
surement in blind audio source separation,” IEEE Trans. Audio
Speech Lang. Process., vol. 14, no. 4, pp. 1462–1469, 2006.
[31] F.-R. Stoter, A. Liutkus, and N. Ito, “The 2018 signal separa- ¨
tion evaluation campaign,” in LVA/ICA, 2018, pp. 293–305.
[32] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in ICLR, 2019.
[33] R. Solovyev, A. Stempkovskiy, and T. Habruseva, “Benchmarks and leaderboards for sound demixing tasks,” arXiv
preprint arXiv:2305.07489, 2023.
[34] M. Kim and J. H. Lee, “Sound demixing challenge 2023–
music demixing track technical report,” arXiv preprint
arXiv:2306.09382, 2023.
[35] G. Fabbro et al., “The sound demixing challenge 2023–music
demixing track,” arXiv preprint arXiv:2308.06979, 2023.
[36] R. Hennequin, A. Khlif, F. Voituret, and M. Moussallam,
“Spleeter: a fast and efficient music source separation tool with
pre-trained models,” Journal of Open Source Software, vol. 5,
no. 50, pp. 2154, 2020.
[37] A. Defossez, “Hybrid spectrogram and waveform source sep- ´
aration,” arXiv preprint arXiv:2111.03600, 2021.