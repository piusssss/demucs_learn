STFT
TEncoder1(Cin = 2, Cout = 48)
T time steps
T/4 time steps
TEncoder2(Cin = 48, Cout = 96)
T/16 time steps
. . .
T/256 time steps
TEncoder5(Cin = 384, Cout = 768)
T/1024 time steps
ZEncoder1(Cin = 2 · 2, Cout = 48)
T /1024 time steps 2048 freq.
T/1024 time steps, 512 freq.
ZEncoder2(Cin = 48, Cout = 96)
T/1024 time steps, 256 freq.
. . .
T/1024 time steps, 8 freq.
ZEncoder5(Cin = 384, Cout = 768)
T/1024 time steps, 1 freq. +
Encoder6(Cin = 768, Cout = 1586)
T/2048 time steps
Decoder6(Cin = 1586, Cout = 768)
T/1024 time steps
ZDecoder5(Cin = 768, Cout = 386)
T/1024 time steps, 8 freq.
. . .
T/1024 time steps, 256 freq.
ZDecoder2(Cin = 96, Cout = 48)
T/1024 time steps, 512 freq.
ZDecoder2(Cin = 48, Cout = 4 · 2 · 2)
T/1024 time steps, 2048 freq.
ISTFT
TDecoder5(Cin = 768, Cout = 386)
T/1024 time steps
. . .
T/256 time steps
TDecoder2(Cin = 96, Cout = 48)
T/4 time steps
TDecoder1(Cin = 48, Cout = 4 · 2 · 2)
T time steps.
+
Figure 1: Hybrid Demucs architecture. The input waveform is processed both through a temporal
encoder, and ﬁrst through the STFT followed by a spectral encoder. The two representations are
summed when their dimensions align. The decoder is built symmetrically. The output spectrogram go
through the ISTFT and is summed with the waveform outputs, giving the ﬁnal model output. The Z
preﬁx is used for spectral layers, and T preﬁx for the temporal ones.
Frequency-wise convolutions
In the spectral branch, we use frequency-wise convolutions, dividing the number of frequency
bins by 4 with every layer. For simplicity we drop the highest bin, giving 2048 frequency
bins after the STFT. The input of the 5th layer has 8 frequency bins, which we reduce to 1
with a convolution with a kernel size of 8 and no padding. It has been noted that unlike the
time axis, the distribution of musical signals is not truely invariant to translation along the
frequency axis. Instruments have speciﬁc pitch range, vocals have well deﬁned formants etc.
To account for that, Isik et al. (2020) suggest injecting an embedding of the frequency before
applying the convolution. We use the same approach, with the addition that we smooth the
initial embedding so that close frequencies have similar embeddings. We inject this embedding
just before the second encoder layer. We also investigated using speciﬁc weights for diﬀerent
frequency bands. This however turned out more complex for a similar result.
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 4
Spectrogram representation
We investigated both with representing the spectrogram as complex numbers (Choi et al., 2020)
or as amplitude spectrograms. We also experimented with using Wiener ﬁltering (Nugraha
et al., 2016), using Open-Unmix diﬀerentiable implementation (Stöter et al., 2019), which uses
an iterative procedure. Using more iterations at evaluation time is usually optimal, but sadly
doesn’t work well with the hybrid approach, as changing the spectrogram output, without the
waveform output being able to adapt will drastically reduce the SDR, and using a high number
of iterations at train time is prohibitively slow. In all cases, we diﬀerentiably transform the
spectrogram branch output to a waveform, summed to the waveform branch output, and the
ﬁnal loss is applied in the waveform domain.
Compressed residual branches
The original Demucs encoder layer is composed of a convolution with kernel size of 8 and stride
of 4, followed by a ReLU, and of a convolution with kernel size of 1 followed by a GLU. Between
those two convolutions, we introduce two compressed residual branches, composed of dilated
convolutions, and for the innermost layers, a biLSTM with limited span and local attention.
Remember that after the ﬁrst convolution of the 5th layer, the temporal and spectral branches
have the same shape. The 5th layer of each branch actually only contains this convolution,
with the compressed residual branch and 1x1 convolution being shared.
Inside a residual branch, all convolutions are with respect to the time dimension, and diﬀerent
frequency bins are processed separately. There are two compressed residual branch per encoder
layer. Both are composed of a convolution with a kernel size of 3, stride of 1, dilation of 1
for the ﬁrst branch and 2 for the second, and 4 times less output dimensions than the input,
followed by layer normalization (Ba et al., 2016) and a GELU activation.
For the 5th and 6th encoder layers, long range context is processed through a local attention
layer (see deﬁnition hereafter) as well as a biLSTM with 2 layers, inserted with a skip connection,
and with a maximum span of 200 steps. In practice, the input is splitted into frames of 200
time steps, with a stride of 100 steps. Each frame is processed concurrently, and for any time
step, the output from the frame for which it is the furthest away from the edge is kept.
Finally, and for all layers, a ﬁnal convolution with a kernel size of 1 outputs twice as many
channels as the input dimension of the residual branch, followed by a GLU. This output is then
summed with the original input, after having been scaled through a LayerScale layer (Touvron
et al., 2021), with an initial scale of 1e−3. A complete representation of the compressed
residual branches is given on 2.
Local attention
Local attention builds on regular attention (Vaswani et al., 2017) but replaces positional
embedding by a controllable penalty term that penalizes attending to positions that are far
away. Formally, the attention weights from position i to position j is given by
wi,j = softmax(QT
i Kj −
4∑
k=1
kβi,k|i − j|)
where Qi are the queries and Kj are the keys. The values βi,k are obtained as the output of
a linear layer, initialized so that they are initially very close to 0. Having multiple βi,k with
diﬀerent weights k allows the network to eﬃciently reduce its receptive ﬁeld without requiring
βi,k to take large values. In practice, we use a sigmoid activation to derive the values βi,k.
Interestingly, a similar idea has been developed in NLP (Press et al., 2021), although with a
ﬁxed penalty rather than a dynamic and learnt one done here.
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 5
GELU(Conv1d(Cin, Cout, K = 8, S = 4))
GELU(LN(Conv1d(Cout, Cout/4, K = 3, D = 1)))
BiLSTM(layers = 2, span = 200)
LocalAttention(heads = 4) }
if i ∈ {5, 6}
GLU(LN(Conv1d(Cout/4, 2 · Cout, K = 1)))
LayerScale(init = 1e−3)
GELU(LN(Conv1d(Cout, Cout/4, K = 3, D = 2)))
BiLSTM(layers = 2, span = 200)
LocalAttention(heads = 4) }
if i ∈ {5, 6}
GLU(LN(Conv1d(Cout/4, 2 · Cout, K = 1)))
LayerScale(init = 1e−3)
GLU(Conv1d(Cout, 2 · Cout, K = 1, S = 1))
Decoderi
Encoderi−1 or input
Encoderi+1
Figure 2: Representation of the compressed residual branches that are added to each encoder layer.
For the 5th and 6th layer, a BiLSTM and a local attention layer are added.
Stabilizing training
We observed that Demucs training could be unstable, especially as we added more layers
and increased the training set size with 150 extra songs. Loading the model just before its
divergence point, we realized that the weights for the innermost encoder and decoder layers
would get very large eigen values.
A ﬁrst solution is to use group normalization (with 4 groups) just after the non residual
convolutions for the layers 5 and 6 of the encoder and the decoder. Using normalization on all
layers deteriorates performance, but using it only on the innermost layers seems to stabilize
training without hurting performance. Interestingly, when the training is stable (in particular
when trained only on MusDB), using normalization was at best neutral with respect to the
separation score, but never improved it, and considerably slowed down convergence during the
ﬁrst half of the epochs. When the training was unstable, using normalization would improve
the overall performance as it allows the model to train for a larger number of epochs.
A second solution we investigated was to use singular value regularization (Yoshida and Miyato,
2017). While previous work used the power method iterative procedure, we obtained better
and faster approximations of the largest singular value using a low rank SVD method (Halko
et al., 2011). This solution has the advantage of always improving generalization, even when
the training was already stable. Sadly, it was not suﬃcient on its own to remove entirely
instabilities, but only to reduce them. Another down side was the longer training time due to
the extra low rank SVD evaluation. In the end, in order to both achieve the best performance
and remove entirely training instabilities, the two solutions were combined.
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 6
Experimental Results
Important: The results presented in this section are results obtained as part of the MDX
challenge. We provide easier to reproduce results and detailed ablation that were conducted
after the challenge in the Section "Reproducibility and Ablation" hereafter.
Datasets
The 2021 MDX challenge (Mitsufuji et al., 2021) oﬀered two tracks: Track A, where only
MusDB HQ (Raﬁi et al., 2019) could be used for training, and Track B, where any data could
be used. MusDB HQ, released under mixed licensing1 is composed of 150 tracks, including 86
for the train set, 14 for the valid, and 50 for the test set. For Track B, we additionally trained
using 150 tracks for an internal dataset, and repurpose the test set of MusDB as training
data, keeping only the original validation set for model selection. Models are evaluated either
through the MDX AI Crowd API2, or on the MusDB HQ test set.
Realistic remix of tracks
We achieved further gains, by ﬁne tuning the models on a speciﬁcally crafted dataset, and with
longer training samples (30 seconds instead of 10). This dataset was built by combining stems
from separate tracks, while respecting a number of conditions, in particular beat matching and
pitch compatibility. Note that training from scratch on this dataset led to worse performance,
likely because the model could rely too much on melodic structure, while random remixing
forces the model to separate without this information.
We use librosa (McFee et al., 2015) for both beat tracking and tempo estimation, as well as
chromagram estimation. Beat tracking is applied only on the drum source, while chromagram
estimation is applied on the bass line. We aggregate the chromagram over time to a single
chroma distribution and ﬁnd the optimal pitch shift between two stems to maximize overlap
(as measured by the L1 distance). We assume that the optimal shift for the bass line is the
same for the vocals and accompaniments. Similarly, we align the tempo and ﬁrst beat. In
order to limit artifacts, we only allow two stems to blend if they require less than 3 semi-tones
of shift and 15% of tempo change.
Metrics
The MDX challenge introduced a novel Signal-To-Distortion measure. Another SDR measure
existed, as introduced by Vincent et al. (2006). The advantage of the new deﬁnition is its
simplicty and fast evaluation. The new deﬁnition is simply deﬁned as
SDR = 10 log10
∑
n‖s(n)‖2 + 
∑
n‖s(n) − ˆs(n)‖2 +  , (1)
where s is the ground truth source, ˆs the estimated source, and n the time index. In order to
reliably compare to previous work, we will refer to this new SDR deﬁnition as nSDR, and to
the old deﬁnition as SDR. Note that when using nSDR on the MDX test set, the metric is
deﬁned as the average across all songs. The evaluation on the MusDB test set follows the
traditional median across the songs of the median over all 1 second segments of each song.
Models
The model submitted to the competitions were actually bags of 4 models. For Track A, we
had to mix hybrid and non hybrid Demucs models, as the hybrid ones were having worse
performance on the bass source. On Track B, we used only hybrid models, as the extra training
1https://github.com/sigsep/website/blob/master/content/datasets/assets/tracklist.csv
2https://www.aicrowd.com/challenges/music-demixing-challenge-ismir-2021
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 7
data allowed them to perform better for all sources. Note that a mix of Hybrid models using
CaC or real masking were used, mostly because it was too costly to reevaluate all models for
the competition. For details on the exact architecture and hyper-parameter used, we refer the
reader to our Github repository facebookresearch/demucs.
For the baselines, we report the numbers from the top participants at the MDX competition
(Mitsufuji et al., 2021). We focus particularly on the KUIELAB-MDX-Net model, which came
in second. This model builds on (Choi et al., 2020) and combines a pure spectrogram model
with the prediction from the original Demucs (Défossez et al., 2019) model for the drums and
bass sources. When comparing models on MusDB, we also report the numbers for some of
the best performing methods outside of the MDX competition, namely D3Net (Takahashi and
Mitsufuji, 2020) and ResUNetDecouple+ (Kong et al., 2021), as well as the original Demucs
model (Défossez et al., 2019). Note that those models were evaluated on MusDB (not HQ)
which lacks the frequency content between 16 kHz and 22kHz. This can bias the metrics.
Results on MDX
We provide the results from the top participants at the MDX competition on Table 1 for the
track A (trained on MusDB HQ only) and on Table 2 for track B (any training data). We
also report for track A the metrics for the Demucs architecture improved with the residual
branches, but without the spectrogram branch. The hybrid approach especially improves the
nSDR on the Other and Vocals source. Despite this improvement, the Hybrid Demucs model
is still performing worse than the KUIELAB-MDX-Net on those two sources. On Track B,
we notice again that the Hybrid Demucs architecture is very strong on the Drums and Bass
source, while lagging behind on the Other and Vocals source.
Table 1: Results of Hybrid Demucs on the MDX test set, when trained only on MusDB (track A)
using the nSDR metric.
Method All Drums Bass Other Vocals
Hybrid Demucs 7.33 8.04 8.12 5.19 7.97
KUIELAB-MDX-Net 7.24 7.17 7.23 5.63 8.90
Music_AI 6.88 7.37 7.27 5.09 7.79
Table 2: Results of Hybrid Demucs on the MDX test set, when trained with extra training (track B)
using the nSDR metric.
Method All Drums Bass Other Vocals
Hybrid Demucs 8.11 8.85 8.86 5.98 8.76
KUIELAB-MDX-Net 7.37 7.55 7.50 5.53 8.89
AudioShake 8.33 8.66 8.34 6.51 9.79
Results on MusDB
We show on Table 3 the SDR metrics as measured on the MusDB dataset. Again, Hybrid
Demucs achieves the best performance for the Drums and Bass source, while improving quite
a lot over waveform only Demucs for the Other and Vocals, but not enough to surpasse
KUIELAB-MDX-Net, which is purely spectrogram based for those two sources. Interestingly,
the best performance on the Vocals source is also achieved by ResUNetDecouple+ (Kong
et al., 2021), which uses a novel complex modulation of the input spectrogram.
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 8
Human evaluations
We also performed Mean Opinion Score human evaluations. We re-use the same protocol as
in (Défossez et al., 2019): we asked human subjects to evaluate a number of samples based
on two criteria: the absence of artifacts, and the absence of bleeding (contamination). Both
are evaluated on a scale from 1 to 5, with 5 being the best grade. Each subject is tasked with
evaluating 25 samples of 12 seconds, drawn randomly from the 50 test set tracks of MusDB.
All subjects have a strong experience with music (amateur and professional musicians, sound
engineers etc). The results are given on Table 4 for the quality, and 5 for the bleeding. We
observe strong improvements over the original Demucs, although we observe some regression
on the bass source when considering quality. The model KUIELAb-MDX-Net that came in
second at the MDX competition performs the best on vocals. The Hybrid Demucs architecture
however reduces by a large amount bleeding across all sources.
Table 3: Comparison on the MusDB (HQ for Hybrid Demucs) test set, using the original SDR metric.
This includes methods that did not participate in the competition. “Mode” indicates if the waveform
(W) or spectrogram (S) domain is used. Model with a “*” were evaluated on MusDB HQ.
Method Mode All Drums Bass Other Vocals
Hybrid Demucs* S+W 7.68 8.24 8.76 5.59 8.13
Demucs v2 W 6.28 6.86 7.01 4.42 6.84
KUIELAB-MDX-Net* S+W 7.47 7.20 7.83 5.90 8.97
D3Net S 6.01 7.01 5.25 4.53 7.24
ResUNetDecouple+ S 6.73 6.62 6.04 5.29 8.98
Table 4: Mean Opinion Score results when asking to rate the quality and absence of artifacts in the
generated samples, from 1 to 5 (5 being the best grade). Standard deviation is around 0.15.
Method All Drums Bass Other Vocals
Ground Truth 4.12 4.12 4.25 3.92 4.18
Hybrid Demucs 2.83 3.18 2.58 2.98 2.55
KUIELAB-MDX-Net 2.86 2.70 2.68 2.99 3.05
Demucs v2 2.36 2.62 2.89 2.31 1.78
Table 5: Mean Opinion Score results when asking to rate the absence of bleeding between the sources,
from 1 to 5 (5 being the best grade). Standard deviation is around 0.15.
Method All Drums Bass Other Vocals
Ground Truth 4.40 4.51 4.52 4.13 4.43
Hybrid Demucs 3.04 2.95 3.25 3.08 2.88
KUIELAB-MDX-Net 2.44 2.23 2.19 2.64 2.66
Demucs v2 2.37 2.24 2.96 1.99 2.46
Reproducibility and Ablation
In this section, we provide ablation of the performance of the model, as well as a simpler setup
for reproducing the performance of the model submitted to MDX Track A. Note that the
numbers and analysis presented here might diﬀer slightly from the ones presented up to now,
and should be preferred when referring to this work.
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 9
Reproducibility
The model submitted to the MDX competition Track A used heterogeneous conﬁgurations, as
we used any model that was suﬃciently trained at any given time. This led to a complex bag
of 4 models, some of which were used only for some sources. While suitable for a competition,
such a complex model does get in the way of reproducing easily the performance achieved.
The training grids for all the models presented in this section can be found on GitHub repository
facebookresearch/demucs, in the ﬁle demucs/grids/repro.py, and demucs/grids/repro_
ft.py for the ﬁne tuned models.
We reproduced the performance of the model submitted to Track A by training two time
domain Demucs (with the residual branches depicted on Figure 2), and two hybrid Demucs
using Complex-As-Channels representation. All four models were trained for 600 epochs, using
the SVD penalty and exponential moving average on the weights. Within each domain, only
the random seed changes between the two models. The four models were ﬁned tuned on the
realistic remix of tracks dataset. The predictions of the four models are averaged into the ﬁnal
prediction, with equal weights over all sources.
As a ﬁrst ablation, we also form one bag composed of the two time domain only models, and
another bag with the two hybrid models only. For fairness, we evaluate each model twice with
a random shift, which is known to improve the performance (Défossez et al., 2019). We also
compare to the original Demucs model, retrained on MusDB HQ, using 10 random shifts, as
done in (Défossez et al., 2019).
The results are presented on Table 6. We reach an overall SDR of 7.64 dB, just 0.04 dB of
the model submitted to MDX Track A. We notice a diﬀerence in performance between time
only, and hybrid only bags only for the bass source and the other source. We still decided
to use the same weights over all sources for each type of model for simplicity. We can see
the advantage of averaging multiple models, as the combination of both time only and hybrid
only model surpasses either ones individually for instance on the drums or the vocals sources.
Note however that when training with extra training data, e.g. for the MDX Track B models,
the hybrid models were always better than the time only ones.
Table 6: Comparison on the MusDB HQ test set, using the original SDR metric of diﬀerent bags of
models, as well as with the original Demucs v2 model retrained on MusDB HQ. “Mode” indicates if
the waveform (W) or spectrogram (S) domain is used.
Method Mode All Drums Bass Other Vocals
Bag time + hybrid S+W 7.64 8.12 8.43 5.65 8.35
Bag time only W 7.27 7.57 8.38 5.17 7.96
Bag hybrid only S+W 7.34 7.96 7.85 5.63 7.95
Demucs v2 HQ W 6.17 6.54 7.08 4.21 6.85
Ablation
We report on Table 7 a short ablation study of the model. We start from a time only improved
Demucs, e.g. trained with residual branches, local attention and svd penalty. We can ﬁrst
oberve the eﬀect of ﬁne tuning on a set of realistic remixes, which improves by 0.3 dB the
SDR overall. Further gains are achieved using the bagging. Using Exponential Moving Average
on the weights improves the SDR by 0.2dB. The eﬀect of the SVD penalty is more contrasted,
with on overall gain of 0.1dB, mainly due to the improved vocals (+0.7 dB), but with a
deterioration on the drums source (-0.4 dB).
Finally, removing the LSTM or the local attention in the residual branches lead to a strong
decrease of the SDR. Interestingly, the local attention is the most important, despite the
absence of positional embedding. One decision taken during the challenge was to switch to
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 10
GELU instead of ReLU. The ablation indicates that no real gain is achieved here.
Table 7: Ablation study, all models are trained and evaluated on MusDB HQ. The base model is a
time only improved Demucs, with local attention, residual branches and svd penalty. Note that we
report single model performance instead of bags of model.
Model All Drums Bass Other Vocals
Improved Time Demucs 6.83 7.06 7.78 4.81 7.65
+ ﬁne tuning 7.11 7.42 8.18 5.08 7.75
+ ﬁne tuning and bagging 7.27 7.57 8.38 5.17 7.96
- LSTM in branch 6.44 6.66 6.68 4.89 7.54
- Local Attention 6.29 6.39 6.76 4.68 7.33
- SVD penalty 6.73 7.45 8.01 4.48 6.98
- EMA on weights 6.63 6.99 7.36 4.74 7.43
- GELU + ReLU 6.84 7.19 7.81 4.73 7.63
Conclusion
We introduced a number of architectural changes to the Demucs architecture that greatly
improved the quality of source separation for music. On the MusDB HQ benchark, the gain is
around 1.4 dB. Those changes include compressed residual branches with local attention and
chunked biLSTM, and most importantly, a novel hybrid spectrogram/temporal domain U-Net
structure, with parallel temporal and spectrogram branches, that merge into a common core.
Those changes allowed to achieve the ﬁrst rank at the 2021 Sony Music DemiXing challenge,
and translated into strong improvements of the overall quality and absence of bleeding between
sources as measured by human evaluations. For all its gain, one limitation of our approach is
the increased complexity of the U-Net encoder/decoder, requiring careful alignmement of the
temporal and spectral signals through well shaped convolutions.
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 11
References
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoﬀrey E Hinton. Layer normalization. arXiv preprint
arXiv:1607.06450, 2016.
Woosung Choi, Minseok Kim, Jaehwa Chung, Daewon Lee, and Soonyoung Jung. Investigating u-nets
with various intermediate blocks for spectrogram-based singing voice separation. In ISMIR, editor,
21th International Society for Music Information Retrieval Conference, 2020.
Woosung Choi, Minseok Kim, Jaehwa Chung, and Soonyoung Jung. Lasaft: Latent source attentive
frequency transformation for conditioned source separation. In IEEE International Conference on
Acoustics, Speech and Signal Processing (ICASSP), 2021.
Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated
convolutional networks. In Proceedings of the International Conference on Machine Learning, 2017.
Alexandre Défossez, Nicolas Usunier, Léon Bottou, and Francis Bach. Music source separation in the
waveform domain. arXiv preprint arXiv:1911.13254, 2019.
Nathan Halko, Per-Gunnar Martinsson, and Joel A Tropp. Finding structure with randomness:
Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review, 53(2):
217–288, 2011.
Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415,
2016.
Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8):
1735–1780, 1997.
Umut Isik, Ritwik Giri, Neerad Phansalkar, Jean-Marc Valin, Karim Helwani, and Arvindh Krishnaswamy.
Poconet: Better speech enhancement with frequency-positional embeddings, semi-supervised
conversational data, and biased loss. arXiv preprint arXiv:2008.04470, 2020.
Andreas Jansson, Eric Humphrey, Nicola Montecchio, Rachel Bittner, Aparna Kumar, and Tillman
Weyde. Singing voice separation with deep u-net convolutional networks. In ISMIR, 2017.
Qiuqiang Kong, Yin Cao, Haohe Liu, Keunwoo Choi, and Yuxuan Wang. Decoupling magnitude and
phase estimation with deep resunet for music source separation. In 22th International Society for
Music Information Retrieval Conference, 2021.
Kundan Kumar, Rithesh Kumar, Thibault de Boissiere, Lucas Gestin, Wei Zhen Teoh, Jose Sotelo,
Alexandre de Brébisson, Yoshua Bengio, and Aaron C Courville. Melgan: Generative adversarial
networks for conditional waveform synthesis. In Advances in Neural Information Processing Systems,
2019.
Francesc Lluís, Jordi Pons, and Xavier Serra. End-to-end music source separation: is it possible in the
waveform domain? arXiv preprint arXiv:1810.12, 2018.
Yi Luo and Nima Mesgarani. Conv-tasnet: Surpassing ideal time–frequency magnitude masking for
speech separation. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019.
Yi Luo, Zhuo Chen, and Takuya Yoshioka. Dual-path rnn: eﬃcient long sequence modeling for
time-domain single-channel speech separation. In IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), 2020.
Brian McFee, Colin Raﬀel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol
Nieto. librosa: Audio and music signal analysis in python. In Proceedings of the 14th python in
science conference, 2015.
Yuki Mitsufuji, Giorgio Fabbro, Stefan Uhlich, and Fabian-Robert Stöter. Music demixing challenge
2021. arXiv preprint arXiv:2108.13559, 2021.
Aditya Arie Nugraha, Antoine Liutkus, and Emmanuel Vincent. Multichannel music separation with
deep neural networks. In Signal Processing Conference (EUSIPCO). IEEE, 2016.
Hybrid Spectrogram and Waveform Source Separation. Proceedings of the MDX Workshop, 2021. 12
Jordi Pons, Santiago Pascual, Giulio Cengarle, and Joan Serrà. Upsampling artifacts in neural audio
synthesis. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
2021.
Oﬁr Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables
input length extrapolation. 2021.
Zafar Raﬁi, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner.
The musdb18 corpus for music separation, 2017.
Zafar Raﬁi, Antoine Liutkus, Fabian-Robert Stöter, Stylianos Ioannis Mimilakis, and Rachel Bittner.
MUSDB18-HQ - an uncompressed version of musdb18, December 2019. URL https://doi.org/10.
5281/zenodo.3338373.
Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical
image segmentation. In International Conference on Medical image computing and computer-assisted
intervention, 2015.
Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi, and Yuki Mitsufuji. All for one and one for all:
Improving music separation by bridging networks. 2020.
Daniel Stoller, Sebastian Ewert, and Simon Dixon. Wave-u-net: A multi-scale neural network for
end-to-end audio source separation. arXiv preprint arXiv:1806.03185, 2018.
F.-R. Stöter, S. Uhlich, A. Liutkus, and Y. Mitsufuji. Open-unmix - a reference implementation for
music source separation. Journal of Open Source Software, 2019. doi: 10.21105/joss.01667.
Fabian-Robert Stöter, Antoine Liutkus, and Nobutaka Ito. The 2018 signal separation evaluation
campaign. In 14th International Conference on Latent Variable Analysis and Signal Separation,
2018.
Naoya Takahashi and Yuki Mitsufuji. D3net: Densely connected multidilated densenet for music source
separation. arXiv preprint arXiv:2010.01733, 2020.
Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, and Hervé Jégou. Going
deeper with image transformers. arXiv preprint arXiv:2103.17239, 2021.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in neural information processing
systems, 2017.
Emmanuel Vincent, Rémi Gribonval, and Cédric Févotte. Performance measurement in blind audio
source separation. IEEE Transactions on Audio, Speech and Language Processing, 2006.
Yuichi Yoshida and Takeru Miyato. Spectral norm regularization for improving the generalizability of
deep learning. arXiv preprint arXiv:1705.10941, 2017.
Fisher Yu and Vladlen Koltun. Multi-scale context aggregation by dilated convolutions. In ICLR, 2016