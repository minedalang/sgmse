The Consonant Challenge Corpus 
Martin Cooke 
Ikerbasque (Basque Science Foundation), Spain 
Odette Scharenborg 
TU Delft, The Netherlands 
Original version: 2008 
First Zenodo release: 12 Nov 2019 
Abstract 
The Consonant Challenge Corpus is a database for human-machine comparisons of consonant 
recognition in quiet and noise. The Corpus consists of audio waveforms for vowel-consonant-vowel 
tokens embedded in different types of masking noise. Training, development and test data is 
provided to support automatic speech recognition systems. Native listener performance data on 
the test set is also provided. 
The Corpus was used in the 2008 Consonant Challenge described in this article: 
Cooke, M., Scharenborg, O. (2008), “The Interspeech 2008 Consonant Challenge”, Proceedings of 
Interspeech, Brisbane, Australia, September 2008. 
Speakers and VCV material 
Twelve female and 12 male native English talkers aged between 18-49 contributed to the corpus. 
Speakers produced each of the 24 consonants / b, d, g, p, t, k, s, ʃ, f, v, ð, θ, ʧ, z, ʒ, h, ʤ, m, n, ŋ, 
w, r, j, l / in nine vowel contexts consisting of all possible combinations of the three vowels / iː / (as 
in “beat”), / uː / (as in “boot”), and / æ / (as in “bat”). Each VCV was produced using both front and 
back stress (e.g. / ‘æ b æ / vs / æ b ‘æ /) giving a total of 24 (speakers) * 24 (consonants) * 2 
(stress types) * 9 (vowel contexts) = 10368 tokens. Pilot listening tests were carried out to identify 
unusable tokens due to poor pronunciations or other problems. 
Data collection and postprocessing 
Recordings were made in an IAC single-walled acoustically isolated booth at the University of 
Sheffield. Speech material was collected from a single Bruel & Kjaer (B & K) type 4190 1/2 -in. 
microphone placed 30 cm in front of the talker. The signal was preamplified by a B & K Nexus 
model 2690 conditioning amplifier prior to digitisation at 50 kHz by a Tucker-Davis Technologies 
System 3 RP2.1 processor. 
Speakers produced VCVs in isolation by reading out tokens presented on a computer screen, and 
were given both verbal and written instructions on how to interpret token names, with a particular 
focus on / ð, θ, ʒ, ʤ /. Speakers ran through a practice with the experimenter before speaking 
alone in the booth. Speakers first produced all VCVs with front stress, followed by end stress. 
Signals were subsequently post-processed by applying the following steps: 

High-pass filtering to remove frequencies below 50 Hz 
Downsampling to 25 kHz using the MATLAB resample function 
Marking endpoints using an algorithm based on the short-term energy contour 
Normalisation to a target RMS of 0.05 
File names indicate the gender of the speaker (F or M), the speaker ID (a number between 1 and 
24), and the stress pattern: a 1 following the name of the VCV in the file name indicates front 
stress, a 2 indicates end stress. For instance, “m25adu2.wav” is the consonant / d / produced in 
the context / æ / _ / iː / by male speaker 25, with end stress. 
Training, development and test sets 
Training material comes from 8 male and 8 female speakers while tokens from the remaining 8 
speakers are used in the independent test set. After removing unusable tokens identified during 
post-processing, the training set consists of 6664 clean tokens. 
Seven tests sets, corresponding to a quiet and 6 noise conditions, are available. Each test set 
contains 16 instances of each of the 24 consonants, for a total of 384 tokens. 
Maskers 
Apart from the noise-free test set, six further tests sets are provided in which VCV tokens are 
mixed with different maskers at the signal-to-noise ratios (SNRs) specified below: 
test set 1: clean (i.e. no noise) 
test set 2:  competing talker     -6 dB  
test set 3:  8-talker babble   -2 dB  
test set 4: speech-shaped noise (SSN)  -6 dB  
test set 5: factory noise       0 dB  
test set 6: modulated SSN    -6 dB  
test set 7: 3-speaker babble      -3  dB 
These noise types provide a challenging and varied range of conditions. SNRs were determined 
using pilot tests with listeners with the goal of producing similar identification scores (~ 65-70%) in 
each noise condition. 
VCV tokens are additively embedded in noise samples of duration 1.2s. The SNR is computed 
token-wise and refers to the SNR in the section where the speech and noise overlap. The time of 
onset of each VCV takes on one of 8 values ranging from 0 to 400 ms. 
In addition, test materials are also available as 2-channel sound files which are identical to the test 
sets except that the the noise and VCV tokens are in separate channels. We have made the test 
material available in this form to support computational models of human consonant perception 
which may wish to make some assumptions about e.g. ideal noise processing, and also to allow for 
the computation of idealised engineering systems (e.g. to determine performance ceilings). 
Perceptual data for native listeners 
Twenty seven native English listeners aged between 18 and 48 who reported no hearing problems 
identified the 384 VCVs of each test set. Listeners were drawn from the staff and students at the 
University of Sheffield and were paid for their participation. Perception tests ran under computer 
control in the IAC booth. Listeners were presented with a GUI in which the 24 consonants were 
represented using both ASCII symbols and with an example word containing the sound. Listeners 

were phonetically-naive and were given instructions as to the meaning of each symbol. They 
underwent a short practice session prior to the main test. Two listeners failed to reach a criterion 
level of 85% in a practice session using clean tokens. Another failed to complete all conditions, 
while a fourth was an outlier on most of the test conditions. Results are reported for the remaining 
23 listeners. For the main test, listeners started with the clean condition. The order of the noisy 
conditions was randomised. 
Native listener results, averaged over consonants and listeners:  
Set %correct std. err.  
1 93.8   0.57 
2  79.5  0.78 
3 76.5  0.79 
4 72.2  0.75 
5 66.7  0.77 
6 79.2  0.61 
7 71.4  0.74 
Baseline recognition system 
The performance of various acoustic features (MFCC, FBANK, MELSPEC, Ratemaps) and 
recogniser architectures (monophone, triphone, gender-dependent/independent) was investigated. 
Two representative combinations were chosen as baselines for the Consonant Challenge: one 
system based on MFCCs, the other on Ratemaps. 
For both systems, 30 models were trained, 24 consonants and two models for each of the three 
vowels – one to model the initial vowel context and one to model the final vowel context of the 
VCV. The maximum number of training items per consonant is 9 (vowel contexts) * 2 (stress 
conditions) * 16 (speakers) = 288 items. 
MFCC-based recognition system. The speech is parameterised with 12 MFCC coefficients and log 
energy and augmented with first and second temporal derivatives resulting in a 39-dimensional 
feature vector. Each of the monophones consists of 3 emitting states with a 24-Gaussian mixture 
output distribution. No silence model and short pause model are employed in this distribution as 
features are end-pointed. The HMMs were trained from a flat start using HTK. 
Ratemap-based recognition system. Ratemaps are a filterbank-based representation based on 
auditory excitation patterns. 64-dimensional feature vectors and the same model architecture as for 
the MFCC-based system were used. 
The overall consonant recognition accuracy for the MFCC-based recognition system is 88.5% on 
Test set 1 (clean). The overall consonant recognition accuracy for the Ratemap-based system on 
Test set 1 is 84.4%. 
Contact 
For further information on the Corpus, please contact either Martin Cooke 
(m.cooke@ikerbasque.org) or Odette Scharenborg (O.E.Scharenborg@tudelft.nl). 
Acknowledgements:  
The Consonant Challenge Corpus was made possible by financial support from the PASCALA 
Network. We also thank Maria Luisa Garcia Lecumberri of the University of the Basque Country, 
Spain and Ning Ma, Youyi Lu and Matt Gibson from the University of Sheffield, UK for useful 
discussions.