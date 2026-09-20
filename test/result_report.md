# Signal Analysis Findings Report

This report summarises what the automated RF analysis assistant found when it examined a single real radio capture, and why it reached its conclusions. The goal here is to describe the **traits of the signal** in plain words rather than raw numbers.

---

## 1. What Was Examined

A short burst of real over-the-air radio from the MIT RF Challenge dataset. The capture is known to be a quadrature-phase-shift-keyed (QPSK) transmission. It was recorded at a high sample rate over about a millisecond and a half.

The assistant carried out the usual sequence of steps on this one file:

- **Load and clean** the recording (remove any DC offset, normalise power).
- **Scan the spectrum** to find where energy lives and how strong the signal is relative to the noise.
- **Measure the signal's shape** — bandwidth, envelope behaviour, phase structure, symmetry.
- **Guess the modulation** from those traits.
- **Demodulate** — actually pull symbols out of the waveform.
- **Attempt to decode** a data frame and verify it with a checksum.
- **Rank and explain** the most likely interpretations, quoting the evidence.

---

## 2. What the Signal Looks Like

| Trait | What was observed |
|-------|-------------------|
| Strength | A clearly present signal, moderately strong above the noise floor |
| Footprint | Occupies only a small slice of the recorded spectrum — a narrow-band signal inside a wide capture band |
| Structure | Arrives in several distinct energy bursts rather than one continuous wall of energy |
| Symmetry | Shows four-fold rotational symmetry in the constellation — the signature of four equally-spaced phase states |
| Envelope | Single amplitude shell with an occasional second level, consistent with a phase-encoded waveform |
| Peak-to-average | A modest, multi-level-style ratio — louder peaks than a pure constant-envelope tone, but far quieter than heavy amplitude modulation |

Together these traits read as **digital phase modulation with four states** — exactly what QPSK looks like.

---

## 3. Modulation Identification

The classifier looked at the measured traits and placed QPSK clearly at the top:

- **QPSK — strongest match, high confidence.**
- **16-QAM — the nearest rival**, but it lost because the signal shows a single amplitude shell with only mild excursions, while 16-QAM would show several clearly separated amplitude rings.
- Everything else (simple two-state BPSK, frequency-shift variants, plain noise) was judged unlikely.

**Why:** the strongest single discriminator was the four-fold phase symmetry combined with a mostly constant envelope. That combination is essentially unique to four-state phase modulation.

---

## 4. Demodulation — Pulling the Symbols Out

Once the assistant committed to QPSK, it synchronised to the carrier, recovered the symbol timing, and pulled out roughly a couple of thousand symbols.

- Symbol timing was successfully established and the residual phase error locked to near zero.
- **However, the symbol quality was poor.** The recovered symbols sit noticeably scattered around their ideal positions. On real, noisy, imperfect hardware captures this is normal, but it means:
  - the waveform is real and physically demodulable, but
  - the exact constellation is fairly smeared — a hard recording to decode cleanly.

The poor symbol quality is partly expected (real radio) and partly the result of a timing-estimation quirk noted in the last section.

---

## 5. Decoding the Data Frame

The assistant then tried to turn the recovered bits into a valid data frame using several candidate decode chains (de-interleaving, forward-error correction, preamble search, and checksum verification).

**Result: no data frame could be verified.** None of the checked candidate chains produced a bit stream whose checksum passed. This is reported honestly — the assistant does not invent a decode when the evidence does not support one.

This is consistent with the poor symbol quality in the previous step: when symbols are this scattered, the bits that come out are too corrupted for the checksum to ever agree, even if the transmission truly is QPSK.

---

## 6. Hypothesis Engine — Ranked Explanation

The hypothesis engine combined all of the above evidence and ranked complete explanations (modulation → demodulation → de-interleaving → error correction → framing). Its verdict:

- **The most supported overall explanation is the QPSK chain**, matching the modulation analysis.
- The 16-QAM interpretation was the leading alternative — the engine explicitly listed what it had considered and why it ranked lower.
- Every top hypothesis carries a natural-language **"why"**, citing the specific traits (four-fold symmetry, single-shell envelope, classifier score, symbol quality).
- The engine also disclosed its **uncertainty**: the ranking is only moderately confident, not a forensic certainty, because the symbol quality is poor and nothing decoded cleanly.

So the honest summary is: *"This is almost certainly QPSK, but on this particular recording we could not recover clean enough symbols to read out a verified data frame."*

---

## 7. Honest Caveats — Two Measurements to Ignore

During the run the assistant produced two readings that it itself should have flagged as suspect:

1. **Symbol-timing estimate.** The automatic timing estimator reported a rate far too fast for a signal whose bandwidth is as narrow as the one measured here. A wide-bandwidth waveform can carry a fast symbol rate; this signal is narrow, so the fast reading is inconsistent and should not be trusted. This misstep also inflated the symbol scatter seen during demodulation.

2. **Frequency-offset estimate.** The initial offset measurement used a method that is only appropriate for narrow continuous tones, not for a wide phase-modulated signal. The resulting offset number is therefore meaningless and was not the reason the demodulator still locked on (the demodulator recovers carrier offset itself).

Both are known limitations of choosing the wrong tool from the toolbox; they are noted here so that the summary reflects what is trustworthy, not simply everything that was printed.

---

## 8. Bottom Line

- The signal is a real, moderately strong, narrow-band digital transmission.
- Its dominant traits — four-fold phase symmetry and a near-constant envelope — point strongly to **QPSK**.
- The assistant ranked QPSK as the most credible explanation with an attached "why" and an honest uncertainty.
- It could **not** extract a verified data frame — the symbols recovered from this particular noisy recording were too scattered for any candidate decode chain to pass a checksum.
- Two auto-measurements (symbol timing and initial frequency offset) were inconsistent with the rest of the analysis and are flagged as untrustworthy.

**Net answer: a confident identification of QPSK, an honest failure on full decoding, and a transparent account of which measurements to trust.**