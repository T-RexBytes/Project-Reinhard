import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

samples = ['CommSignal2_demod_train_0000', 'CommSignal3_demod_train_0000', 'EMISignal1_demod_train_0000']
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for i, name in enumerate(samples):
    base = Path('outputs/plot_data') / name
    d = np.load(base / 'psd.npz')
    s = np.load(base / 'spectrogram.npz')
    with open(base / 'metrics.json') as f: m = json.load(f)
    ax = axes[i,0]
    ax.plot(d['frequencies']/1e3, d['psd_db'], linewidth=0.8)
    ax.axhline(m['noise_floor']['noise_floor_db'], color='r', ls='--', lw=1, label='noise floor')
    for seg in m['segmentation']['segments']:
        ax.axvspan(seg['start_freq_hz']/1e3, seg['end_freq_hz']/1e3, alpha=0.15, color='orange')
    ax.set_title(name + " - PSD (" + str(m['segmentation']['num_segments']) + " segs, SNR " + str(round(m['snr']['psd_based_db'],1)) + " dB)")
    ax.set_xlabel('Freq (kHz)'); ax.set_ylabel('PSD (dB)'); ax.legend(fontsize=7)
    ax = axes[i,1]
    im = ax.imshow(s['spectrogram_db'], aspect='auto', origin='lower',
                   extent=[s['time_bins'][0]*1e3, s['time_bins'][-1]*1e3, s['frequencies'][0]/1e3, s['frequencies'][-1]/1e3],
                   cmap='viridis', vmin=-110, vmax=-50)
    ax.set_title(name + " - Spectrogram")
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Freq (kHz)')

plt.tight_layout()
plt.savefig('outputs/plot_data/preview.png', dpi=150)
print('Saved preview.png')
