from sklearn.decomposition import FastICA
# each 'signal' variable is an array. e.g. audio waveform
X = list(zip(signal_1, signal_2, signal_3))

ica = FastICA(n=n_components=3)
components = ica.fit_transform(X)

# components new contains the independent components retrieved via ICA