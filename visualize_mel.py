import matplotlib.pyplot as plt
import librosa
import numpy as np
# dir = '/home/datasets/Speaker_Recognition/train_wav/VAD_1/mel/Subset'
# mel = np.load(dir + '/' + '6362-f-sre2006-jaqm-A_vad_mel.npy')
mel = np.load('6362-f-sre2006-jaqm-A_vad_mel.npy')
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()

from matplotlib import mpl,pyplot
import numpy as np

# make values from -5 to 5, for this example
# zvals = np.random.rand(100,100)*10-5

# make a color map of fixed colors
# cmap = mpl.colors.ListedColormap(['blue','black','red'])
# bounds=[-6,-2,2,6]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# tell imshow about color map so that only set colors are used
img = plt.imshow(mel,interpolation='nearest')

# make a color bar
# pyplot.colorbar(img,cmap=cmap,
  #               norm=norm,boundaries=bounds,ticks=[-5,0,5])

pyplot.show()