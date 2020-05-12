from pydub.utils import make_chunks
import os
import os.path as path
import librosa.display
import pylab
import numpy as np
from pydub import AudioSegment

list = os.listdir(path.abspath(path.join(__file__ , "../dataset/testing/input/")))
number_files = len(list)

iCount = 0

for x in list:

    # Absolute paths
    outPath = (os.getcwd() + "/dataset/testing/output/{0}/".format(iCount)).replace("\\", "/")
    spectroPath = (os.getcwd() + "/dataset/testing/spectrogram/{0}/".format(iCount)).replace("\\", "/")

    # Create directories on specified absolute paths
    makePathOut = os.mkdir(outPath)
    makePathSpectro = os.mkdir(spectroPath)

    # Get audio and split
    myaudio = AudioSegment.from_file(path.join(__file__ , "../dataset/testing/input/")+ list[iCount], "wav")
    chunk_length_ms = 1000
    chunks = make_chunks(myaudio, chunk_length_ms)

    for iSplit, chunk in enumerate(chunks):

        chunk_name = "chunkie{0}.wav".format(iSplit)

        print("exporting", chunk_name)
        chunk.export(outPath + chunk_name, format="wav")

        sig, fs = librosa.load(outPath + chunk_name)
        pylab.axis('off')
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])

        S = librosa.feature.melspectrogram(y=sig, sr=fs)

        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))

        pylab.savefig(os.getcwd() + "/dataset/testing/spectrogram/{0}".format(iCount) + "/{0}".format(iSplit), dpi=400, bbox_inches=None, pad_inches=0)

    iCount = iCount + 1
