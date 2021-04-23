<img src="https://github.com/ShenLab/gMVP/blob/main/docs/_static/mvp_logo_nobackground_24kb.png?raw=true" width="256" alt="gMVP">

## Predicting functional effect of missense variants using graph attention neural networks

gMVP is a graph attention neural network model designed to effectively represent or learn the representation of protein sequence and structure context to improve missense variant prediction of disease impact. 
The main component is a graph with nodes capturing predictive features of amino acids and edges weighted by coevolution strength, which allows for effective pooling of information from functionally correlated positions. Evaluated by deep mutational scan data, gMVP outperforms published methods in identifying damaging variants in TP53, PTEN, BRCA1, and MSH2. It achieves the best separation of de novo variants in neurodevelopmental disorder cases from controls. Finally, the model supports transfer learning to optimize gain- and loss-of-function predictions in sodium and calcium channels. In summary, we demonstrate that gMVP can improve interpretation of missense variants in genetic studies.

### Precomputed gMVP scores
We have generated gMVP scores for all possible missense variants in canonical transcripts on human hg38 which can be acessed through
https://www.dropbox.com/s/nce1jhg3i7jw1hx/gMVP.2021-02-28.csv.gz?dl=0.

### Required libraries

We implemented gMVP using Tensorflow2. The following libraries are required: tensorflow (>= 2.2), numpy, scipy, json, and scikit-learn.



