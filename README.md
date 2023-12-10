# ML-Stock-Market-Prediction

We recommend using Google Colab to run the code since all requirements come preinstalled. Simply open a new Colab enviornment and copy&paste the commands below. No GPU is required so you can run everything Colab's CPU. The repository includes the results of our last run (logfile and plots). Running the snippet below will produce new results (may be slightly different due to random initialization of weights).

```bash
!git clone https://github.com/nikhilmanda9/ML-Stock-Market-Prediction.git
print("\n")
!python ML-Stock-Market-Prediction/src/main.py

from IPython.display import Image, display
display(Image('ML-Stock-Market-Prediction/plots/original_plot.png'))
display(Image('ML-Stock-Market-Prediction/plots/sequenced_plot.png'))
display(Image('ML-Stock-Market-Prediction/plots/training_errors.png'))
print("\n")
!cat ML-Stock-Market-Prediction/gru_logs.txt
```