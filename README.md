# MiniTorch Module 4

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments.

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py project/run_fast_tensor.py project/parallel_check.py tests/test_tensor_general.py


<b>Results from training the Sentiment model on the terminal (Full output can be found on project/SentimentTrainingFinal.txt): </b>

Best Training Accuracy: 88.89%<br>
Best Validation Accuracy: 73%<br>
Mean Validation Accuracy: ~66%<br>
Average time per epoch: 26.6 seconds<br>
Number of Epochs: 87<br>
Best Loss: 8.7<br>

<b>Results from training the Sentiment model on Streamlit using the default parameters:</b>

Best Training Accuracy: 0.889<br>
Best Validation Accuracy: 0.74<br>
Average time per epoch: 36.5 seconds<br>
Number of Epochs: 75<br>
Best Loss: 9.132<br>

<b>Accuracy plot:</b>
<img src="SentimentAccuracies.JPG"> <br>
<b>Loss plot:</b>
<img src="SentimentLoss.JPG"> <br>
<b>Epochs table:</b>
<img src="SentimentTable.JPG"> <br>

<b>Results from training the Multiclass image classification model on the terminal (Full output can be found on project/TrainMnistFinal.txt)</b>

Best result: 16 correct<br>
Convergence result: 15 correct<br>
Number of Epochs: 2<br>
Best Loss: 0.77<br>

<b>Results from training the Multiclass image classification model on streamlit with 100 epochs and learning rate of 0.05:</b>

Best result: 15 correct<br>
Convergence result: 14 correct<br>
Number of Epochs: 100<br>
Average time per epoch: 48.4 seconds<br>
Best Loss: 1.259<br>
<b>Layers plot:<br>
<img src="MnistPicture.JPG"> <br>
Loss plot:<br>
<img src="MnistLoss.JPG"> <br>
Epochs table:<br>
<img src="MnistTable.JPG"> <br></b>