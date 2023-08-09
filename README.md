# TFNOTrainingSOMAData
<h1>TFNO Training SOMA Data README</h1>
This repository relies on files provided in two other repositories, namely:
<ul>
  <li>DeepAdjoint from user iamyixuan - https://github.com/iamyixuan/DeepAdjoint</li>
  <li>neuraloperator from user JeanKossaifi - https://github.com/neuraloperator/neuraloperator</li>
</ul>
The purpose of this repository is to train a fourier neural operator with a tucker factorization on generated SOMA data. The data set for this repository is not currently provided in this repository and is only accessible to those working on the ImPACTS project. 
<br />
A few notes:
<ul>
  <li>User should change the data_path variable on line 23 of SOMAforward.py to the location of their SOMA data set. </li>
  <li>When testing, user should set the checkpoint variable on line 40 of SOMAforward.py to the path of where the most recent checkpoint is stored. </li>
  <li>When saving the results of testing, user should name and file path of where they want their testing results to be saved on line 41 of SOMAforward.py. </li>
  <li>If the user wants to pick up training from a check point, they can uncomment out line 19 in SOMAforward.py and set it to the location of the checkpoint file. </li>
</ul>
