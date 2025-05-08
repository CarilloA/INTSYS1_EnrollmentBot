

****TO BEGIN****
(0) To create venv folder (If meron na, kahit wag na toh) type in terminal: python -m venv venv
(1) To activate venv, Type into terminal: venv\Scripts\activate
(2) To run it into browser, Type into terminal: python app.py
(3) To open it -> Go to browser -> Type in the url: http://127.0.0.1:5000/
(4) To exit from developer server -> Go to terminal -> ctrl+c
(5) To deactivate the venv environment, Type: deactivate



****To Feed Data****
(1) Go to intents.json file and add additional data there
(2) After finish adding all data -> Type into terminal: python train.py
(3) Open chatbot in the browser to check if data feeds successfully


****To Cluster Unkown Queries****
(1) To cluster all unknown_queries -> Type into terminal: python cluster_unknown_queries.py
(1.1) This will group all similar queries into cluster_unknown.json file for easier analysis


****For Error****
cannot be loaded because running scripts is disabled on this system.

Solution: 
(1) Click Start, type PowerShell.
(2) Right-click Windows PowerShell and choose Run as administrator.
(3) Type in the terminal: Set-ExecutionPolicy RemoteSigned
(4) Type in the terminal to confirm: Y
(5) Try to activite again by going to the steps in: ****TO BEGIN****

****For Error****
Resource punkt_tab not found

Solution: 
(1) Go to gdrive Tokenizers folder and download it
(2) Go to the Tokenizers location in your computer (for me its: C:\Users\juanp\AppData\Roaming\nltk_data)
(3) Paste/move here the downloaded tokenizers data
(4) Try to train again, Type: python train.py
(5) Then run it into browser, Type: python app.py
(6) To open it -> Go to browser -> Type in the url: http://127.0.0.1:5000/

******

NOTE: Your terminal should be in (venv) mode, it should be activated

Sample Terminal: (venv) PS C:\Users\juanp\OneDrive\Desktop\PythonProjects\INTSYS1_EnrollmentBot> python app.py
Note that (venv) is activated at the leftmost part of your path

NOTE: If you made changes to intents.json, retrain it again before running the browser by typing: python train.py



****INSTALLATIONS IF NEEDED****

python -m venv venv

python -m pip install torch torchvision torchaudio

python -m pip install numpy nltk

pip install flask

pip install scikit-learn
