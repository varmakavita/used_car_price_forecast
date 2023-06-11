README 

DESCRIPTION

This application is designed to predict the best value of your car based on 5 different predictors. There are three components
to this program, two backend pieces ML model & Deep Learning model and a Frontend component that lets the user interact with 
the model. The backend consists of models that need to be trained and pickled and this pickle file can then be passed into 
the code for the frontend. The frontend is a raw HTML, CSS form paired with a python Flask based API, with two endpoints. The
/predict endpoint is connected to the frontend to process the form data, while the /api endpoint is configured to be called from 
a direct POST request to test the model. 

Deep learning model is not integrated to the frontend, but can be used to predict the Make, Model & Year of a new car image of size 224 x 224. 





INSTALLATION

	FRONT END
	- Make sure Python3 is installed. Frontend app.py is built on top of python3
	- Install Fask 
		run following command in terminal: pip3 install flask

	ML MODEL
	- Install and import the following libraries
		import numpy as np
		import pandas as pd
		from datetime import datetime
		import uszipcode
		from uszipcode import SearchEngine
		import gc
		import os

		import matplotlib.pyplot as plt
		import seaborn as sns
		from sklearn.tree import export_graphviz

		from sklearn.preprocessing import MinMaxScaler
		from sklearn.model_selection import train_test_split
		from sklearn.model_selection import cross_val_score
		from sklearn.metrics import mean_squared_error
		from sklearn.metrics import r2_score
		from sklearn.inspection import permutation_importance
		from sklearn.model_selection import KFold
		from sklearn.model_selection import cross_val_score

		from statsmodels.stats.outliers_influence import variance_inflation_factor
		from statsmodels.tools.tools import add_constant
		from scipy import stats

		from sklearn.linear_model import LinearRegression
		from sklearn.ensemble import RandomForestRegressor
		from sklearn.tree import DecisionTreeRegressor
		from sklearn.model_selection import GridSearchCV
		from sklearn import tree

		import fredapi as fa


	DL MODEL
	-Python Version - 3.6.9, Linux 18.04
	-Install the following python packages (most of them can be install by using pip)
		- sklearn
		- progressbar
		- pickle
		- matplotlib
		- numpy
		- shutil
		- cv2
		- csv
		- imutils
		- requests
		- pandas
		- logging
		- mxnet  --follow the instructions in this link to install mxnet, section#4 : https://pyimagesearch.com/2017/11/13/how-to-install-mxnet-for-deep-learning/
	




EXECUTION

	ML model
	-------------------
	Run code in Jupyter Notebook
	Output will be a pickled model that can be interacted with through its predict function. 
	The Front End's inputs correspond to the predict function, which will create the prediction.


	Frontend 
	-------------------
	1. Update line 20 of the app.py file with the name of the pickle file created in backend step
	2. Start Python application 
		b. Run following command in terminal: python3 app.py 
	3. Copy url from output of previous command. Ex: "http://127.0.0.1:5001"
	4. Paste this url into your browser and you should see a form pop up
	5. Enter vehicle information and click "Predict" to see the output.
	6. Click "Reset Form" to reset the form and try a new prediction


	DL Model
	-------------------
	** Dataset Creation

	Follow the steps to create a Used car images dataset
	Check "./config/cars_config.py" file to modify any parameters (if needed)
	
	1) Run download_images.py  --This will download the images using the links provided in "car_image_links.csv" file.
	2) Run data_cleaning.py   --Clean the downlaoded data and segregate them to their respective classes
	3) Run create_dataset.py  -- Creates the dataset based on the threshold and number of images per class provided in the config file
	4) build_dataset.py       --creates train, test, val lists and a pikle file with encoded labels
	
	We used the "mxnet" module to train the VGG16 model on the used cars image dataset.
	Record files are used for training purpose.
	Record files for train, val and test should be created. Execute the following commands in the linux terminal.
	
	a) ~/mxnet/bin/im2rec ./lists/train.lst "" ./rec/train.rec resize=256 encoding='.jpg' quality=100 
	b) ~/mxnet/bin/im2rec ./lists/val.lst "" ./rec/val.rec resize=256 encoding='.jpg' quality=100
	c) ~/mxnet/bin/im2rec ./lists/test.lst "" ./rec/val.rec resize=256 encoding='.jpg' quality=100


	** Training the model
	
	1) Run fine_tune_cars.py    --adjust the hyper parameters if required.
	2) Trained models are saved in ./checkpoints folder.
	3) Run plot_log.py to plot the logged training data.
	
	** Testing

	1) Run test_model.py    --imagePath should be provided in Line10. 































