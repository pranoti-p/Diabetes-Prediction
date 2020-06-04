from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def diabetes(request):
	"""	
	Reading the training data set
	"""
	dfx=pd.read_csv('static/Diabetes_XTrain.csv')
	dfy=pd.read_csv('static/Diabetes_YTrain.csv')
	X= dfx.values
	Y= dfy.values
	Y=Y.reshape((-1,))

	"""
	Reading data from user
	"""
	value=''
	if request.method=='POST':
		pregnancies=float(request.POST['pregnancies'])
		glucose=float(request.POST['glucose'])
		bloodpressure=float(request.POST['bloodpressure'])
		skinthickness=float(request.POST['skinthickness'])
		bmi=float(request.POST['insulin'])
		insulin=float(request.POST['insulin'])
		pedigree=float(request.POST['pedigree'])
		age=float(request.POST['age'])

		user_data=np.array(
			(pregnancies,glucose,bloodpressure,skinthickness,bmi,insulin,pedigree,age)
		).reshape(1,8)

		knn=KNeighborsClassifier(n_neighbors=3)
		knn.fit(X,Y)
	
		predictions=knn.predict(user_data)
	
		if int(predictions[0])==1:
			value='are diagnosed with'
		elif int (predictions[0])==0:
			value="are not diagnosed with"
	return render(request,
				'diabetes.html',
				{
					'context':value,
					'title':'Diabetes Disease Prediction',
					'active':'btn btn-success peach-gradient text-white',
					'diabetes':True,
					'background':'bg-dark text-white'
				}
				)

def home(request):
	return render(request,'home.html')

"""
Error Page Handling
"""
def handler404(request):
	return render(request,'404.html',status=404)

