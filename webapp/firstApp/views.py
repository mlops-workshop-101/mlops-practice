from django.shortcuts import render # type: ignore
import yaml, json, os
import joblib
from .models import aids

def index(request):
    return render(request, 'index.html')

def result(request):
    cls = joblib.load("../models/models.joblib")
    list = []
    list.append(int(request.GET['age']))
    list.append(int(request.GET['sex']))
    list.append(float(request.GET['bmi']))
    list.append(int(request.GET['children']))
    list.append(int(request.GET['smoker']))
    list.append(int(request.GET['region']))

    answer = cls.predict([list])

    b = aids(age = int(request.GET['age']), 
    sex = int(request.GET['sex']),
    bmi = int(request.GET['bmi']),
    children = int(request.GET['children']),
    smoker = int(request.GET['smoker']),
    region = int(request.GET['region']))

    return render(request, 'index.html', {'answer': answer[0]})
