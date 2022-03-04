from django.shortcuts import render
from .static.function.test import TestModel
from django.http import HttpResponse
# Create your views here.
def index(request):
    t = TestModel()
    src = request.GET.get('src')
    t.src = src
    result = t.predict()
    context = {
        'src': src,
        'result': result,
    }
    # context = {}
    return render(request,'webs/index.html',context)