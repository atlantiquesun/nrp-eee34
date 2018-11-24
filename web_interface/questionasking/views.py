from django.http import HttpResponse
from django.shortcuts import get_object_or_404, render, redirect

from .models import Image

# Create your views here.
def index(request):
    return HttpResponse("Index page, slightly buggy.")

def upload(request):
    if request.user.is_staff:
        return None
    else:
        return HttpResponse("You're not staff, get out!")

def answer(request):
    if request.user.is_authenticated:
        user=request.user
        print(user)
        if Image.objects.filter(user_working_on_task=user.username).exists():
            image = Image.objects.filter(user_working_on_task=user.username).first()
        else: #Here is where I generate the new set.
            image = Image.objects.filter(number_of_times_served=0, breed=None).first()
            #image.number_of_times_served=1
            image.user_working_on_task=user.username
            image.save()
        context = {'image_to_classify': image,
                    'user' : user}
        if(request.GET.get('yes_btn')):
            print("yest")
        return render(request, 'answer.html', context)
    else:
        return redirect('/accounts/login/')