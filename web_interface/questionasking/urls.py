from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('answer/', views.answer, name='answer'),
    path('process_answer/<int:pk>/<str:answer>', views.process_answer, name='process_answer'),
]