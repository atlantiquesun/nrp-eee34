from django.db import models
from ast import literal_eval
from django.forms import ModelForm
from .ulamrenyi import *

class Trait(models.Model):
    trait_name=models.CharField(max_length=50)
    def __str__(self):
        return self.trait_name

class Breed(models.Model):
    breed_name=models.TextField(null=True)
    traits=models.ManyToManyField(Trait)
    def __str__(self):
        return self.breed_name

class Image(models.Model):
    # Fields for the database.
    user_working_on_task = models.CharField(default=None, max_length=20, blank=True, null=True)
    number_of_times_served = models.IntegerField(default=0) #Not being used for anything rn, but i just left it in in case we wanna tweak stuff.
    errors = models.IntegerField(default=int(Trait.objects.count()/2), null=True) #Right now I'm assuming that they will only mess up a maximum of half the time. This corresponds to e in the paper.
    image = models.ImageField(default=None, null=True)  # URL linking to the image
    game_state = models.TextField(default=None, blank=True, null=True)  # Current state of the Ulam-Renyi game being played.
    sigma_game_state = models.TextField(default=None, blank=True, null=True)  # Current sigma state, included since it would be a waste of time to re-calculate sigma states over and over again.
    question_set_constraint = models.TextField(default=None, blank=True, null=True) #What question to ask given the set T as dictated in the paper.
    question_set = models.TextField(default=None, blank=True, null=True)
    question_text = models.TextField(default=None, blank=True, null=True)
    breed = models.ForeignKey(Breed, default=None, blank=True, on_delete=models.CASCADE, null=True) #Ultimately the breed that is decided.

    def __str__(self):
        return self.image.url
    
    def save(self, *args, **kwargs):
        if self.game_state is "":
            self.game_state=str([list(range(0,Breed.objects.count()))] + [[]]*self.errors)
            print(self.game_state)
        self.sigma_game_state=str(set_sigma(literal_eval(self.game_state)))
        print(self.sigma_game_state)
        self.question_set_constraint = str(run_algorithm(literal_eval(self.sigma_game_state)))
        self.question_set=str(generate_question(literal_eval(self.question_set_constraint)))
        self.question_text=str(nlp_generate_string(literal_eval(self.question_set)))
        super(Image, self).save(*args, **kwargs)