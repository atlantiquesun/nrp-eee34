from django.db import models
from ast import literal_eval
from django.forms import ModelForm
from .ulamrenyi import *
from django.db.models import Exists

# For generating Breed matrix.
from django.db.models.signals import m2m_changed
from django.dispatch import receiver

# To whoever has the displeasure of reading this mess.
# Due to inherent limitations regarding the storing of arrays in databases, I am storing the array as a string and running literal_eval on it. You should not do this.


class Trait(models.Model):
    trait_name = models.CharField(max_length=50)

    def __str__(self):
        return self.trait_name


class Breed(models.Model):
    breed_name = models.TextField(null=True)
    breed_matrix = models.TextField(default=None, blank=True, null=True)
    traits = models.ManyToManyField(Trait)

    def __str__(self):
        return self.breed_name


@receiver(m2m_changed, sender=Breed.traits.through)
# if no traits are set, this does not work. so please don't set breeds with no traits.
def eval_breed_matrix(sender, instance, **kwargs):
    print("received!")
    trait_list = Trait.objects.order_by('id').all()
    print(trait_list)
    print(instance.traits)
    breed_matrix = []
    for x in trait_list:
        if x in instance.traits.all():
            breed_matrix.append(1)
        else:
            breed_matrix.append(0)
    instance.breed_matrix = str(breed_matrix)
    instance.save()


class Image(models.Model):
    # Fields for the database.
    # Pre-setup
    trait_dictionary = models.TextField(default=None, blank=True, null=True)
    breed_dictionary = models.TextField(default=None, blank=True, null=True)
    trait_breed_matrix = models.TextField(default=None, blank=True, null=True)
    # Right now I'm assuming that they will only mess up a maximum of half the time. This corresponds to e in the paper.
    errors = models.IntegerField(default=int(
        Trait.objects.count()/2), null=True)
    # Administration
    user_working_on_task = models.CharField(
        default=None, max_length=20, blank=True, null=True)
    # URL linking to the image
    image = models.ImageField(default=None, null=True)
    # Not being used for anything rn, but i just left it in in case we wanna tweak stuff.
    number_of_times_served = models.IntegerField(default=0)
    # Game states
    # Current state of the Ulam-Renyi game being played.
    game_state = models.TextField(default=None, blank=True, null=True)
    # Current sigma state, included since it would be a waste of time to re-calculate sigma states over and over again.
    sigma_game_state = models.TextField(default=None, blank=True, null=True)
    # Question
    # What question to ask given the set T as dictated in the paper.
    question_set_constraint = models.TextField(
        default=None, blank=True, null=True)
    question_set = models.TextField(default=None, blank=True, null=True)
    question_text = models.TextField(default=None, blank=True, null=True)
    # Final breed
    # Ultimately the breed that is decided.
    breed = models.ForeignKey(
        Breed, default=None, blank=True, on_delete=models.CASCADE, null=True)

    def __str__(self):
        return self.image.url

    def save(self, *args, **kwargs):
        # If object has been newly created.
        if self.pk is None:
            # Populate dictionaries.
            # This is very inefficient, but I just want to get this working...
            self.trait_dictionary = str(dict(enumerate((x for x in list(
                Trait.objects.values_list('trait_name', flat=True).order_by('id'))), 1)))
            self.breed_dictionary = str(dict(enumerate((x for x in list(
                Breed.objects.values_list('breed_name', flat=True).order_by('id'))), 1)))
            # Generate trait-breed matrix.
            self.trait_breed_matrix = str(self.generate_trait_breed_matrix())
            # Generate game_state
            #1-indexed.
            self.game_state = str([list(range(1, Breed.objects.count()+1))] + [[]]*self.errors)
            print(self.game_state)
        self.sigma_game_state = str(set_sigma(literal_eval(self.game_state)))
        print(self.sigma_game_state)
        self.question_set_constraint = str(run_algorithm(literal_eval(self.sigma_game_state)))
        (a,b)=naturalquestion(self.errors, literal_eval(self.game_state), literal_eval(self.trait_dictionary), literal_eval(self.question_set_constraint),  Breed.objects.count(), literal_eval(self.trait_breed_matrix))
        (self.question_set,self.question_text) = (str(a),str(b))
        super(Image, self).save(*args, **kwargs)

    def generate_trait_breed_matrix(self):
        tuberculosis_matrix = []
        print(Breed.objects.values_list('breed_matrix').order_by('id'))
        for x in list(Breed.objects.values_list('breed_matrix', flat=True).order_by('id')):
            tuberculosis_matrix.append(literal_eval(x))
        return tuberculosis_matrix
