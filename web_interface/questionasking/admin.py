from django.contrib import admin
from .models import Trait
from .models import Breed
from .models import Image

# Register your models here.
admin.site.register(Trait)
admin.site.register(Breed)
admin.site.register(Image)
