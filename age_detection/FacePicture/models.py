from django.db import models



class FacePicture(models.Model):

    filename = models.CharField(max_length=72, primary_key=True)
    age = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    @classmethod
    def create(cls, filename, age):
        facePicture = cls(filename=filename, age=age)
        # do something with the book
        return facePicture

        



    


