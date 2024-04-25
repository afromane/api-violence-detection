from rest_framework import serializers

class CreateBookWithFileSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=100)
    author = serializers.CharField(max_length=100)
    publication_year = serializers.IntegerField()
    file = serializers.FileField()
