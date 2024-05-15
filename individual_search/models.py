from djongo import models
from djongo.models.fields import JSONField



class RecordedVideo(models.Model):
    """
    Modèle pour représenter les enregistrements vidéo.
    """
    _id = models.ObjectIdField(primary_key=True)
    name = models.CharField(max_length=100, blank=True)
    description = models.TextField()
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)

    image_path = models.TextField()
    video_path = models.TextField()

    class Meta:
        verbose_name = "Recorded Video"
        verbose_name_plural = "Recorded Videos"


class CameraStreamResult(models.Model):
    """
    Modèle de base pour enregistrer les recherche issu des camera.
    """
    _id = models.ObjectIdField(primary_key=True)
    threshold = models.IntegerField()
    similarity = models.TextField(blank=True)
    path_frame = models.TextField(blank=True)
    detected_time = models.TextField(blank=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    #camera = models.ForeignKey('RecordedVideo', on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Camera Stream Result"
        verbose_name_plural = "Camera Stream Result"

class IndividualSearchFromRecordedVideo(models.Model):
    """
    Modèle de base pour enregistrer les recherche issu des des enregistrement videos.
    """
    _id = models.ObjectIdField(primary_key=True)
    duration = models.IntegerField()
    threshold = models.IntegerField()
    similarity = JSONField(default=list)
    path_video = models.TextField(blank=True)
    recognition_path = models.TextField(blank=True)
    detected_time = JSONField(default=list)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    recorded_video = models.ForeignKey('RecordedVideo', on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Recorded Video Result"
        verbose_name_plural = "Recorded Video Results"