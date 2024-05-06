from djongo import models
from djongo.models.fields import JSONField


class RecordedVideo(models.Model):
    """
    Modèle pour représenter les enregistrements vidéo.
    """
    _id = models.ObjectIdField(primary_key=True)
    name = models.CharField(max_length=100, blank=True)
    description = models.TextField(blank=True)
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    path = models.TextField()
    class Meta:
        verbose_name = "Recorded Video"
        verbose_name_plural = "Recorded Videos"

class ViolenceEvent(models.Model):
    """
    Modèle de base pour les evenements violente.
    """
    _id = models.ObjectIdField(primary_key=True)
    cadence = models.IntegerField()
    violence = models.FloatField()  # ratio violence
    non_violence = models.FloatField(blank=True)  # ratio non violence
    path_frame = models.TextField(blank=True)
    path_video = models.TextField(blank=True)
    interval = JSONField(default=list)  # Champ pour stocker la liste de float
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    video_stream = models.ForeignKey('RecordedVideo', on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Violence Event"
        verbose_name_plural = "Violence Events"
class ViolenceEventCameraStream(models.Model):
    """
    Modèle de base pour les evenements violente.
    """
    _id = models.ObjectIdField(primary_key=True)
    violence = models.FloatField()  
    path_video = models.TextField(blank=True)
    #interval = models.ArrayField(model_container=models.FloatField(),default=list )
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    #video_stream = models.ForeignKey(CameraStream, on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Violence Event camera strem"
        verbose_name_plural = "Violence Events camera stream"
