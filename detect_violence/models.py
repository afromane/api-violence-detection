from djongo import models

class VideoStream(models.Model):
    """
    Modèle de base pour les flux vidéo.
    """
    _id = models.ObjectIdField(primary_key=True)
    name = models.CharField(max_length=100, blank=True)
    description = models.TextField()
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)

    class Meta:
        verbose_name = "Video Stream"
        verbose_name_plural = "Video Streams"

class RecordedVideo(VideoStream):
    """
    Modèle pour représenter les enregistrements vidéo.
    """
    path = models.TextField()

    class Meta:
        verbose_name = "Recorded Video"
        verbose_name_plural = "Recorded Videos"

    def save(self, *args, **kwargs):
        # Ensure path field is saved
        self.full_clean()  # Perform full validation
        super().save(*args, **kwargs)

class CameraStream(VideoStream):
    """
    Modèle pour représenter les flux de caméra en direct.
    """
    url = models.URLField()

    class Meta:
        verbose_name = "Camera Stream"
        verbose_name_plural = "Camera Streams"

class ViolenceEvent(models.Model):
    """
    Modèle de base pour les evenements violente.
    """
    _id = models.ObjectIdField(primary_key=True)
    cadence = models.IntegerField()
    violence = models.FloatField()  # ratio violence
    non_violence = models.FloatField()  # ratio non violence
    path_frame = models.TextField()
    path_video = models.TextField(blank=True)
    #interval = models.ArrayField(model_container=models.FloatField(),default=list )
    createdAt = models.DateTimeField(auto_now_add=True, blank=True)
    video_stream = models.ForeignKey('RecordedVideo', on_delete=models.CASCADE)

    class Meta:
        verbose_name = "Violence Event"
        verbose_name_plural = "Violence Events"
