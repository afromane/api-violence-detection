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



class CameraStream(VideoStream):
    """
    Modèle pour représenter les flux de caméra en direct.
    """
    url = models.TextField()

    class Meta:
        verbose_name = "Camera Stream"
        verbose_name_plural = "Camera Streams"
