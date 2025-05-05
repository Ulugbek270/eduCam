from django.db import models
from django.utils import timezone
import numpy as np
import pickle
import base64


# ==== Utilities for storing face embeddings ====
def serialize_embedding(embedding: np.ndarray) -> str:
    return base64.b64encode(pickle.dumps(embedding)).decode('utf-8')

def deserialize_embedding(embedding_str: str) -> np.ndarray:
    return pickle.loads(base64.b64decode(embedding_str.encode('utf-8')))



class Teacher(models.Model):
    full_name = models.CharField(max_length=100)
    phone_number = models.CharField(max_length=15)

    def __str__(self):
        return self.full_name


# For current camera in the room
class Camera(models.Model):
    name = models.CharField(max_length=100, unique=True)
    camera_ip = models.GenericIPAddressField()
    camera_username = models.CharField(max_length=50)
    camera_password = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class Classroom(models.Model):
    name = models.CharField(max_length=100)
    teacher = models.ForeignKey(Teacher, on_delete=models.SET_NULL, null=True)
    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return self.name


class Student(models.Model):
    full_name = models.CharField(max_length=100)
    student_id = models.CharField(max_length=20, unique=True)
    classroom = models.ForeignKey(Classroom, on_delete=models.CASCADE)
    photo = models.ImageField(upload_to='students/')
    face_embedding = models.TextField()

    def set_embedding(self, embedding: np.ndarray):
        self.face_embedding = serialize_embedding(embedding)

    def get_embedding(self) -> np.ndarray:
        return deserialize_embedding(self.face_embedding)

    def __str__(self):
        return f"{self.full_name} ({self.student_id})"


class AttendanceRecord(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(default=timezone.now)
    camera = models.ForeignKey(Camera, on_delete=models.SET_NULL, null=True)
    recognized = models.BooleanField(default=True) # present
    snapshot = models.ImageField(upload_to='camera_photos/', null=True, blank=True)

    class Meta:
        unique_together = ('student',)

    def __str__(self):
        return f"{self.student.full_name} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
