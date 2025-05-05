from django.contrib import admin
from .models import Student, Teacher, AttendanceRecord, Camera, Classroom
# Register your models here.


admin.site.register(Student)
admin.site.register(Teacher)
admin.site.register(AttendanceRecord)
admin.site.register(Camera)
admin.site.register(Classroom)