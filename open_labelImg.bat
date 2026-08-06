@echo off
cd /d D:\python\ObjectDetectionModel
call .venv\Scripts\activate
labelImg raw_images classes.txt raw_labels
pause
