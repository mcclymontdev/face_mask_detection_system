FOR /R "%~dp0\dataset\" %%F IN (*.png) DO (
    magick mogrify "%%F"
)