/usr/local/bin/ffmpeg -framerate 24 -i $1/%03d.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -c:v h264 -pix_fmt yuv420p $2
