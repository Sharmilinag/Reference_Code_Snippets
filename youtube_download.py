from pytube import YouTube

url = 'https://www.youtube.com/watch?v=k5dLk9a-LDM'

yt = YouTube(url)
yt.streams.filter(only_audio=True).first().download(output_path='ddd')
