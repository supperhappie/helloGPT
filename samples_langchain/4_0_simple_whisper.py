# ffmpeg : compress video, extract audio from a video, ...
# cml : ffmpeg -i .\samples_langchain\files\video_sample.mp4 -vn .\samples_langchain\files\video_sample_audio.mp3
from pydub import AudioSegment
import openai
import yt_dlp
import subprocess # run cml on python 
import os, io

def extract_audio_from_video(video_path, audio_path):
    if os.path.exists(audio_path):
        os.remove(audio_path)
    command = ["ffmpeg", "-i", video_path, "-vn", audio_path]
    subprocess.run(command)
    
def split_audio(audio_path, unit_ms, o_audio_path = None):
    if o_audio_path is None:
        o_audio_path = audio_path
        
    print(f"split_audio({audio_path})")
    audio = AudioSegment.from_mp3(audio_path)
    audio_chunks = [audio[i:i+unit_ms] for i in range(0, len(audio), unit_ms)]
    
    base, ext = os.path.splitext(o_audio_path)
    for idx, chunk in enumerate(audio_chunks):
        output_file = f"{base}_{idx:02d}{ext}"
        if os.path.exists(output_file):
            os.remove(output_file)
        chunk.export(output_file, format="mp3")
    return audio_chunks

# # test code for extract_audio_from_video and split_audio
# i_videos_path = "./samples_langchain/files/video_sample.mp4"
# o_audio_path = "./samples_langchain/files/video_sample_audio.mp3"
# o_audio_10s_path = "./samples_langchain/files/video_sample_audio_first_30s.mp3"

# extract_audio_from_video(i_videos_path, o_audio_path)
# second = 1000
# minute = 60 * 1000  # unit : milisecond 

# split_audio(o_audio_path, 10 * second)
# # delete all test files 
# if os.path.exists(o_audio_path):
#     os.remove(o_audio_path)
    
# base, ext = os.path.splitext(o_audio_10s_path)
# count = 0
# while True:
#     split_file = f"{base}_{count:02d}{ext}"
#     if os.path.exists(split_file):
#         os.remove(split_file)
#         print(f"{split_file} is deleted.")
#         count += 1
#     else:
#         break
# print("all splited files are deleted.")

def download_video_from_youtube(youtube_url, output_path="temp_video.mp4"):
    if os.path.exists(output_path):
        os.remove(output_path)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

# # YouTube 비디오 URL 예시
# youtube_url = "https://www.youtube.com/watch?v=9-2e7PvXsN4"
# output_video_path = "./samples_langchain/files/youtube_video.mp4"

# # YouTube에서 비디오 다운로드
# downloaded_video_path = download_video_from_youtube(youtube_url, output_video_path)
# print(f"비디오가 다음 경로에 다운로드되었습니다: {downloaded_video_path}")

def download_audio_from_youtube(youtube_url, output_path="temp_audio.mp3"):
    if os.path.exists(output_path):
        os.remove(output_path)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            # 파일 내용을 메모리에 로드
            audio_data = io.BytesIO(audio_file.read())
            audio_data.name = "audio.mp3"  # 파일 이름 설정
            
            # Whisper API를 사용하여 음성을 텍스트로 변환
            result = openai.Audio.transcribe("whisper-1", audio_data)
            return result["text"]
    except IOError as e:
        print(f"file to open the file[{file_path}] : {e}")
        return None
    

def transcribe_youtube_video(youtube_url, split_unit=10*60*1000):
    # split_unit의 최대값 확인
    max_split_unit = 10*60*1000
    if split_unit > max_split_unit:
        print(f"경고: split_unit이 최대값 {max_split_unit}을 초과했습니다. 최대값으로 설정됩니다.")
        split_unit = max_split_unit

    tmpfile_path = "./"
    tmpfile_name = "temp_audio"
    tmpfile_ext = ".mp3"
    tmpfile = tmpfile_path + tmpfile_name + tmpfile_ext
    # YouTube URL에서 오디오 파일 가져오기
    download_audio_from_youtube(youtube_url, tmpfile_path + tmpfile_name)
    print("end download_audio_from_youtube")
    # 오디오 파일 분할
    audio_chunks = split_audio(tmpfile, split_unit)
    print("end split_audio")
    
    transcriptions = []
    # 오디오 파일을 바이너리 모드로 열기
    # for audio_chunk in audio_chunks:
    total_chunks = len(audio_chunks)
    for i in range(total_chunks):
        tmpfile_splited = f"{tmpfile_name}_{i:02d}{tmpfile_ext}"
        result = transcribe_audio(tmpfile_splited)
        transcriptions.append(result)
        progress = (i + 1) / total_chunks * 100
        print(f"Transcribe Audio Process : {progress:.2f}%")
    print("Complete Transcribe Audio")
    # 임시 파일 삭제
    for i in range(len(audio_chunks)):
        os.remove(f"{tmpfile_name}_{i:02d}{tmpfile_ext}")
    os.remove(tmpfile)
    print("Remove tmp files")
    
    # 전체 텍스트 반환
    return " ".join(transcriptions)

# test code for transcribe_youtube_video
transcriptions = transcribe_youtube_video("https://www.youtube.com/watch?v=7J44j6Fw8NM")
print(transcriptions)

def transcribe_video(video_file_path, split_unit=10*60*1000):
    # split_unit의 최대값 확인
    max_split_unit = 10*60*1000
    if split_unit > max_split_unit:
        print(f"경고: split_unit이 최대값 {max_split_unit}을 초과했습니다. 최대값으로 설정됩니다.")
        split_unit = max_split_unit

    tmpfile_path = "./"
    tmpfile_name = "temp_audio"
    tmpfile_ext = ".mp3"
    tmpfile = tmpfile_path + tmpfile_name + tmpfile_ext
    
    # 비디오 파일에서 오디오 추출
    extract_audio_from_video(video_file_path, tmpfile)
    print("오디오 추출 완료")
    
    # 오디오 파일 분할
    audio_chunks = split_audio(tmpfile, split_unit)
    print("오디오 분할 완료")
    
    transcriptions = []
    total_chunks = len(audio_chunks)
    for i in range(total_chunks):
        tmpfile_splited = f"{tmpfile_name}_{i:02d}{tmpfile_ext}"
        result = transcribe_audio(tmpfile_splited)
        transcriptions.append(result)
        progress = (i + 1) / total_chunks * 100
        print(f"음성 변환 진행률: {progress:.2f}%")
    print("음성 변환 완료")
    
    # 임시 파일 삭제
    for i in range(len(audio_chunks)):
        os.remove(f"{tmpfile_name}_{i:02d}{tmpfile_ext}")
    os.remove(tmpfile)
    print("임시 파일 삭제 완료")
    
    # 전체 텍스트 반환
    return " ".join(transcriptions)

# test code for transcribe_video
# transcriptions = transcribe_video("./samples_langchain/files/video_sample.mp4")
# print(transcriptions)