import os
import vosk
import soundfile as sf
from telebot import TeleBot
from pydub import AudioSegment
import dotenv
# Замените 'YOUR_BOT_TOKEN' на токен вашего бота

dotenv.load_dotenv()

TOKEN = os.getenv('TOKEN')
bot = TeleBot(TOKEN)

# Загрузка модели Vosk

model = vosk.Model("C:\models\model_big")
print("DONE")

# Путь к директории, куда будут сохраняться файлы
DOWNLOAD_DIR = './downloads'

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def s2t(audio_file: str):
    with sf.SoundFile(audio_file) as sound_file:
        # Получение параметров аудиофайла
        samplerate = sound_file.samplerate
        # Создание распознавателя с соответствующей частотой дискретизации
        rec = vosk.KaldiRecognizer(model, samplerate)
        # Чтение и распознавание аудиофайла блоками
        while True:
            data = sound_file.read(-1, dtype="int16")
            if len(data) == 0:
                break
            # Преобразование numpy.ndarray в байты
            data_bytes = data.tobytes()
            if rec.AcceptWaveform(data_bytes):
                pass
            else:
                pass
                # print(rec.PartialResult())
                print(eval(rec.PartialResult())["partial"])
        res = eval(rec.Result())["text"]
        print(res)
        return str(res)


def oga_to_mono_wav(input_file_path, output_file_path):
    audio = AudioSegment.from_ogg(input_file_path)
    mono_audio = audio.set_channels(1)  # Преобразование в моно
    silence = AudioSegment.silent(duration=4000)  # Создание 4 секунд тишины
    final_audio = mono_audio + silence  # Добавление тишины к аудио
    final_audio.export(output_file_path, format="wav")  # Экспорт в формат WAV

@bot.message_handler(content_types=['voice', 'video'])
def handle_docs_photo(message):
    file_info = bot.get_file(message.voice.file_id if message.content_type == 'voice' else message.video.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Сохраняем файл в локальную директорию
    fn = f'{message.from_user.id}_{file_info.file_path.split("/")[-1]}'
    with open(os.path.join(DOWNLOAD_DIR, fn), 'wb') as new_file:
        new_file.write(downloaded_file)

    oga_to_mono_wav(f"./downloads/{fn}", f"./downloads/{fn[:(len(fn) - 4)]}.wav")
    os.remove(f'./downloads/{fn}')
    text = s2t(f"./downloads/{fn[:(len(fn) - 4)]}.wav")
    bot.reply_to(message, f"Ваш текст: {text}")


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет Отправь мне голосовое или видео сообщение, и я сохраним его.")


bot.polling()
