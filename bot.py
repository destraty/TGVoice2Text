import os

import dotenv
import soundfile as sf
import vosk
from pydub import AudioSegment
from telebot import TeleBot
from recasepunc import CasePuncPredictor
from recasepunc import WordpieceTokenizer  # НЕ УДАЛЯТЬ
from recasepunc import Config

dotenv.load_dotenv()

# Токен бота
bot = TeleBot(os.getenv('TOKEN'))
# Путь к директории, куда будут сохраняться файлы
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR")
# Загрузка распознавания Vosk
model = vosk.Model(os.getenv("SPEECH_MODEL_PATH"))
# Модель пунктуации Vosk
predictor = CasePuncPredictor(os.getenv("PUNC_MODEL_PATH"), lang="ru")
print("DONE")

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def s2t(audio_file: str):
    """
    Переводит аудио в текст
    :param audio_file: путь к файлу
    :return: строка с расшифровкой текста и пунктуацией
    """
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
            data_bytes = data.tobytes()
            if rec.AcceptWaveform(data_bytes):
                continue

        res = eval(rec.Result())["text"]
        tokens = list(enumerate(predictor.tokenize(res)))

        results = ""
        for token, case_label, punc_label in predictor.predict(tokens, lambda x: x[1]):
            prediction = predictor.map_punc_label(predictor.map_case_label(token[1], case_label), punc_label)
            if token[1][0] != '#':
                results = results + ' ' + prediction
            else:
                results = results + prediction

        return str(results.strip())


def to_mono_wav(input_file_path: str,
                output_file_path: str):
    if input_file_path.endswith(".oga"):
        audio = AudioSegment.from_ogg(input_file_path)
    elif input_file_path.endswith(".mp4"):
        audio = AudioSegment.from_file(input_file_path, format="mp4")
    else:
        raise TypeError("Input files must be only .oga or .mp4 formats")

    mono_audio = audio.set_channels(1)  # Преобразование в моно
    silence = AudioSegment.silent(duration=4000)  # Создание 4 секунд тишины
    final_audio = mono_audio + silence  # Добавление тишины к аудио
    final_audio.export(output_file_path, format="wav")  # Экспорт в формат WAV


@bot.message_handler(content_types=['voice', 'video_note'])
def voice_message_handler(message):
    file_info = bot.get_file(message.voice.file_id if message.content_type == 'voice' else message.video_note.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Сохраняем файл в локальную директорию
    fn = f'{message.from_user.id}_{file_info.file_path.split("/")[-1]}'
    with open(os.path.join(DOWNLOAD_DIR, fn), 'wb') as new_file:
        new_file.write(downloaded_file)

    to_mono_wav(f"{DOWNLOAD_DIR}/{fn}", f"{DOWNLOAD_DIR}/{fn[:(len(fn) - 4)]}.wav")
    os.remove(f'{DOWNLOAD_DIR}/{fn}')
    text = s2t(f"{DOWNLOAD_DIR}/{fn[:(len(fn) - 4)]}.wav")
    if text:
        bot.reply_to(message, f"{text}")
    else:
        bot.reply_to(message, f"Не удалось распознать.")


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привет Отправь мне голосовое и я его переведу в текст")


bot.polling()
