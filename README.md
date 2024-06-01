# TGVoice2Text - бот для голосовых и кружочков

TODO Общее описание 

# Установка
### Windows
1. Установите FFMpeg [отсюда](https://www.ffmpeg.org/download.html#build-windows). Он нужен для распознавания речи.
2. Добавьте FFMpeg в PATH:

   Свойства системы > Переменные среды > Переменные среды пользователя > PATH > ~~Например:~~ ```C:\Program Files\ffmpeg-2024-05-27-git-01c7f68f7a-full_build\bin\```
   
   Перезагрузите компьютер

3. Клонируйте репозиторий командой:
```bash
git clone https://github.com/destraty/TGVoice2Text
```
4. Перейдтие в папку и установите все зависимости
```bash
cd TGVoice2Text
pip install -r requirements.txt
```
5. Скачайте модели [распознавания речи](https://alphacephei.com/vosk/models/vosk-model-ru-0.10.zip) и [пунктуации](https://alphacephei.com/vosk/models/vosk-recasepunc-ru-0.22.zip)
6. Сконфиругируйте файл .env и укажите пути для обоих моделей по своему желанию. Хранить их на рабочем столе настоятельно **НЕ РЕКОМЕНДУЕТСЯ!**
7. Запустите бота
```bash
python bot.py
```
