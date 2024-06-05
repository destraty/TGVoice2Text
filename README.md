# TGVoice2Text - бот для голосовых и кружочков
![Logo](https://github.com/destraty/TGVoice2Text/blob/master/assets/v2t.png?raw=true)

С помощью данного бота вам больше не придется часами слушать голосовые и смотреть кружочки в телеграмме. Благодаря нейросетям можно ~~быстро~~ бесплатно перевести голосовые и кружочки в текст.

### Как это работает?
Файлы форматов ```.mp4```(кружочки) или ```.oga```(голосовые) поступаю в бота и преобразовываются в формат ```.wav``` с моно звуковой дорожкой. Далее этот файл поступает на вход [Vosk](https://alphacephei.com/vosk/index.ru) модели, которая преобразует голос в текст. Затем при помощи уже второй Vosk модели добавляется пунктуация в сообщение. 

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
6. Распакуйте их в любое удобное место.
7. Сконфиругируйте файл .env и укажите пути для обоих моделей по своему желанию. Хранить их на рабочем столе настоятельно **НЕ РЕКОМЕНДУЕТСЯ!**
- Для ```TOKEN``` укажите токен вашего бота телеграмм. Получить его можно у [BotFather](https://t.me/botfather). Текствая/видео инструкция [тут](https://help.zoho.com/portal/en/kb/desk/support-channels/instant-messaging/telegram/articles/telegram-integration-with-zoho-desk#How_to_create_a_Telegram_Bot)
- Для ```SPEECH_MODEL_PATH``` укажите путь к папке с моделью.
- Для ```PUNC_MODEL_PATH``` укажите путь к файлу checkpoint.

8. Запустите бота
```bash
python bot.py
```

Дождитесь вывода ```DONE``` и пользуйтесь своим ботом)
