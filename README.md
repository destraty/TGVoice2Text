# TGVoice2Text - бот для голосовых и кружочков
![Logo](https://github.com/destraty/TGVoice2Text/blob/master/assets/v2t.png?raw=true)

С помощью данного бота вам больше не придется часами слушать голосовые и смотреть кружочки в телеграмме. Благодаря нейросетям можно ~~быстро~~ бесплатно перевести голосовые и кружочки в текст.

# Содержание 
1. [Как это работает?]()
2. [Минимальные системные требования]()
3. [Установка на Windows]()
4. [Установка на Linux]()
5. [Материалы для изучения]()

### Как это работает?
Файлы форматов ```.mp4```(кружочки) или ```.oga```(голосовые) поступаю в бота и преобразовываются в формат ```.wav``` с моно звуковой дорожкой. Далее этот файл поступает на вход [Vosk](https://alphacephei.com/vosk/index.ru) модели, которая преобразует голос в текст. Затем при помощи уже второй Vosk модели добавляется пунктуация в сообщение. 

# Системные требования
>Только x64 архитектура

>Более 4GB RAM

>От 10GB свободного пространства на диске

# Установка
### Windows
1. Скачайте и установите ```Python 3.11.7```, если у вас все еще его нет [отсюда](https://www.python.org/downloads/release/python-3117/).
2. Установите FFMpeg [отсюда](https://www.ffmpeg.org/download.html#build-windows). Он нужен для распознавания речи.
3. Добавьте FFMpeg в PATH:

   Свойства системы > Переменные среды > Переменные среды пользователя > PATH > ~~Например:~~ ```C:\Program Files\ffmpeg-2024-05-27-git-01c7f68f7a-full_build\bin\```
   
   Перезагрузите компьютер

4. Клонируйте репозиторий командой:
```bash
git clone https://github.com/destraty/TGVoice2Text
```
5. Перейдтие в папку и установите все зависимости
```bash
cd TGVoice2Text
pip install -r requirements.txt
```
6. Скачайте модели [распознавания речи](https://alphacephei.com/vosk/models/vosk-model-ru-0.10.zip) и [пунктуации](https://alphacephei.com/vosk/models/vosk-recasepunc-ru-0.22.zip)
7. Распакуйте их в любое удобное место.
8. Сконфиругируйте файл .env и укажите пути для обоих моделей по своему желанию. Хранить их на рабочем столе настоятельно **НЕ РЕКОМЕНДУЕТСЯ!**
- Для ```TOKEN``` укажите токен вашего бота телеграмм. Получить его можно у [BotFather](https://t.me/botfather). Текствая/видео инструкция [тут](https://help.zoho.com/portal/en/kb/desk/support-channels/instant-messaging/telegram/articles/telegram-integration-with-zoho-desk#How_to_create_a_Telegram_Bot)
- Для ```SPEECH_MODEL_PATH``` укажите путь к папке с моделью.
- Для ```PUNC_MODEL_PATH``` укажите путь к файлу checkpoint.

9. Запустите бота
```bash
python bot.py
```
Дождитесь вывода ```DONE``` и пользуйтесь своим ботом)

## Linux
1. Установите Python 3.11 из [гайда](https://www.linuxcapable.com/how-to-install-python-3-11-on-ubuntu-linux/)
2. Установите FFMpeg и HDF5 
```shell
sudo apt update
sudo apt install ffmpeg libhdf5-dev -y
ffmpeg -version
```

3. Клонируйте репозиторий командой:
```shell
git clone https://github.com/destraty/TGVoice2Text
```

4. Перейдтие в папку и создайте виртуальное окружение
```shell
cd TGVoice2Text
python -m venv env
source env/bin/activate
```

5. Установите зависисмости. Установка ```h5py``` может занять продолжительное время, так и должно быть. Если на этом этапе возникают ошибки, проверьте наличие ```libhdf5-dev```.
```shell
pip install -r requirements.txt
```

6. Создайте папку для моделей и скачайте их:
```shell 
mkdir models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-ru-0.10.zip
wget https://alphacephei.com/vosk/models/vosk-recasepunc-ru-0.22.zip
```
7. В этой же папке распаковываем модели. Процесс тоже может быть достаточно длительным, общий вес моделей порядка 6ГБ.
```shell
unzip unzip vosk-model-ru-0.10.zip
rm vosk-model-ru-0.10.zip

unzip vosk-recasepunc-ru-0.22.zip
rm vosk-recasepunc-ru-0.22.zip

```
8. Отредактируем файл ```nano .env``` в папке ```TGVoice2Text``` на что-то подобное:
```shell
TOKEN = 'ВАШ ТОКЕН ТГ БОТА'
DOWNLOAD_DIR = './downloads'
SPEECH_MODEL_PATH = "./models/vosk-model-ru-0.10"
PUNC_MODEL_PATH = "./models/vosk-recasepunc-ru-0.22/checkpoint"
```
9. Создадим демон .service для хостинга бота:
```shell
sudo nano /etc/systemd/system/v2t_bot.service
```
```shell
[Unit]
Description=v2t
After=network.target

[Service]
Type=simple
User=ПОЛЬЗОВАТЕЛЬ
WorkingDirectory=/home/ПОЛЬЗОВАТЕЛЬ/TGVoice2Text/
ExecStart=/usr/bin/python3.11 /path/to/TGVoice2Text/bot.py

[Install]
WantedBy=multi-user.target
```
10. Запустим бота
```shell
sudo systemctl daemon-reload
sudo systemctl enable v2t_bot.service
sudo systemctl start v2t_bot.service
```
Проверьте правильность запуска используя:
```shell
sudo systemctl status v2t_bot.service
```
