# IEMS 5780 Face Classification Report

## Functions of the system 

Face detection & emotion/gender classification.

User can send photo through Telegram bot, and receive the result image.

**e.g. Send by URL.**

<img src="https://raw.githubusercontent.com/Paranoid-kid/Scalable-Emotion-Gender-Classification/master/img/1.png" style="zoom:40%" />

**Or send by image.**

<img src="https://raw.githubusercontent.com/Paranoid-kid/Scalable-Emotion-Gender-Classification/master/img/2.png" style="zoom:40%" />

> This model is from the B-IT-BOTS robotics team.
>
> GitHub: https://github.com/oarriaga/face_classification

## Model Performance

The training uses fer2013/imdb datasets with a keras CNN model and openCV.

- IMDB gender classification test accuracy: 96%.
- fer2013 emotion classification test accuracy: 66%.

## System architecture 

### Diagram

![image-20181222205752643](https://raw.githubusercontent.com/Paranoid-kid/Scalable-Emotion-Gender-Classification/master/img/3.png)

### Components:

In this case, there are four key components used to implement the system.

- **Flask**: Backend 
- **Nginx**: Load Banlancer, Reverse Proxy
- **Gunicorn**: Web Server, Multiple **Async** Worker (Gevent)
- **Docker**: Easy to deployment

### Basic Flow:

1. **bot.py**: receive the requests and forward to Nginx
2. **Nginx**: distribute the requests to different gunicorn worker based the different URL
3. **Gunicorn** **& Flask**: do prediction and send image back to user

### Bot.py:

In bot.py, there are two thread, thread 1 and thread 2. 

Thread 1 is used to receive the request from user and store the wrappered message into queue.

Wrapped message look like this:

```python
wrapped_msg = {
                'type': 'url',
                'url': image_url,
                'chat_id': chat_id,
			}
```

Thread 2 will retrive the message in queue and send them to different endpoint based on the **type**.

There are two endpoints to Nginx:

```python
url_classify = 'http://127.0.0.1:80/classify' 				# If type is 'file_id'
url_classify_url = 'http://127.0.0.1:80/classify_url'		# If type is 'url'
```

### app_image &app_url

Since the I/O will slow the perfomance.

In this case, I only do the nessessary I/O operations:

- Server side: download once, save predicted-image once, read image once
- bot.py: **None**
- Total I/O: Three times

Compare to assignment 2:

- Server side: download once, save predicted-image once, read image once
- bot.py:  download image once, read image once
- Total I/O: Five times

### Gunicorn

The picture down below is the time consumed by two predictions.

![image-20181222215508805](https://raw.githubusercontent.com/Paranoid-kid/Scalable-Emotion-Gender-Classification/master/img/4.png)

We can find that the prediction actually is not very time consuming. The I/O took a long time.

So I set the gunicorn **asyn** mode by:

```bash
gunicorn -w 4 app_image:app -b 0.0.0.0:8001 -k gevent
gunicorn -w 4 app_url:app -b 0.0.0.0:8000 -k gevent
```

Each gunicorn server has 4 asyn workers to achieve **concurrency**.

### Nginx

The configuation file is down below:

```bash
server {
    listen 80;
    server_name localhost;

    location /classify {
        proxy_pass http://python:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /classify_url {
        proxy_pass http://python:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

Nginx does reverse proxy and will distribute different requests to different Web server.

This could achieve load balance and improve the ability of concurrency.

### Docker

Docker compose file:

```python
version: '3'
services:
  python:
    build:
      context: .
      dockerfile: docker/python/Dockerfile
    volumes:
      - .:/predict_app
    ports:
      - 8000:8000
      - 8001:8001
    command:
      - /bin/sh
      - -c
      - |
        gunicorn -w 4 app_image:app -b 0.0.0.0:8001 -k gevent -D
        gunicorn -w 4 app_url:app -b 0.0.0.0:8000 -k gevent

  nginx:
    build:
      context: .
      dockerfile: docker/nginx/Dockerfile
    depends_on:
      - python
    ports:
    - 80:80
```

By dockerize the app, we can easily deploy and scale the system.

### Scalability

![image-20181222224337098](https://raw.githubusercontent.com/Paranoid-kid/Scalable-Emotion-Gender-Classification/master/img/5.png)

Since the Machine Learning Model is not affected by user input. We can easily scale out by adding more node.

By using docker we can duplicate Web container easily.

Then we just need to slightly change the Nginx configuration file(e.g. host, port).

## requirements

```
Flask==1.0.2
Keras==2.2.4
matplotlib==3.0.2
numpy==1.15.4
opencv-python==3.4.4.19
pandas==0.23.4
Pillow==5.3.0
requests==2.21.0
scipy==1.2.0
telepot==12.7
tensorflow==1.12.0
gunicorn==19.9.0
gevent==1.3.7
```

## bot.py

Receive requests and forward them to Nginx.

## app_url.py

Receive image URL sent from the user and do prediction. 

Then send the resulting image back to the user.

## app_image.py

Receive image file_id(stored in telegram server) and do prediction. 

Then send the resulting image back to the user.

## docker/

This folder contains the Dockerfile and nginx conf file.

```
.
├── nginx
│   ├── Dockerfile
│   └── pred_nginx.conf
└── python
    └── Dockerfile
```

## pic_server_side/

This folder is used to store the image that Flask received.

Image name is: chat_id + time_stamp + random integer

## result/

This folder is used to store the image after predicted. 

The image name is the same as the image in pic_server_side folder.

## trained_models/

This folder contains the Machine Learning model needed to do the task.

## utils/

This folder contains some tools to run the Machine Learning model.

It's written by the model owner.

## Run the program:

**To run the project, docker environment is required.**

First, build the project, type in the command line:

```bash
$ docker-composer -f prod.yml build
```

Then run:

```bash
$ docker-composer -f prod.yml up
```

Or run in the back stage:

```bash
$ docker-composer -f prod.yml up -d
```

Then run bot.py to receive the image from telegram.

```bash
$ python bot.py
```

Open Telegram and search the bot  "iems5780-1155118093" and send some images :)