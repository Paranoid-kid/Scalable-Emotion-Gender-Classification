import time
import telepot
from telepot.loop import MessageLoop
import logging.config
import requests
from queue import Queue
import threading

logging.config.fileConfig('logconfig.ini')
IMG_PATH = 'pic/'
url_classify = 'http://127.0.0.1:81/classify'
url_classify_url = 'http://127.0.0.1:81/classify_url'


#  Thread 1 ­ Receiving Messages
def handle(msg):
    start = time.time()
    content_type, chat_type, chat_id = telepot.glance(msg)
    logging.info('----------Got Connection From User [{}]-------'.format(chat_id))
    logging.debug('content_type is: {}'.format(content_type))
    logging.debug('chat_id is: {}'.format(chat_id))
    message = 'The prediction is on the way, please hold on for a sec.'

    if content_type == 'text':
        image_url = msg['text']
        if image_url.startswith('http://') or image_url.startswith('https://'):
            logging.info('----------Received URL of Image----------')
            logging.debug('Image URL is: {}'.format(image_url))
            wrapped_msg = {
                'type': 'url',
                'url': image_url,
                'chat_id': chat_id,
            }
            queue_1.put(wrapped_msg)
            bot.sendMessage(chat_id, message)

    if content_type == 'photo':
        logging.info('----------Download Image From Telegram----------')
        file_id = msg['photo'][-1]['file_id']
        wrapped_msg = {
            'type': 'file_id',
            'file_id': file_id,
            'chat_id': chat_id,
        }
        queue_1.put(wrapped_msg)
        bot.sendMessage(chat_id, message)
        end = time.time()
        logging.info('[Time] for all process(): {}'.format(end - start))


#  Thread 2 ­ Client Thread
def send_recv_img(in_queue_1):
    while True:
        wrapped_msg = in_queue_1.get()
        if wrapped_msg['type'] == 'url':
            requests.post(url_classify_url, json=wrapped_msg)
            logging.info('Image has been sent to Server')

        if wrapped_msg['type'] == 'file_id':
            requests.post(url_classify, json=wrapped_msg)
            logging.info('Image has been sent to Server')


if __name__ == "__main__":
    bot = telepot.Bot('Input your bot key')
    queue_1 = Queue()
    MessageLoop(bot, handle).run_as_thread()
    logging.debug('thread %s is running...' % threading.current_thread().name)
    t2 = threading.Thread(target=send_recv_img, args=(queue_1,), name='Thread 2')
    t2.start()
