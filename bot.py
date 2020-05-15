# https://github.com/tejonaco/facepaster

from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
from facepaster import FacePaster, BadFaceError
from token_ import TOKEN
from pass_ import pass_md5
from hashlib import md5
import pickle
from time import sleep
from io import BytesIO
import requests



import logging
# logging.basicConfig(level=logging.WARNING, filename='logs.log', format='%(asctime)s - %(message)s')
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(message)s')



# load users to avoid lost of data in every reboot
try:
    with open('users', 'rb') as f:
        users = pickle.load(f) # dict: {user_id: last_input_face_url}
except FileNotFoundError:
    users = {}


def access(f):
    def wraper(update, context, *args, **kwargs):
        id_ = update.message.chat.id
        if id_ in users.keys() or not pass_md5:
            return f(update, context, *args, **kwargs)
        else:
            update.message.reply_text('Primero debes introducir la contraseña')
    return wraper


def start(update, context):
    id_ = update.message.chat.id
    if id_ in users:
        update.message.reply_text('Comienza cuando quieras, enviando una cara de origen con /entrada o una foto para editar')
    else:
        update.message.reply_text('Bienvenido, para comenzar introduce la contraseña.')


def password(update, context):
    id_ = update.message.chat.id
    if id_ in users.keys() or not pass_md5:
        update.message.reply_text('Envía una imagen para continuar')
    else:
        text = update.message.text
        if md5(text.encode()).hexdigest() == pass_md5:
            users[id_] = 'new'
            with open('users', 'wb') as f: #update the persistent database
                pickle.dump(users, f)
            update.message.reply_text('Contraseña correcta, envia la cara a colocar en las fotos con el comando /entrada')
        else:
            sleep(1) # little protection against brute force attacks
            update.message.reply_text('Contraseña incorrecta')

@access
def input_img(update, context):
    id_ = update.message.chat.id
    users[id_] = 'new'
    update.message.reply_text('Envia la cara a colocar en las fotos *como archivo*', parse_mode='markdown')


@access
def get_face(update, context):
    id_ = update.message.chat.id
    if users[id_] == 'new':
        img_url = context.bot.get_file(update.message.document.file_id).file_path
        users[id_] = img_url #only save the url to save space
        with open('users', 'wb') as f: #update the persistent database
            pickle.dump(users, f)
        update.message.reply_text('Vale, ahora envia la imagen a editar, esta vez no hay que enviarla como archivo')


@access
def get_img(update, context):
    id_ = update.message.chat.id

    img_url = context.bot.get_file(update.message.photo[-1].file_id).file_path

    if users[id_] != 'new':
        # get the img and convert to stream
        img_content = requests.get(img_url).content
        img = BytesIO(img_content)

        # convert the face to stream
        face_content = requests.get(users[id_]).content
        face = BytesIO(face_content)

        # use the facepaster library to generate the output handling exceptions
        try:
            with FacePaster(face) as fp:
                output_pil = fp.paste_faces(img)
        except ValueError:
            update.message.reply_text('Parece que la imagen de /entrada no es correcta. \
            La has subido como archivo?')
            return
        except BadFaceError:
            update.message.reply_text('Algo ha salido mal, revisa la imagen de /entrada')
            return
        
        output = BytesIO()
        output_pil.save(output, 'JPEG')
        output.seek(0)
        context.bot.send_photo(id_, photo=output)
    else:
        update.message.reply_text('Envia la cara a colocar en las fotos *como archivo*', parse_mode='markdown')





# create updater and add handlers
updater = Updater(TOKEN, use_context=True)
updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('entrada', input_img))
updater.dispatcher.add_handler(MessageHandler(Filters.text, password))
updater.dispatcher.add_handler(MessageHandler(Filters.photo, get_img))
updater.dispatcher.add_handler(MessageHandler(Filters.document, get_face))

# start bot
updater.start_polling()
updater.idle() # allows ctrl+c to stop
