import asyncio
from aiogram import Router, F
from aiogram.filters import  CommandStart, Command, CommandObject
from aiogram.types import Message
from keyboards.all_keyborad import main_kb, create_spec_kb,create_rat, create_foto
from keyboards.inline_kbs import ease_link_kb, get_inline_kb, create_qst_inline_kb
from utils.utils import get_random_person
from aiogram.types import CallbackQuery
from create_bot import questions, bot, admins
from aiogram.utils.chat_action import ChatActionSender
from filters.is_admin import IsAdmin
start_router = Router()

@start_router.message(CommandStart())
async def cmd_start(message: Message, command: CommandObject):
    command_args: str = command.args
    if command_args:
        await message.answer(
            f'Запуск сообщения по команде /start используя фильтр CommandStart() с меткой <b>{command_args}</b>',
            reply_markup=main_kb(message.from_user.id))
    else:
        await message.answer(
            f'Запуск сообщения по команде /start используя фильтр CommandStart() без метки',
            reply_markup=main_kb(message.from_user.id))

@start_router.message(Command('start_2'))
async def cmd_start_2(message: Message):
    await message.answer('Запуск сообщения по команде /start_2 используя фильтр Command()', \
                         reply_markup=create_spec_kb())

@start_router.message(F.text == '/start_3')
async def cmd_start_3(message: Message):
    await message.answer('Запуск сообщения по команде /start_3 используя магический фильтр F.text!', \
                         reply_markup=create_rat())
@start_router.message(F.text=='ДД')
async def get_inline_btk_link(message: Message):
    await message.answer('Вот тебе клавиатура со ссылками!', reply_markup=ease_link_kb())

@start_router.message(F.text=='11')
async def get_inline_btk_link(message: Message):
    await message.answer('Вот тебе клавиатура со ссылками!', reply_markup=get_inline_kb())

@start_router.callback_query(F.data == 'get_person')
async def send_random_person(call: CallbackQuery):
    await call.answer('Генерирую случайного пользователя',show_alert=True)
    user = get_random_person()
    formatted_message = (
        f"👤 <b>Имя:</b> {user['name']}\n"
        f"🏠 <b>Адрес:</b> {user['address']}\n"
        f"📧 <b>Email:</b> {user['email']}\n"
        f"📞 <b>Телефон:</b> {user['phone_number']}\n"
        f"🎂 <b>Дата рождения:</b> {user['birth_date']}\n"
        f"🏢 <b>Компания:</b> {user['company']}\n"
        f"💼 <b>Должность:</b> {user['job']}\n"
    )
    await call.message.answer(formatted_message)



@start_router.callback_query(F.data == 'back_home')
async def back_home(call: CallbackQuery):
    await call.message.delete_reply_markup()
    # await call.message.edit_reply_markup(reply_markup=None) #удаление кнопок
    # await call.message.answer()
@start_router.message(Command('faq'))
async def cmd_start_2(message: Message):
    await message.answer('Сообщение с инлайн клавиатурой с вопросами', reply_markup=create_qst_inline_kb(questions))


@start_router.callback_query(F.data.startswith('qst_'))
async def cmd_start(call: CallbackQuery):
    await call.answer()
    qst_id = int(call.data.replace('qst_', ''))
    qst_data = questions[qst_id]
    msg_text = f'Ответ на вопрос {qst_data.get("qst")}\n\n' \
               f'<b>{qst_data.get("answer")}</b>\n\n' \
               f'Выбери другой вопрос:'
    async with ChatActionSender(bot=bot, chat_id=call.from_user.id, action="typing"):
        await asyncio.sleep(1)
        await call.message.delete_reply_markup()
        await call.message.answer(msg_text, reply_markup=create_qst_inline_kb(questions))



@start_router.message(F.text.lower().contains('qq'), IsAdmin(admins))
async def process_find_word(message: Message):
    await message.answer('О, админ, здарова! А тебе можно писать подписывайся.')


@start_router.message(Command("www"))
async def start_command(message: Message):

    await message.answer(
        "Пожалуйста, сфотографируйте объект и отправьте фото в этот чат.",
        reply_markup=create_foto()
    )

# Хендлер, который ловит отправленное фото
@start_router.message(F.photo)
async def handle_photo(message: Message):
    await message.answer("Спасибо! Фото получено.")