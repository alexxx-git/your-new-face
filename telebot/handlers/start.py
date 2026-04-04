import asyncio
from aiogram import Bot
from aiogram import Router, F
from aiogram.filters import  CommandStart, Command, CommandObject
from aiogram.types import Message
from telebot.keyboards.all_keyborad import main_kb, create_spec_kb,create_rat, create_foto
from telebot.keyboards.inline_kbs import ease_link_kb, get_inline_kb, create_qst_inline_kb, create_age_keyboard
from telebot.utils.utils import get_random_person
from aiogram.types import CallbackQuery
from telebot.create_bot import bot, admins
from aiogram.utils.chat_action import ChatActionSender
from telebot.filters.is_admin import IsAdmin
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import os


start_router = Router()


@start_router.message(Command('Qq'))
async def cmd_start_2(message: Message):
    await message.answer('Выберите кнопками возраст от 20 до 100 лет, после выбора нажмите на кнопку возраст', \
                         reply_markup=create_age_keyboard())

@start_router.callback_query(F.data.startswith('age:'))
async def cmd_start(call: CallbackQuery,state: FSMContext):
    await call.answer()
    delta = int(call.data.split(":")[1])
    data = await state.get_data()
    old_age = data.get("age", 20)
    # if age is None:
    #     age = 20  
    #     await state.update_data(age=age)

    age = max(20, min(old_age + delta, 100))
    if age == old_age:
        # ничего не менялось — НЕ редактируем сообщение
        return

    await state.update_data(age=age)

    # async with ChatActionSender(bot=bot, chat_id=call.from_user.id, action="typing"):
    await call.message.edit_reply_markup(
        reply_markup=create_age_keyboard(age)
    )


@start_router.message(CommandStart())
async def cmd_start(message: Message, command: CommandObject):
    await message.answer(f"{Greating})'

@start_router.message(F.text=='11')
async def get_inline_btk_link(message: Message):
    await message.answer('Вот тебе клавиатура со ссылками!', reply_markup=get_inline_kb())

@start_router.callback_query(F.data == 'back_home')
async def back_home(call: CallbackQuery):
    await call.message.delete_reply_markup()
    # await call.message.edit_reply_markup(reply_markup=None) #удаление кнопок
    # await call.message.answer()

@start_router.message(Command('faq'))
async def cmd_start_2(message: Message):
    await message.answer('Сообщение с инлайн клавиатурой с вопросами', reply_markup=create_qst_inline_kb(questions))


@start_router.message(F.text.lower().contains('qq'), IsAdmin(admins))
async def process_find_word(message: Message):
    await message.answer('О, админ, здарова! А тебе можно писать подписывайся.')

class WwwState(StatesGroup):
    waiting_photo = State()


@start_router.message(Command("foto"))
async def start_command(message: Message, state:FSMContext):
    await state.set_state(WwwState.waiting_photo)
    await message.answer(
        "Пожалуйста, сфотографируйте объект и отправьте фото в этот чат.",
        reply_markup=create_foto()
    )

# Хендлер, который ловит отправленное фото
@start_router.message(WwwState.waiting_photo)
async def handle_file(message: Message, state: FSMContext, bot: Bot):
    if message.photo:
        file_id = message.photo[-1].file_id
        filename = f"{file_id}.jpg"
    elif message.document and message.document.mime_type.startswith("image/"):
        file_id = message.document.file_id
        filename = message.document.file_name
    else:
        await message.answer("Пожалуйста, отправьте изображение (фото или документ).")
        return
    # 3️⃣ Получаем объект File и скачиваем
    file = await bot.get_file(file_id)
    file_bytes = await bot.download_file(file.file_path)

    # 4️⃣ Сохраняем на диск
    os.makedirs("./downloads", exist_ok=True)
    path = os.path.join("./downloads", filename)
    with open(path, "wb") as f:
        f.write(file_bytes.read())

    await message.answer(f"Файл сохранён: {path}")

    # 5️⃣ Сбрасываем состояние
    await state.clear()

@start_router.callback_query(F.data == "noop")
async def noop_callback(call: CallbackQuery):
    await call.answer()