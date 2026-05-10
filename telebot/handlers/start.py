import asyncio
import aiohttp
from aiogram import Bot
from aiogram import Router, F
from aiogram.filters import CommandStart, Command, CommandObject
from aiogram.types import BufferedInputFile, Message
from telebot.keyboards.all_keyborad import (
    main_kb,
    create_spec_kb,
    create_rat,
    create_foto,
)
from telebot.keyboards.inline_kbs import (
    ease_link_kb,
    get_inline_kb,
    create_qst_inline_kb,
    create_age_keyboard,
)
from telebot.utils.utils import get_random_person
from aiogram.types import CallbackQuery
from telebot.create_bot import bot, admins
from aiogram.utils.chat_action import ChatActionSender
from telebot.filters.is_admin import IsAdmin
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from telebot.text import STARTING_MESSAGE
import os


start_router = Router()
API_BASE_URL = os.getenv("BOT_API_BASE_URL", "http://localhost").rstrip("/")
POLL_INTERVAL_SECONDS = 2
MAX_POLL_ATTEMPTS = 60


@start_router.message(Command("Qq"))
async def cmd_start_2(message: Message):
    await message.answer(
        "Выберите кнопками возраст от 20 до 100 лет, после выбора нажмите на кнопку возраст",
        reply_markup=create_age_keyboard(),
    )


@start_router.callback_query(F.data.startswith("age:"))
async def cmd_start(call: CallbackQuery, state: FSMContext):
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
    await call.message.edit_reply_markup(reply_markup=create_age_keyboard(age))


@start_router.message(CommandStart())
async def cmd_start(message: Message, command: CommandObject):
    await message.answer(STARTING_MESSAGE)


@start_router.message(F.text == "11")
async def get_inline_btk_link(message: Message):
    await message.answer(
        "Вот тебе клавиатура со ссылками!", reply_markup=get_inline_kb()
    )


@start_router.callback_query(F.data == "back_home")
async def back_home(call: CallbackQuery):
    await call.message.delete_reply_markup()
    # await call.message.edit_reply_markup(reply_markup=None) #удаление кнопок
    # await call.message.answer()


@start_router.message(Command("faq"))
async def cmd_start_2(message: Message):
    await message.answer("FAQ пока не настроен.")


@start_router.message(F.text.lower().contains("qq"), IsAdmin(admins))
async def process_find_word(message: Message):
    await message.answer("О, админ, здарова! А тебе можно писать подписывайся.")


class WwwState(StatesGroup):
    waiting_photo = State()


@start_router.message(Command("foto"))
async def start_command(message: Message, state: FSMContext):
    await state.set_state(WwwState.waiting_photo)
    await message.answer(
        "Пожалуйста, сфотографируйте объект и отправьте фото в этот чат.",
        reply_markup=create_foto(),
    )


# Хендлер, который ловит отправленное фото
@start_router.message(WwwState.waiting_photo)
async def handle_file(message: Message, state: FSMContext, bot: Bot):
    if message.photo:
        file_id = message.photo[-1].file_id
        filename = f"{file_id}.jpg"
        content_type = "image/jpeg"
    elif message.document and message.document.mime_type.startswith("image/"):
        file_id = message.document.file_id
        filename = message.document.file_name or f"{file_id}.jpg"
        content_type = message.document.mime_type
    else:
        await message.answer("Пожалуйста, отправьте изображение (фото или документ).")
        return

    await message.answer("Фото получено. Отправляю на обработку...")

    try:
        file = await bot.get_file(file_id)
        downloaded = await bot.download_file(file.file_path)
        image_bytes = downloaded.read()

        async with aiohttp.ClientSession() as session:
            task_id = await _create_processing_task(
                session=session,
                image_bytes=image_bytes,
                filename=filename,
                content_type=content_type,
            )
            await message.answer(f"Задача создана: {task_id}. Жду результат...")

            result_url = await _wait_processing_result(session=session, task_id=task_id)
            result_bytes = await _download_result(session=session, result_url=result_url)

        result_file = BufferedInputFile(
            result_bytes,
            filename=f"mirrored_{filename}",
        )
        await message.answer_photo(result_file, caption="Готово. Фото отзеркалено.")
    except Exception as exc:
        await message.answer(f"Ошибка обработки: {exc}")
        return

    await state.clear()


@start_router.callback_query(F.data == "noop")
async def noop_callback(call: CallbackQuery):
    await call.answer()


async def _create_processing_task(
    session: aiohttp.ClientSession,
    image_bytes: bytes,
    filename: str,
    content_type: str,
) -> str:
    form = aiohttp.FormData()
    form.add_field(
        "file",
        image_bytes,
        filename=filename,
        content_type=content_type,
    )

    async with session.post(f"{API_BASE_URL}/api/tasks/upload", data=form) as response:
        data = await _read_json_response(response)
        if response.status >= 400:
            raise RuntimeError(data.get("detail", "Не удалось создать задачу"))
        return data["task_id"]


async def _wait_processing_result(
    session: aiohttp.ClientSession,
    task_id: str,
) -> str:
    for _ in range(MAX_POLL_ATTEMPTS):
        async with session.get(f"{API_BASE_URL}/api/tasks/{task_id}") as response:
            data = await _read_json_response(response)

        status = data.get("status")
        if status == "SUCCESS":
            return data["result"]["result_url"]
        if status == "FAILURE":
            error = (data.get("result") or {}).get("error", "Worker завершился ошибкой")
            raise RuntimeError(error)

        await asyncio.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError("Время ожидания результата истекло")


async def _download_result(session: aiohttp.ClientSession, result_url: str) -> bytes:
    async with session.get(f"{API_BASE_URL}{result_url}") as response:
        if response.status >= 400:
            raise RuntimeError("Не удалось скачать результат")
        return await response.read()


async def _read_json_response(response: aiohttp.ClientResponse) -> dict:
    try:
        return await response.json()
    except aiohttp.ContentTypeError:
        text = await response.text()
        raise RuntimeError(text[:300])
