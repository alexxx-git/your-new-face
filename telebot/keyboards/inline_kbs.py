from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder

def ease_link_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text="Мой хабр", url='https://habr.com/ru/users/yakvenalex/')],
        [InlineKeyboardButton(text="Мой Telegram", url='tg://resolve?domain=yakvenalexx')],
        [InlineKeyboardButton(text="Веб приложение", web_app=WebAppInfo(url="https://tg-promo-bot.ru/questions"))]
    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)

def get_inline_kb():
    inline_kb_list = [
        [InlineKeyboardButton(text="Генерировать пользователя", callback_data='get_person')],
        [InlineKeyboardButton(text="На главную", callback_data='back_home')]
    ]
    return InlineKeyboardMarkup(inline_keyboard=inline_kb_list)

def create_qst_inline_kb(questions: dict) -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    # Добавляем кнопки вопросов
    for question_id, question_data in questions.items():
        builder.row(
            InlineKeyboardButton(
                text=question_data.get('qst'),
                callback_data=f'qst_{question_id}'
            )
        )

    builder.row(
        InlineKeyboardButton(
            text='На главную',
            callback_data='back_home'
        )
    )

    builder.adjust(1)
    return builder.as_markup()

def create_age_keyboard(age:int = 20) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
            inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text=f"Возраст: {age}",
                                callback_data="noop"
                            )
                        ],
                        [
                            InlineKeyboardButton(text="<< -10", callback_data="age:-10"),
                            InlineKeyboardButton(text="< -5", callback_data="age:-5"),
                            InlineKeyboardButton(text="+5 >", callback_data="age:+5"),
                            InlineKeyboardButton(text="+10 >>", callback_data="age:+10"),
                        ],
                    ]
                )