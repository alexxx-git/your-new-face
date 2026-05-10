import logging
import os
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from pathlib import Path
from dotenv import load_dotenv


dotenv_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
# tyt nado prorabotat moget smenit na drugue bbilioteku
admins_raw = os.getenv("ADMINS", "")
admins = [int(admin_id) for admin_id in admins_raw.split(",") if admin_id.strip()]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))

dp = Dispatcher(storage=MemoryStorage())
