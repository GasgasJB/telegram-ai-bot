import os
import logging
from dotenv import load_dotenv
import anthropic
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ChatAction

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "20"))

SYSTEM_PROMPT = (
    "Eres un asistente inteligente, cercano y con sentido del humor. "
    "Respondes siempre en el mismo idioma que el usuario. "
    "Eres directo y conciso: das respuestas útiles sin rodeos, pero con calidez. "
    "Cuando no sabes algo, lo reconoces con honestidad en lugar de inventar. "
    "Puedes usar emojis con moderación para hacer la conversación más natural."
)

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Per-user conversation history: {user_id: [{"role": ..., "content": ...}]}
conversation_history: dict[int, list[dict]] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hola! Soy un asistente impulsado por Claude de Anthropic.\n"
        "Escríbeme cualquier mensaje y te responderé.\n\n"
        "Comandos disponibles:\n"
        "/start - Mostrar este mensaje\n"
        "/clear - Borrar el historial de conversación"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    conversation_history.pop(user_id, None)
    await update.message.reply_text("Historial de conversación borrado.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    history = conversation_history.setdefault(user_id, [])
    history.append({"role": "user", "content": user_text})

    # Keep only the last MAX_HISTORY messages (must stay even to preserve role alternation)
    if len(history) > MAX_HISTORY:
        conversation_history[user_id] = history[-MAX_HISTORY:]
        history = conversation_history[user_id]

    try:
        response = claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=history,
        )

        assistant_text = response.content[0].text
        history.append({"role": "assistant", "content": assistant_text})

        await update.message.reply_text(assistant_text)

    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e)
        # Remove the user message that failed so history stays consistent
        history.pop()
        await update.message.reply_text(
            "Lo siento, hubo un error al contactar con la API de Claude. Inténtalo de nuevo."
        )


def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot iniciado. Esperando mensajes...")
    app.run_polling()


if __name__ == "__main__":
    main()
