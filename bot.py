import json
import logging
import os
import urllib.parse
import urllib.request
from datetime import datetime

import anthropic
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

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
    "Puedes usar emojis con moderación para hacer la conversación más natural. "
    "Tienes acceso a herramientas para consultar el tiempo actual de cualquier ciudad "
    "y para obtener la fecha y hora actuales; úsalas cuando el usuario las necesite."
)

TOOLS = [
    {
        "name": "get_weather",
        "description": (
            "Obtiene el tiempo meteorológico actual de una ciudad usando wttr.in. "
            "Úsalo cuando el usuario pregunte por el tiempo, temperatura, lluvia, "
            "nieve, humedad u otras condiciones climáticas de cualquier ciudad o lugar."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Nombre de la ciudad, p. ej.: Madrid, London, New York",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_current_datetime",
        "description": (
            "Devuelve la fecha y hora actuales del sistema. "
            "Úsalo cuando el usuario pregunte qué hora es, qué día es, "
            "la fecha de hoy o cualquier consulta sobre el momento presente."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Per-user conversation history: {user_id: [{"role": ..., "content": ...}]}
conversation_history: dict[int, list[dict]] = {}


# --- Tool implementations ---

_DAYS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
_MONTHS_ES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]


def _get_weather(city: str) -> str:
    encoded = urllib.parse.quote(city)
    url = f"https://wttr.in/{encoded}?format=j1"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        current = data["current_condition"][0]
        area = data["nearest_area"][0]["areaName"][0]["value"]
        country = data["nearest_area"][0]["country"][0]["value"]
        desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        return (
            f"Tiempo en {area}, {country}:\n"
            f"- Condición: {desc}\n"
            f"- Temperatura: {temp_c}°C (sensación térmica: {feels_like}°C)\n"
            f"- Humedad: {humidity}%"
        )
    except Exception as exc:
        logger.warning("Weather fetch failed for %r: %s", city, exc)
        return f"No se pudo obtener el tiempo para '{city}'. Comprueba que el nombre de la ciudad es correcto."


def _get_current_datetime() -> str:
    now = datetime.now()
    day_name = _DAYS_ES[now.weekday()]
    month_name = _MONTHS_ES[now.month - 1]
    return (
        f"Fecha: {day_name}, {now.day} de {month_name} de {now.year}\n"
        f"Hora: {now.strftime('%H:%M:%S')}"
    )


def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "get_weather":
        return _get_weather(tool_input["city"])
    if name == "get_current_datetime":
        return _get_current_datetime()
    return f"Herramienta '{name}' desconocida."


# --- Telegram handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hola! Soy un asistente impulsado por Claude de Anthropic.\n"
        "Escríbeme cualquier mensaje y te responderé.\n\n"
        "También puedo consultar el tiempo de cualquier ciudad 🌤️ "
        "y decirte la fecha y hora actuales 🕐.\n\n"
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

    if len(history) > MAX_HISTORY:
        conversation_history[user_id] = history[-MAX_HISTORY:]
        history = conversation_history[user_id]

    # Working copy for the agentic loop; history only gets the final text.
    messages = list(history)

    try:
        assistant_text = ""
        while True:
            response = claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                # Append assistant turn (may contain text + tool_use blocks)
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                        logger.info("Calling tool %r with input %s", block.name, block.input)
                        result = _execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})
                continue  # send results back to Claude

            # end_turn or any other stop reason — extract text and finish
            assistant_text = next(
                (b.text for b in response.content if b.type == "text"),
                "Lo siento, no pude generar una respuesta.",
            )
            break

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
