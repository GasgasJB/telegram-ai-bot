import ast
import json
import logging
import operator
import os
import sqlite3
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
DB_PATH = os.getenv("DB_PATH", "conversations.db")

SYSTEM_PROMPT = (
    "Eres un asistente inteligente, cercano y con sentido del humor. "
    "Respondes siempre en el mismo idioma que el usuario. "
    "Eres directo y conciso: das respuestas útiles sin rodeos, pero con calidez. "
    "Cuando no sabes algo, lo reconoces con honestidad en lugar de inventar. "
    "Puedes usar emojis con moderación para hacer la conversación más natural. "
    "Tienes acceso a herramientas para consultar el tiempo actual de cualquier ciudad, "
    "obtener la fecha y hora actuales, calcular expresiones matemáticas, "
    "y guardar o recuperar notas personales del usuario. "
    "Cuando una tarea requiera varias herramientas, encadénalas en el mismo turno."
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
    {
        "name": "calculate",
        "description": (
            "Evalúa una expresión matemática de forma segura y devuelve el resultado. "
            "Soporta operaciones básicas (+, -, *, /), potencias (**), módulo (%), "
            "paréntesis y números decimales. "
            "Úsalo cuando el usuario pida calcular, resolver o evaluar una expresión numérica."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Expresión matemática a evaluar, p. ej.: '2 + 2', '(3.5 * 4) / 2', '2 ** 10'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "save_note",
        "description": (
            "Guarda una nota personal del usuario en la base de datos. "
            "Úsalo cuando el usuario quiera apuntar, recordar o guardar algo para más tarde."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Título breve de la nota",
                },
                "content": {
                    "type": "string",
                    "description": "Contenido completo de la nota",
                },
            },
            "required": ["title", "content"],
        },
    },
    {
        "name": "get_notes",
        "description": (
            "Recupera las notas personales guardadas del usuario. "
            "Úsalo cuando el usuario pida ver, listar o recuperar sus notas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "search": {
                    "type": "string",
                    "description": "Término opcional para filtrar notas por título o contenido",
                }
            },
            "required": [],
        },
    },
]

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Per-user conversation history: {user_id: [{"role": ..., "content": ...}]}
conversation_history: dict[int, list[dict]] = {}

# SQLite connection (asyncio is single-threaded so check_same_thread=False is safe)
db: sqlite3.Connection


# --- Database helpers ---

def _init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            role      TEXT    NOT NULL,
            content   TEXT    NOT NULL,
            created_at TEXT   NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            title      TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        )
    """)
    conn.commit()
    logger.info("Base de datos inicializada en %s", DB_PATH)
    return conn


def _load_history(user_id: int) -> list[dict]:
    rows = db.execute(
        "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
        (user_id, MAX_HISTORY),
    ).fetchall()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def _save_message(user_id: int, role: str, content: str) -> None:
    db.execute(
        "INSERT INTO messages (user_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (user_id, role, content, datetime.now().isoformat()),
    )
    db.commit()


def _delete_history(user_id: int) -> None:
    db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    db.commit()


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


# Nodos AST permitidos en expresiones matemáticas (no eval arbitrario)
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Operación no permitida: {ast.dump(node)}")


def _calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree)
        # Mostrar entero si no hay decimales significativos
        if result == int(result):
            return f"{expression} = {int(result)}"
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: división por cero."
    except Exception as exc:
        return f"No se pudo evaluar la expresión: {exc}"


def _save_note(user_id: int, title: str, content: str) -> str:
    db.execute(
        "INSERT INTO notes (user_id, title, content, created_at) VALUES (?, ?, ?, ?)",
        (user_id, title, content, datetime.now().isoformat()),
    )
    db.commit()
    return f"Nota '{title}' guardada correctamente."


def _get_notes(user_id: int, search: str = "") -> str:
    if search:
        rows = db.execute(
            "SELECT title, content, created_at FROM notes WHERE user_id = ? "
            "AND (title LIKE ? OR content LIKE ?) ORDER BY id DESC",
            (user_id, f"%{search}%", f"%{search}%"),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT title, content, created_at FROM notes WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()

    if not rows:
        return "No tienes notas guardadas." if not search else f"No se encontraron notas con '{search}'."

    parts = []
    for title, content, created_at in rows:
        date = created_at[:10]
        parts.append(f"📌 {title} ({date})\n{content}")
    return "\n\n".join(parts)


def _execute_tool(name: str, tool_input: dict, user_id: int = 0) -> str:
    if name == "get_weather":
        return _get_weather(tool_input["city"])
    if name == "get_current_datetime":
        return _get_current_datetime()
    if name == "calculate":
        return _calculate(tool_input["expression"])
    if name == "save_note":
        return _save_note(user_id, tool_input["title"], tool_input["content"])
    if name == "get_notes":
        return _get_notes(user_id, tool_input.get("search", ""))
    return f"Herramienta '{name}' desconocida."


# --- Telegram handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hola! Soy un asistente impulsado por Claude de Anthropic.\n"
        "Escríbeme cualquier mensaje y te responderé.\n\n"
        "Herramientas disponibles:\n"
        "🌤️ Consultar el tiempo de cualquier ciudad\n"
        "🕐 Fecha y hora actuales\n"
        "🧮 Calcular expresiones matemáticas\n"
        "📝 Guardar y recuperar notas personales\n\n"
        "Comandos disponibles:\n"
        "/start - Mostrar este mensaje\n"
        "/clear - Borrar el historial de conversación"
    )


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    _delete_history(user_id)
    conversation_history.pop(user_id, None)
    await update.message.reply_text("Historial de conversación borrado.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_id = update.effective_user.id
    user_text = update.message.text

    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Load history from DB on first contact this session
    if user_id not in conversation_history:
        conversation_history[user_id] = _load_history(user_id)
        logger.info("Historial cargado para user_id=%d (%d mensajes)", user_id, len(conversation_history[user_id]))

    history = conversation_history[user_id]
    history.append({"role": "user", "content": user_text})
    _save_message(user_id, "user", user_text)

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
                        result = _execute_tool(block.name, block.input, user_id)
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
        _save_message(user_id, "assistant", assistant_text)
        await update.message.reply_text(assistant_text)

    except anthropic.APIError as e:
        logger.error("Anthropic API error: %s", e)
        # Remove the user message that failed so history stays consistent
        history.pop()
        _delete_history(user_id)
        # Repopulate DB from the in-memory history that we rolled back
        for msg in history:
            _save_message(user_id, msg["role"], msg["content"])
        await update.message.reply_text(
            "Lo siento, hubo un error al contactar con la API de Claude. Inténtalo de nuevo."
        )


def main() -> None:
    global db
    db = _init_db()

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot iniciado. Esperando mensajes...")
    app.run_polling()


if __name__ == "__main__":
    main()
