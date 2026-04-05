import os
import io
import asyncio
import hashlib
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import discord
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, db
from PyPDF2 import PdfReader


logging.basicConfig(
    level=logging.DEBUG,  # change to INFO after debugging
    format="%(asctime)s - %(levelname)s - %(message)s"
)

GOOGLE_API_KEY = os.getenv("ml_api")
MODEL_ID = os.getenv("ML_MODEL", "gemini-2.5-flash")
DATASET_NODE = "AI_LEARNING"

FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")  
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")            

if not DISCORD_TOKEN or DISCORD_TOKEN == "PASTE_DISCORD_BOT_TOKEN_HERE":
    raise RuntimeError("Set DISCORD_TOKEN in your environment or paste your bot token into main.py.")

if not GOOGLE_API_KEY:
    raise RuntimeError("Set ml_api in your environment.")

if not FIREBASE_CREDENTIALS or not FIREBASE_DB_URL:
    raise RuntimeError("Set FIREBASE_CREDENTIALS and FIREBASE_DB_URL in your environment.")


if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})


intents = discord.Intents.default()
bot = discord.Bot(intents=intents)

class TF2AIApp:
    def __init__(self, bot_instance: discord.Bot):
        self.bot = bot_instance
        self.api_key = GOOGLE_API_KEY
        self.model_id = MODEL_ID
        self.dataset_node = DATASET_NODE

        self.message_cache = None
        self.message_dict = {}
        self.history_text = ""

        self.executor = ThreadPoolExecutor(max_workers=6)
        self.tasks = []

        genai.configure(api_key=self.api_key)
        logging.info("ml_api present? %s; using model id: %s", bool(self.api_key), self.model_id)

        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_id,
                generation_config={
                    "temperature": 1.0,
                    "top_p": 0.8,
                    "top_k": 50,
                    "max_output_tokens": 1024,
                    "response_mime_type": "text/plain",
                }
            )
        except Exception:
            logging.exception("Exception while creating GenerativeModel (check model id / SDK version).")
            self.model = None

    def create_task_with_logging(self, coro, name=None):
        task = asyncio.create_task(coro, name=name)

        def _on_done(t: asyncio.Task):
            try:
                exc = t.exception()
                if exc:
                    logging.error(
                        "Background task %s raised an exception:\n%s",
                        name or t.get_name(),
                        "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                    )
            except asyncio.CancelledError:
                logging.info("Background task %s was cancelled", name or t.get_name())
            except Exception:
                logging.exception("Error while inspecting background task %s", name or t.get_name())

        task.add_done_callback(_on_done)
        self.tasks.append(task)
        return task

    async def start_background_tasks(self):
        self.create_task_with_logging(self.preload_cache(), name="preload_cache")
        self.create_task_with_logging(self.refresh_cache_periodically(), name="refresh_cache")

    async def preload_cache(self):
        try:
            await self.bot.wait_until_ready()
            await self.get_message_history(force_refresh=True)
            logging.info("Preload cache complete.")
        except Exception:
            logging.exception("Exception during preload_cache")

    async def get_message_history(self, force_refresh=False):
        if self.message_cache is not None and not force_refresh:
            return self.message_cache

        try:
            ref = db.reference(self.dataset_node)
            start = time.perf_counter()

            loop = asyncio.get_running_loop()
            messages = await loop.run_in_executor(self.executor, ref.get)

            elapsed = time.perf_counter() - start
            logging.debug("Firebase get() took %.4f seconds", elapsed)

            if messages:
                self.message_cache = [{"role": "user", "parts": [v["message"]]} for v in messages.values()]
                self.message_dict = {
                    hashlib.sha256(v["message"].encode("utf-8")).hexdigest(): v["message"]
                    for v in messages.values()
                }
            else:
                self.message_cache = []
                self.message_dict = {}

            self.history_text = "\n".join([entry["parts"][0] for entry in self.message_cache])
            return self.message_cache

        except Exception:
            logging.exception("Error fetching message history from Firebase")
            self.message_cache = []
            self.message_dict = {}
            self.history_text = ""
            return self.message_cache

    async def refresh_cache_periodically(self, interval: int = 300):
        while True:
            await asyncio.sleep(interval)
            try:
                await self.get_message_history(force_refresh=True)
                logging.info("Cache refreshed from Firebase.")
            except Exception:
                logging.exception("Exception during periodic cache refresh")

    async def clear_cache(self):
        self.message_cache = None
        self.message_dict = {}
        self.history_text = ""
        logging.info("Message cache cleared.")

    async def learn_from_message(self, message_content: str):
        try:
            message_hash = hashlib.sha256(message_content.encode("utf-8")).hexdigest()

            if message_hash in self.message_dict:
                logging.debug("Duplicate message (local) - skipping insert.")
                return

            ref = db.reference(self.dataset_node)

            loop = asyncio.get_running_loop()

            start = time.perf_counter()
            existing = await loop.run_in_executor(self.executor, ref.child(message_hash).get)
            elapsed = time.perf_counter() - start
            logging.debug("Firebase child.get() took %.4f seconds", elapsed)

            if existing:
                logging.debug("Duplicate message found in Firebase - skipping insert.")
                self.message_dict[message_hash] = message_content
                return

            start = time.perf_counter()
            await loop.run_in_executor(
                self.executor,
                ref.child(message_hash).set,
                {"message": message_content, "status": "learned"}
            )
            elapsed = time.perf_counter() - start
            logging.debug("Firebase set() took %.4f seconds", elapsed)

            self.message_cache = self.message_cache or []
            self.message_cache.append({"role": "user", "parts": [message_content]})
            self.message_dict[message_hash] = message_content

            if self.history_text:
                self.history_text += "\n" + message_content
            else:
                self.history_text = message_content

            logging.info("New message appended to local cache.")

        except Exception:
            logging.exception("Exception in learn_from_message")

    def _generate_ai_response(self, prompt: str, chat_history):
        if not self.model:
            raise RuntimeError("Generative model not initialized.")
        session = self.model.start_chat(history=chat_history)
        return session.send_message(prompt)

    async def send_to_google_ai(self, prompt: str, history_text: str):
        if not self.api_key:
            logging.error("Missing ml_api environment variable.")
            return "AI API key missing (ml_api). Please set the environment variable."

        MAX_HISTORY_CHARS = 30_000
        if history_text and len(history_text) > MAX_HISTORY_CHARS:
            history_text = history_text[-MAX_HISTORY_CHARS:]
            logging.debug("Truncated history_text to last %d chars", MAX_HISTORY_CHARS)

        full_prompt = f"Context:\n{history_text}\n\nUser: {prompt}\nAI:"
        logging.info(
            "Sending prompt to Google AI (preview): %s",
            (full_prompt[:300] + "...") if len(full_prompt) > 300 else full_prompt
        )

        try:
            start = time.perf_counter()
            loop = asyncio.get_running_loop()

            coro = loop.run_in_executor(
                self.executor,
                self._generate_ai_response,
                full_prompt,
                []
            )

            response = await asyncio.wait_for(coro, timeout=60.0)
            elapsed = time.perf_counter() - start
            logging.debug("Model call took %.4f seconds", elapsed)

            logging.info("Model response type: %s", type(response))
            try:
                logging.debug("repr(response) truncated: %s", repr(response)[:1000])
            except Exception:
                logging.debug("Couldn't repr(response)")

            text = None

            if hasattr(response, "text"):
                text = response.text
            elif hasattr(response, "result"):
                text = response.result
            elif hasattr(response, "content"):
                text = response.content
            elif isinstance(response, dict):
                cands = response.get("candidates") or response.get("outputs") or response.get("choices")
                if isinstance(cands, list) and cands:
                    first = cands[0]
                    if isinstance(first, dict):
                        text = first.get("content") or first.get("text") or first.get("message") or first.get("output")
                if not text:
                    text = response.get("output") or response.get("text")

            if not text:
                logging.warning("Could not find textual field in model response; falling back to str(response).")
                text = str(response)

            if not isinstance(text, str):
                text = str(text)

            return text if text else "I didn't quite catch that."

        except asyncio.TimeoutError:
            logging.exception("Timeout while waiting for Google AI response")
            return "The AI took too long to respond (timeout)."
        except Exception:
            logging.exception("Error while calling Google AI")
            return "There was an error processing your request (check bot logs for details)."

    def read_pdf(self, file_content: bytes) -> str:
        try:
            start = time.perf_counter()
            reader = PdfReader(io.BytesIO(file_content))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(pages)
            elapsed = time.perf_counter() - start
            logging.debug("read_pdf processing time: %.4f seconds", elapsed)
            return text
        except Exception:
            logging.exception("Exception while reading PDF")
            return ""


app_state = TF2AIApp(bot)


@bot.event
async def on_ready():
    logging.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "unknown")

    if not getattr(bot, "_tf2ai_tasks_started", False):
        bot._tf2ai_tasks_started = True
        await app_state.start_background_tasks()
        logging.info("Background tasks started.")


@bot.slash_command(name="maxwell", description="Ask the AI a question.")
async def maxwell(ctx, user_input: str):
    overall_start = time.perf_counter()

    try:
        await ctx.defer(ephemeral=False)
    except discord.NotFound:
        logging.warning("Interaction expired before defer() could run.")
        try:
            await ctx.respond("Sorry but john is really dumb")
        except Exception:
            pass
        return
    except Exception:
        logging.exception("Unexpected error during ctx.defer()")
        try:
            await ctx.respond("Failed to acknowledge interaction (check bot logs).")
        except Exception:
            pass
        return

    try:
        await app_state.get_message_history()
        history_text = app_state.history_text

        response_text = await app_state.send_to_google_ai(user_input, history_text)

        safe_text = response_text if len(response_text) <= 4000 else (response_text[:3997] + "...")
        embed = discord.Embed(title="AI Response", description=safe_text, color=discord.Color.blue())

        try:
            await ctx.followup.send(embed=embed)
        except discord.NotFound:
            logging.warning("Interaction expired before followup.send() could run.")
        except Exception:
            logging.exception("Error while sending response to Discord")

    except Exception:
        logging.exception("Unhandled exception in command")
        try:
            await ctx.followup.send("An error occurred while processing your request (check bot logs).")
        except Exception:
            pass
    finally:
        overall_elapsed = time.perf_counter() - overall_start
        logging.info("maxwell overall processing time: %.4f seconds", overall_elapsed)


@bot.slash_command(name="feed_dataset", description="Feed text or a file into the dataset.")
async def feed_dataset(ctx, file: discord.Attachment = None, text_input: str = None):
    overall_start = time.perf_counter()

    if not file and not text_input:
        await ctx.respond("Please provide a file or text input.")
        return

    try:
        await ctx.defer(ephemeral=True)
    except Exception:
        logging.exception("Failed to defer feed_dataset interaction")
        try:
            await ctx.respond("Failed to acknowledge interaction.")
        except Exception:
            pass
        return

    try:
        if file:
            start = time.perf_counter()
            file_content = await file.read()
            elapsed = time.perf_counter() - start
            logging.debug("Reading file content took %.4f seconds for %s", elapsed, file.filename)

            if file.filename.lower().endswith(".pdf"):
                start = time.perf_counter()
                loop = asyncio.get_running_loop()
                text = await loop.run_in_executor(app_state.executor, app_state.read_pdf, file_content)
                elapsed = time.perf_counter() - start
                logging.debug("PDF processing took %.4f seconds", elapsed)
            elif file.filename.lower().endswith(".txt"):
                text = file_content.decode("utf-8", errors="replace")
            else:
                await ctx.followup.send("Upload a valid PDF or TXT file.")
                return

            await app_state.learn_from_message(text)
            await ctx.followup.send(f"Processed and stored dataset from {file.filename}.")

        elif text_input:
            await app_state.learn_from_message(text_input)
            await ctx.followup.send("Processed and stored your text input.")

    except Exception:
        logging.exception("Exception while processing feed_dataset")
        try:
            await ctx.followup.send("An error occurred while processing the dataset (check bot logs).")
        except Exception:
            pass
    finally:
        overall_elapsed = time.perf_counter() - overall_start
        logging.info("feed_dataset processing time: %.4f seconds", overall_elapsed)


if __name__ == "__main__":
    bot.run("MTQ5MDIzMjkzMjExNDIzOTc0MA.GAfInQ.NvqOY-gE74vz1ZlSN-cYiCbDC0D7FGrDB1ihU4")
