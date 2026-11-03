"""Optional LLM step: the rules do the editing; a model (the user's own API key) fixes mis-heard lyrics/captions and
writes the post caption. Provider/key/model come from env (set by the app from Settings):
RS_PROVIDER = anthropic | openai | none, RS_API_KEY, RS_MODEL, RS_BASE_URL (OpenAI-compatible endpoint)."""
import os, json, urllib.request

CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"]
FALLBACK_OK = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}  # server-side refusal fallback


class LLMError(Exception):
    pass


def available():
    return os.environ.get("RS_PROVIDER", "none") in ("anthropic", "openai") and bool(os.environ.get("RS_API_KEY"))


def ask(prompt, system="", schema=None, max_tokens=4000):
    provider = os.environ.get("RS_PROVIDER", "none")
    key = os.environ.get("RS_API_KEY", "")
    if provider == "anthropic":
        import anthropic
        model = os.environ.get("RS_MODEL") or "claude-opus-5-5"
        client = anthropic.Anthropic(api_key=key)
        kw = dict(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
                  output_config={"effort": "low", **({"format": {"type": "json_schema", "schema": schema}} if schema else {})})
        if system: kw["system"] = system
        try:
            if model in FALLBACK_OK:
                r = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
            else:
                r = client.messages.create(**kw)
        except anthropic.AuthenticationError:
            raise LLMError("The API key was rejected. Check it in Settings.")
        except anthropic.NotFoundError:
            raise LLMError(f"Model '{model}' was not found for this key.")
        except anthropic.RateLimitError:
            raise LLMError("Rate limited by the provider. Try again in a minute.")
        except anthropic.APIStatusError as e:
            raise LLMError(f"Provider error {e.status_code}: {e.message}")
        except anthropic.APIConnectionError:
            raise LLMError("Could not reach the provider. Check the internet connection.")
        if r.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        text = "".join(b.text for b in r.content if b.type == "text")
    elif provider == "openai":
        base = (os.environ.get("RS_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        model = os.environ.get("RS_MODEL", "")
        if not model: raise LLMError("Set a model name in Settings.")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt + (
            "\n\nReply with JSON only." if schema else "")}]
        req = urllib.request.Request(base + "/chat/completions", data=json.dumps({"model": model, "messages": msgs}).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                text = json.load(resp)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            raise LLMError(f"Provider error {e.code}: {e.read()[:300].decode(errors='ignore')}")
        except urllib.error.URLError:
            raise LLMError("Could not reach the provider. Check the base URL / internet.")
    else:
        raise LLMError("No AI provider set (Settings).")
    if schema:
        t = text.strip()
        if t.startswith("```"): t = t.strip("`").split("\n", 1)[1] if "\n" in t else t
        try:
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except ValueError:
            raise LLMError("The model's reply was not valid JSON.")
    return text.strip()


def fix_lines(template, song, texts):
    """correct speech-recognition mistakes, same number of lines back"""
    what = (f"someone singing a cover of \"{song.get('title')}\" by {song.get('artist')}" if template == "cover" and song.get("title")
            else "someone singing" if template == "cover" else "someone talking to camera for a short video")
    schema = {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
              "required": ["lines"], "additionalProperties": False}
    prompt = (f"These on-screen text lines were auto-transcribed from a recording of {what}. Speech recognition often "
              f"mis-hears sung words. Correct each line to what was actually said or sung. Keep exactly {len(texts)} lines in "
              f"the same order, one output line per input line, sentence case, no added lines, no line numbers. If a line "
              f"is already right, return it unchanged.\n\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts)))
    out = ask(prompt, schema=schema).get("lines", [])
    return out if len(out) == len(texts) else texts


def post_caption(template, song, texts, platform):
    topic = (f"an acoustic cover of \"{song.get('title')}\" by {song.get('artist')}" if template == "cover"
             else "a short talking video. Transcript: " + " ".join(texts)[:3000])
    prompt = (f"Write a {platform} post caption for {topic}. Format: a hook line (can quote one short line of the video), "
              f"1-2 short lines about it, a question to drive comments, then 8-10 relevant hashtags. "
              + ("For covers, credit the original artist and songwriters in one line. " if template == "cover" else "")
              + ("Also give a title under 100 characters first, labelled 'Title:'. " if platform == "YouTube Shorts" else "")
              + "Plain text, no markdown.")
    return ask(prompt)
