import re
import httpx
from fastapi import HTTPException
from config import settings

# Persistent client
http_client = httpx.AsyncClient(timeout=60.0)

PROVIDER_URLS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "deepseek": "https://api.deepseek.com/chat/completions",
}

def parse_data_url(data_url: str) -> tuple[str, str]:
    match = re.match(r"data:([^;]+);base64,(.+)", data_url)
    if not match:
        raise ValueError("Invalid data URL")
    return match.group(1), match.group(2)

def convert_messages_for_provider(messages: list, provider: str) -> list:
    result = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "user")
        msg_type = msg.get("type", "text")

        if role == "user" and msg_type == "text":
            next_msg = messages[i + 1] if i + 1 < len(messages) else None
            if next_msg and next_msg.get("role") == "user" and next_msg.get("type") == "image":
                text_content = msg.get("content", "")
                image_content = next_msg.get("content", "")

                if provider == "anthropic":
                    media_type, base64_data = parse_data_url(image_content)
                    content_blocks = [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        },
                    ]
                elif provider == "openrouter":
                    content_blocks = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": {"url": image_content}},
                    ]
                else:  # mistral
                    content_blocks = [
                        {"type": "text", "text": text_content},
                        {"type": "image_url", "image_url": image_content},
                    ]
                result.append({"role": "user", "content": content_blocks})
                i += 2
                continue

        if msg_type == "image":
            image_content = msg.get("content", "")
            if provider == "anthropic":
                media_type, base64_data = parse_data_url(image_content)
                content_blocks = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64_data,
                        },
                    }
                ]
            elif provider == "openrouter":
                content_blocks = [
                    {"type": "image_url", "image_url": {"url": image_content}}
                ]
            else:  # mistral
                content_blocks = [
                    {"type": "image_url", "image_url": image_content}
                ]
            result.append({"role": "user", "content": content_blocks})
        else:
            result.append({"role": role, "content": msg.get("content", "")})
        i += 1

    return result

async def proxy_request(provider: str, model: str, messages: list, max_tokens: int = 1024, system: str | None = None) -> dict:
    api_key = getattr(settings, f"{provider.upper()}_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=400, detail=f"Provider '{provider}' is not configured")

    if provider == "anthropic":
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            body["system"] = system

        resp = await http_client.post(PROVIDER_URLS["anthropic"], headers=headers, json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return {"response": text}

    elif provider == "openrouter":
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        prepared_messages = []
        if system:
            prepared_messages.append({"role": "system", "content": system})
        prepared_messages.extend(messages)

        body = {"model": model, "max_tokens": max_tokens, "messages": prepared_messages}

        resp = await http_client.post(PROVIDER_URLS["openrouter"], headers=headers, json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"response": text}

    elif provider == "deepseek":
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        prepared_messages = []
        if system:
            prepared_messages.append({"role": "system", "content": system})
        prepared_messages.extend(messages)

        body = {"model": model, "max_tokens": max_tokens, "messages": prepared_messages}

        resp = await http_client.post(PROVIDER_URLS["deepseek"], headers=headers, json=body)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)

        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return {"response": text}

    raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
