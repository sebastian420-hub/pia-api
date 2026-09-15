"""Singapore LTA rotates image URLs on every refresh; resolve the current one by camera id."""
LIST_URL = "https://api.data.gov.sg/v1/transport/traffic-images"


async def resolve_singapore_image(client, camera_id: str) -> str:
    r = await client.get(LIST_URL, timeout=15)
    r.raise_for_status()
    items = r.json().get("items") or []
    for cam in (items[0].get("cameras") if items else []) or []:
        if str(cam.get("camera_id")) == str(camera_id):
            return cam.get("image") or ""
    return ""
