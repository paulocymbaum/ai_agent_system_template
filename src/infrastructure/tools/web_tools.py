import httpx
from langchain_core.tools import tool


@tool
async def fetch_url(url: str) -> dict:
    """Fetch content from a URL.

    Args:
        url: The URL to fetch

    Returns:
        Dictionary with status_code and content
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            return {
                "status_code": response.status_code,
                "content": response.text[:1000],  # Limit content
                "headers": dict(response.headers),
            }
    except Exception as e:
        return {"error": str(e)}
